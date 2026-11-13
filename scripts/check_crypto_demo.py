"""Проверки на реальных байтах; секреты живут только во временном каталоге."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contextlib import contextmanager
from itertools import combinations
import importlib.metadata
import json
import secrets
import tempfile
import time
from unittest.mock import patch

from crypto_demo.storage import (
    Client, LocalAgent, Vault, DemoError, MAX_OBJECT, KDF,
    canonical, parse, encode, decode, digest, b64, un64, atomic, seal,
    writer_lock, open_sealed,
)


def expect(code, operation):
    try:
        operation()
    except DemoError as error:
        assert error.code == code, (code, error.code)
    else:
        raise AssertionError("expected " + code)


@contextmanager
def removed(paths):
    saved = [(p, p.read_bytes()) for p in paths]
    try:
        for p, _ in saved:
            p.unlink()
        yield
    finally:
        for p, raw in saved:
            atomic(p, raw)


def run():
    started = time.perf_counter()
    checks = []
    def ok(message):
        checks.append(message)
        print("OK:", message, flush=True)

    # Coding, independently of encryption / trusted data.
    sample = bytes(range(37))
    counts = {}
    for k, n in [(3, 5), (10, 13)]:
        shards = encode(sample, k, n)
        count = 0
        for dropped in range(n-k+1):
            for missing in combinations(range(n), dropped):
                available = {i: raw for i, raw in enumerate(shards) if i not in missing}
                assert decode(available, k, n, len(sample)) == sample
                count += 1
        counts[f"{k}/{n}"] = count
        expect("INSUFFICIENT_FRAGMENTS", lambda: decode({0: shards[0]}, k, n, len(sample)))
    ok("RS: все допустимые потери 3/5 и 10/13, дополнение и недостаток фрагментов")

    workspace = Path(__file__).resolve().parents[1]
    temporary_parent = workspace / "runs" / "crypto_demo"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    assert temporary_parent.resolve().is_relative_to(workspace.resolve())
    with tempfile.TemporaryDirectory(prefix="check-", dir=temporary_parent) as temp:
        root = Path(temp)
        assert root.resolve().is_relative_to(temporary_parent.resolve())
        password = secrets.token_urlsafe(32)
        client = Client(root, password)
        client.initialize()
        expected = "Демонстрация TARIS: секретный объект.\n".encode() * 12
        head = client.store(expected)
        oid = head["object_id"]
        assert client.read_latest(oid) == expected
        ok("запись, зашифрованный манифест, сборка и расшифрование UTF-8")

        plan = client.repair_plan(oid)
        paths = [client.agent.path(d["blob_id"]) for d in plan["fragments"]]
        with removed(paths[:3]):
            assert client.read_latest(oid) == expected
            # Agent repair uses a secret-free plan and requires no client vault.
            with removed([client.vault_path]):
                client.agent.repair(plan)
                assert all(p.exists() for p in paths)
        assert client.read_latest(oid) == expected
        ok("потеря трёх фрагментов и ремонт агентом без ключевого файла")
        with removed(paths[:4]):
            expect("INSUFFICIENT_FRAGMENTS", lambda: client.read_latest(oid))
        ok("четыре потери корректно делают объект недоступным")

        original = paths[0].read_bytes()
        try:
            paths[0].write_bytes(bytes([original[0] ^ 1]) + original[1:])
            assert client.read_latest(oid) == expected
            with removed(paths[1:4]):
                expect("INSUFFICIENT_FRAGMENTS", lambda: client.read_latest(oid))
        finally:
            paths[0].write_bytes(original)
        ok("подменённый фрагмент исключается; запас не подменяет проверку целостности")

        class ForgedAgent(LocalAgent):
            def fetch(self, descriptors):
                for i, raw in super().fetch(descriptors):
                    yield i, bytes([raw[0] ^ 1]) + raw[1:]
        expect("INSUFFICIENT_FRAGMENTS", lambda: Client(root, password, ForgedAgent(root)).read_latest(oid))
        ok("злонамеренный агент не выдаёт подменённые данные как корректные")

        expect("AUTH_FAILED", lambda: Client(root, password + "wrong").read_latest(oid))
        vault = Vault(client.vault_path, password)
        old_key_id = vault.active
        old_manifest, _ = client._manifest(vault, head)
        key = un64(old_manifest["object_crypto"]["key_b64"], 32)
        keyless = Client(root, password)
        with removed([client.vault_path]):
            expect("KEY_UNAVAILABLE", lambda: keyless.read_latest(oid))
        ok("неверный пароль и отсутствие ключа")

        for field, unavailable in [("manifest_blob_id", "MANIFEST_UNAVAILABLE"),
                                   ("envelope_blob_id", "ENVELOPE_UNAVAILABLE")]:
            path = client.record_path(head[field])
            raw = path.read_bytes()
            try:
                path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
                expect("AUTH_FAILED", lambda: client.read_latest(oid))
            finally:
                path.write_bytes(raw)
            with removed([path]):
                expect(unavailable, lambda: client.read_latest(oid))
        ok("подмена и недоступность манифеста / оболочки")

        # Ciphertext AEAD: attacker can rewrite storage and untrusted head response,
        # but cannot create a valid tag even if the content hash is recomputed.
        old_mid = head["manifest_blob_id"]
        record = client._record(old_mid, "MANIFEST_UNAVAILABLE")
        body = un64(record["ciphertext_b64"])
        record["ciphertext_b64"] = b64(bytes([body[0] ^ 1]) + body[1:])
        forged_mid = client._put_record(record)
        manifest_key = client._manifest(vault, head)[1]
        forged_head = dict(head, manifest_blob_id=forged_mid)
        forged_head["envelope_blob_id"] = client._envelope(vault, manifest_key, forged_head)
        expect("AUTH_FAILED", lambda: client._read_head(forged_head))
        ok("AEAD отклоняет подмену даже после пересчёта внешнего хеша")

        # Wrong key at manifest layer, not just password-based keyfile.
        wrong_head = dict(head)
        wrong_head["envelope_blob_id"] = client._envelope(vault, secrets.token_bytes(32), wrong_head)
        expect("AUTH_FAILED", lambda: client._read_head(wrong_head))
        ok("неверный ключ манифеста")

        object_header = old_manifest["object_crypto"]["header"]
        valid_record = seal(key, "object", object_header, expected)
        modified = dict(valid_record, header=dict(object_header, data_version=2))
        expect("AUTH_FAILED", lambda: open_sealed(key, "object", modified))
        modified = dict(valid_record)
        raw = un64(modified["ciphertext_b64"])
        modified["ciphertext_b64"] = b64(raw[:-1] + bytes([raw[-1] ^ 1]))
        expect("AUTH_FAILED", lambda: open_sealed(key, "object", modified))
        expect("AUTH_FAILED", lambda: open_sealed(key, "manifest", valid_record))
        assert b64(key).encode() not in canonical(plan)
        for stored in (root / "storage").rglob("*"):
            if stored.is_file():
                assert expected not in stored.read_bytes()
                assert b64(key).encode() not in stored.read_bytes()
        ok("контекст AEAD, подмена тега и отсутствие открытого объекта / ключа в storage")

        # Snapshot remains decryptable: rotation cannot revoke copied secrets.
        backup = root / "recovery" / "vault-backup.json"
        client.backup_key(backup)
        client.rotate_access()
        rotated = client.head(oid)
        assert rotated["head_revision"] == head["head_revision"] + 1
        assert rotated["manifest_blob_id"] == old_mid
        assert client.read_latest(oid) == expected
        old_vault = Vault(backup, password)
        assert old_vault.active == old_key_id
        assert client._manifest(old_vault, head)[0] == old_manifest
        ok("ротация доступа сохраняет чтение; старая копия ключа открывает старый манифест")

        client.reencrypt(oid)
        fresh = client.head(oid)
        assert fresh["data_version"] == 2
        assert fresh["manifest_blob_id"] != rotated["manifest_blob_id"]
        assert client.read_latest(oid) == expected
        # Replay old bytes under the currently trusted storage address.
        new_path = client.record_path(fresh["manifest_blob_id"])
        new_bytes = new_path.read_bytes()
        try:
            new_path.write_bytes(client.record_path(old_mid).read_bytes())
            expect("AUTH_FAILED", lambda: client.read_latest(oid))
        finally:
            new_path.write_bytes(new_bytes)
        ok("новая версия данных и отклонение старого манифеста при доверенном указателе")

        # Trusted state cannot be invented on first use.
        with removed([client._head_path()]):
            expect("UNTRUSTED_HEAD", lambda: client.read_latest(oid))
        stale_head = dict(fresh, head_revision=fresh["head_revision"] - 1)
        expect("VERSION_CONFLICT", lambda: client._publish(stale_head, fresh))
        with writer_lock(client.private):
            expect("VERSION_CONFLICT", lambda: client.rotate_access())
        ok("нет доверенного указателя, конфликт ревизии и один писатель")

        original_read = client._read_head
        def change_during_read(h):
            result = original_read(h)
            heads = client._heads()
            heads[oid] = dict(h, head_revision=h["head_revision"] + 1)
            atomic(client._head_path(), canonical(heads))
            return result
        with patch.object(client, "_read_head", side_effect=change_during_read):
            expect("VERSION_CHANGED", lambda: client.read_latest(oid))
        ok("изменение доверенной ревизии во время чтения")

        # Restore a CURRENT backup; an older backup cannot decrypt future keys.
        client.backup_key(backup)
        before = Vault(client.vault_path, password).active
        client.vault_path.unlink()
        client.restore_key(backup)
        assert Vault(client.vault_path, password).active != before
        assert client.read_latest(oid) == expected
        after = client.store(b"post-recovery", k=3, n=5)
        assert client.read_latest(after["object_id"]) == b"post-recovery"
        ok("восстановление ключа из актуальной копии и новый ключ для последующих записей")

        # Reserve nonce must persist before encryption; collision does not encrypt.
        vault = Vault(client.vault_path, password)
        nonce = vault.reserve_nonce()
        assert nonce in Vault(client.vault_path, password).data["nonces"][vault.active]
        with patch("crypto_demo.storage.secrets.token_bytes", return_value=un64(nonce)):
            expect("NONCE_REUSE", vault.reserve_nonce)
        ok("сохранение nonce до шифрования и отказ при повторе")

        # Bounds and strict parsing before expensive work.
        expect("INVALID_RECORD", lambda: parse(b'{"a":1,"a":2}'))
        expect("INVALID_RECORD", lambda: parse(b'{"a":1.5}'))
        expect("LIMIT_EXCEEDED", lambda: client.store(bytes(MAX_OBJECT + 1)))
        bad_vault = parse(client.vault_path.read_bytes())
        bad_vault["header"]["kdf"] = dict(KDF, n=2**30)
        fake = root / "bad-vault.json"
        fake.write_bytes(canonical(bad_vault))
        expect("FORMAT_UNSUPPORTED", lambda: Vault(fake, password))
        ok("дубли JSON, вещественные числа, лимит объекта и ограничение KDF")

        # Empty/binary inputs and non-secret output only.
        for content in (b"", bytes(range(256))):
            h = client.store(content, k=3, n=5)
            assert client.read_latest(h["object_id"]) == content
        ok("пустой объект и произвольные бинарные данные")

        # Moderate object for timing and storage overhead; not a throughput benchmark.
        payload = secrets.token_bytes(16 * 1024)
        begin = time.perf_counter()
        h = client.store(payload)
        write_seconds = time.perf_counter() - begin
        begin = time.perf_counter()
        assert client.read_latest(h["object_id"]) == payload
        read_seconds = time.perf_counter() - begin
        p = client.repair_plan(h["object_id"])
        fragments_bytes = sum(d["byte_length"] for d in p["fragments"])
        metadata_bytes = sum(client.record_path(h[f]).stat().st_size
                             for f in ("manifest_blob_id", "envelope_blob_id"))
        ok("замер записи и чтения 16 КиБ")

    report = {
        "schema": "taris-crypto-check/1", "passed_groups": len(checks), "checks": checks,
        "versions": {name: importlib.metadata.version(name) for name in ("cryptography", "reedsolo")},
        "rs_combinations": counts, "elapsed_seconds": time.perf_counter() - started,
        "sample": {"plaintext_bytes": len(payload), "fragments_bytes": fragments_bytes,
                   "manifest_and_envelope_bytes": metadata_bytes,
                   "write_seconds": write_seconds, "read_seconds": read_seconds},
        "scope": "local bytes only; no simulator integration; temporary secrets removed",
    }
    output = Path(__file__).resolve().parents[1] / "runs" / "crypto_demo" / "check_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Отчёт:", output)
    print(json.dumps(report["sample"], ensure_ascii=False))


if __name__ == "__main__":
    run()
