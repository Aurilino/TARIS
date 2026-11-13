"""Локальный однописательский прототип зашифрованного манифеста.

Доверенными считаются клиент и client-private. Хранилище blobs недоверенное.
Не сетевой протокол и не промышленное хранилище секретов.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import time

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from reedsolo import RSCodec, ReedSolomonError

MAX_OBJECT = 1024 * 1024
MAX_RECORD = 256 * 1024
MAX_BYTES = 64 * 1024 * 1024
KDF = {"n": 131072, "r": 8, "p": 1}
PROFILE = "taris-local-crypto/1"
HEX = re.compile(r"^[0-9a-f]{32}$")
BLOB = re.compile(r"^sha256:[0-9a-f]{64}$")


class DemoError(Exception):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def fail(code):
    raise DemoError(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("INVALID_RECORD")
        result[key] = value
    return result


def parse(raw):
    if len(raw) > MAX_RECORD:
        fail("LIMIT_EXCEEDED")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                           parse_constant=lambda _: fail("INVALID_RECORD"))
        def check(v, depth=0):
            if depth > 12:
                fail("INVALID_RECORD")
            if isinstance(v, dict):
                for key, item in v.items():
                    if not key.isascii():
                        fail("INVALID_RECORD")
                    check(item, depth + 1)
            elif isinstance(v, list):
                for item in v:
                    check(item, depth + 1)
            elif isinstance(v, str):
                if not v.isascii():
                    fail("INVALID_RECORD")
            elif type(v) is int:
                if not 0 <= v <= 9007199254740991:
                    fail("INVALID_RECORD")
            elif v is not None and type(v) is not bool:
                fail("INVALID_RECORD")
        check(value)
        return value
    except (ValueError, UnicodeError, RecursionError):
        fail("INVALID_RECORD")


def fields(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected.split()):
        fail("INVALID_RECORD")


def integer(value, lower=1, upper=9007199254740991):
    if type(value) is not int or not lower <= value <= upper:
        fail("INVALID_RECORD")
    return value


def b64(raw):
    return base64.b64encode(raw).decode("ascii")


def un64(value, size=None):
    try:
        if not isinstance(value, str):
            fail("INVALID_RECORD")
        raw = base64.b64decode(value, validate=True)
        if b64(raw) != value or (size is not None and len(raw) != size):
            fail("INVALID_RECORD")
        return raw
    except (ValueError, UnicodeError):
        fail("INVALID_RECORD")


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def blob_id(value):
    if not isinstance(value, str) or not BLOB.fullmatch(value):
        fail("INVALID_RECORD")
    return value


def read_limited(path, limit, missing):
    try:
        with Path(path).open("rb") as file:
            raw = file.read(limit + 1)
    except FileNotFoundError:
        fail(missing)
    if len(raw) > limit:
        fail("LIMIT_EXCEEDED")
    return raw


def atomic(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + secrets.token_hex(8) + ".tmp")
    try:
        with temporary.open("xb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def writer_lock(private):
    lock = Path(private) / ".writer.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        fail("VERSION_CONFLICT")
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def seal(key, purpose, header, plaintext):
    aad = canonical({"purpose": purpose, "header": header})
    return {"header": header, "ciphertext_b64": b64(AESGCM(key).encrypt(
        un64(header["nonce_b64"], 12), plaintext, aad))}


def open_sealed(key, purpose, record):
    fields(record, "header ciphertext_b64")
    header = record["header"]
    aad = canonical({"purpose": purpose, "header": header})
    ciphertext = un64(record["ciphertext_b64"])
    if len(ciphertext) < 16:
        fail("INVALID_RECORD")
    try:
        return AESGCM(key).decrypt(un64(header["nonce_b64"], 12), ciphertext, aad)
    except InvalidTag:
        fail("AUTH_FAILED")


class Vault:
    """Ключ доступа зашифрован паролем; память процесса входит в доверенную базу."""
    def __init__(self, path, password):
        self.path, self.password = Path(path), password
        record = parse(read_limited(path, MAX_RECORD, "KEY_UNAVAILABLE"))
        fields(record, "header ciphertext_b64")
        header = record["header"]
        fields(header, "schema profile nonce_b64 salt_b64 kdf")
        if header["schema"] != "taris-local-keystore/1" or header["profile"] != PROFILE:
            fail("FORMAT_UNSUPPORTED")
        if header["kdf"] != KDF:
            fail("FORMAT_UNSUPPORTED")
        wrapping = self._derive(password, un64(header["salt_b64"], 16))
        self.data = parse(open_sealed(wrapping, "local-keystore", record))
        fields(self.data, "active keys nonces")
        if not isinstance(self.data["keys"], dict) or not isinstance(self.data["nonces"], dict):
            fail("INVALID_RECORD")
        if self.data["active"] not in self.data["keys"]:
            fail("INVALID_RECORD")
        for key_id, key in self.data["keys"].items():
            if not isinstance(key_id, str) or not HEX.fullmatch(key_id):
                fail("INVALID_RECORD")
            un64(key, 32)
        nonces = self.data["nonces"]
        if set(nonces) != {self.data["active"]} or not isinstance(nonces[self.active], list):
            fail("INVALID_RECORD")
        used = nonces[self.active]
        if len(used) > 1000:
            fail("INVALID_RECORD")
        for nonce in used:
            un64(nonce, 12)
        if len(set(used)) != len(used):
            fail("INVALID_RECORD")

    @staticmethod
    def _derive(password, salt):
        if not isinstance(password, str) or not password:
            fail("KEY_UNAVAILABLE")
        return Scrypt(salt=salt, length=32, **KDF).derive(password.encode("utf-8"))

    @property
    def active(self):
        return self.data["active"]

    def key(self, key_id):
        if not isinstance(key_id, str) or not HEX.fullmatch(key_id):
            fail("INVALID_RECORD")
        if key_id not in self.data["keys"]:
            fail("KEY_UNAVAILABLE")
        return un64(self.data["keys"][key_id], 32)

    def save(self):
        salt = secrets.token_bytes(16)
        header = {"schema": "taris-local-keystore/1", "profile": PROFILE,
                  "nonce_b64": b64(secrets.token_bytes(12)), "salt_b64": b64(salt), "kdf": KDF}
        raw = canonical(seal(self._derive(self.password, salt), "local-keystore",
                             header, canonical(self.data)))
        if len(raw) > MAX_RECORD:
            fail("LIMIT_EXCEEDED")
        atomic(self.path, raw)

    def new_active(self):
        key_id = secrets.token_hex(16)
        self.data["keys"][key_id] = b64(secrets.token_bytes(32))
        self.data["active"] = key_id
        self.data["nonces"] = {key_id: []}
        self.save()

    @classmethod
    def create(cls, path, password):
        path = Path(path)
        if path.exists():
            fail("VERSION_CONFLICT")
        self = object.__new__(cls)
        self.path, self.password = path, password
        self.data = {"active": "", "keys": {}, "nonces": {}}
        self.new_active()
        return self

    def reserve_nonce(self):
        used = self.data["nonces"][self.active]
        # 1000 rather than 100000: bounded JSON key file, see specification amendment.
        if len(used) >= 1000:
            fail("LIMIT_EXCEEDED")
        nonce = b64(secrets.token_bytes(12))
        if nonce in used:
            fail("NONCE_REUSE")
        used.append(nonce)
        self.save()  # persist reservation BEFORE encryption
        return nonce


def codec(k, n):
    integer(k, 1, 31)
    integer(n, k + 1, 32)
    return RSCodec(nsym=n-k, nsize=n, fcr=0, prim=0x11d, generator=2, c_exp=8)


def encode(raw, k=10, n=13):
    rs = codec(k, n)
    padded = raw + bytes((-len(raw)) % k)
    length = len(padded) // k
    shards = [bytearray(length) for _ in range(n)]
    for offset in range(length):
        word = rs.encode(bytes(padded[index * length + offset] for index in range(k)))
        for index, symbol in enumerate(word):
            shards[index][offset] = symbol
    return [bytes(shard) for shard in shards]


def decode(shards, k, n, size, deadline=None):
    rs = codec(k, n)
    if len(shards) < k:
        fail("INSUFFICIENT_FRAGMENTS")
    length = (size + k - 1) // k
    if any(type(i) is not int or not 0 <= i < n or len(s) != length for i, s in shards.items()):
        fail("FRAGMENT_INVALID")
    missing = [i for i in range(n) if i not in shards]
    result = [bytearray(length) for _ in range(k)]
    try:
        for offset in range(length):
            if deadline is not None and time.monotonic() > deadline:
                fail("LIMIT_EXCEEDED")
            word = bytes(shards[i][offset] if i in shards else 0 for i in range(n))
            message, _, _ = rs.decode(word, erase_pos=missing, only_erasures=True)
            for i in range(k):
                result[i][offset] = message[i]
    except ReedSolomonError:
        fail("FRAGMENT_INVALID")
    return b"".join(result)[:size]


class LocalAgent:
    """Не получает ключи. Для демо один локальный держатель каждого blob."""
    def __init__(self, root):
        self.root = Path(root)

    def path(self, identifier):
        return self.root / "storage" / "blobs" / (blob_id(identifier)[7:] + ".bin")

    def place(self, descriptors, shards):
        for descriptor, raw in zip(descriptors, shards, strict=True):
            if digest(raw) != descriptor["blob_id"]:
                fail("FRAGMENT_INVALID")
            atomic(self.path(descriptor["blob_id"]), raw)

    def fetch(self, descriptors):
        for descriptor in descriptors:
            try:
                raw = read_limited(self.path(descriptor["blob_id"]), MAX_OBJECT + 16,
                                   "INSUFFICIENT_FRAGMENTS")
            except DemoError as error:
                if error.code == "INSUFFICIENT_FRAGMENTS":
                    continue
                raise
            yield descriptor["fragment_index"], raw

    def repair(self, plan):
        fields(plan, "fragments k n ciphertext_length ciphertext_digest")
        descriptors = validate_descriptors(plan["fragments"], plan["k"], plan["n"],
                                           plan["ciphertext_length"])
        shards = collect(self, descriptors, plan["k"], time.monotonic() + 30)
        raw = decode(shards, plan["k"], plan["n"], plan["ciphertext_length"])
        if digest(raw) != blob_id(plan["ciphertext_digest"]):
            fail("FRAGMENT_INVALID")
        rebuilt = encode(raw, plan["k"], plan["n"])
        # Verify ALL outputs before writing any.
        if any(digest(raw) != d["blob_id"] for d, raw in zip(descriptors, rebuilt, strict=True)):
            fail("FRAGMENT_INVALID")
        self.place(descriptors, rebuilt)


def validate_descriptors(descriptors, k, n, size):
    codec(k, n)
    integer(size, 16, MAX_OBJECT + 16)
    if not isinstance(descriptors, list) or len(descriptors) != n:
        fail("INVALID_RECORD")
    indices = set()
    for d in descriptors:
        fields(d, "fragment_index blob_id byte_length")
        index = integer(d["fragment_index"], 0, n - 1)
        blob_id(d["blob_id"])
        integer(d["byte_length"])
        if index in indices or d["byte_length"] != (size + k - 1) // k:
            fail("INVALID_RECORD")
        indices.add(index)
    return sorted(descriptors, key=lambda d: d["fragment_index"])


def collect(agent, descriptors, k, deadline):
    expected = {d["fragment_index"]: d for d in descriptors}
    shards, attempts, total = {}, {}, 0
    for index, raw in agent.fetch(descriptors):
        if time.monotonic() > deadline:
            fail("LIMIT_EXCEEDED")
        if type(index) is not int or index not in expected or not isinstance(raw, bytes):
            fail("FRAGMENT_INVALID")
        attempts[index] = attempts.get(index, 0) + 1
        total += len(raw)
        if attempts[index] > 2 or total > MAX_BYTES:
            fail("LIMIT_EXCEEDED")
        d = expected[index]
        if len(raw) == d["byte_length"] and digest(raw) == d["blob_id"]:
            shards[index] = raw
        if len(shards) >= k:
            return shards
    fail("INSUFFICIENT_FRAGMENTS")


class Client:
    def __init__(self, root, password, agent=None):
        self.root = Path(root)
        self.private = self.root / "client-private"
        self.private.mkdir(parents=True, exist_ok=True)
        self.password = password
        self.agent = agent or LocalAgent(root)

    @property
    def vault_path(self):
        return self.private / "keystore.json"

    def initialize(self):
        with writer_lock(self.private):
            if self.vault_path.exists() or self._head_path().exists():
                fail("VERSION_CONFLICT")
            Vault.create(self.vault_path, self.password)
            atomic(self._head_path(), canonical({}))

    def _head_path(self):
        return self.private / "head-store.json"

    def _heads(self):
        result = parse(read_limited(self._head_path(), MAX_RECORD, "UNTRUSTED_HEAD"))
        if not isinstance(result, dict):
            fail("UNTRUSTED_HEAD")
        return result

    def head(self, object_id):
        if not isinstance(object_id, str) or not HEX.fullmatch(object_id):
            fail("INVALID_RECORD")
        head = self._heads().get(object_id)
        if head is None:
            fail("UNTRUSTED_HEAD")
        fields(head, "schema object_id head_revision data_version manifest_version manifest_blob_id envelope_blob_id")
        if head["schema"] != "taris-object-head/1" or head["object_id"] != object_id:
            fail("INVALID_RECORD")
        for key in ("head_revision", "data_version", "manifest_version"):
            integer(head[key])
        blob_id(head["manifest_blob_id"])
        blob_id(head["envelope_blob_id"])
        return head

    def _publish(self, old, head):
        heads = self._heads()
        if heads.get(head["object_id"]) != old:
            fail("VERSION_CONFLICT")
        heads[head["object_id"]] = head
        raw = canonical(heads)
        if len(raw) > MAX_RECORD:
            fail("LIMIT_EXCEEDED")
        atomic(self._head_path(), raw)

    def record_path(self, identifier):
        return self.root / "storage" / "records" / (blob_id(identifier)[7:] + ".json")

    def _put_record(self, record):
        raw = canonical(record)
        if len(raw) > MAX_RECORD:
            fail("LIMIT_EXCEEDED")
        identifier = digest(raw)
        atomic(self.record_path(identifier), raw)
        return identifier

    def _record(self, identifier, missing):
        raw = read_limited(self.record_path(identifier), MAX_RECORD, missing)
        if digest(raw) != identifier:
            fail("AUTH_FAILED")
        return parse(raw)

    def _envelope(self, vault, manifest_key, head):
        header = {key: head[key] for key in ("object_id", "data_version", "manifest_version", "manifest_blob_id")}
        header.update(schema="taris-key-envelope/1", access_key_id=vault.active,
                      wrap_profile="AES-256-GCM", nonce_b64=vault.reserve_nonce())
        return self._put_record(seal(vault.key(vault.active), "key-envelope", header,
                                     canonical({"manifest_key_b64": b64(manifest_key)})))

    def _manifest(self, vault, head):
        envelope = self._record(head["envelope_blob_id"], "ENVELOPE_UNAVAILABLE")
        fields(envelope, "header ciphertext_b64")
        h = envelope["header"]
        fields(h, "schema object_id data_version manifest_version manifest_blob_id access_key_id wrap_profile nonce_b64")
        if h["schema"] != "taris-key-envelope/1" or h["wrap_profile"] != "AES-256-GCM":
            fail("FORMAT_UNSUPPORTED")
        if any(h[key] != head[key] for key in ("object_id", "data_version", "manifest_version", "manifest_blob_id")):
            fail("AUTH_FAILED")
        inner = parse(open_sealed(vault.key(h["access_key_id"]), "key-envelope", envelope))
        fields(inner, "manifest_key_b64")
        manifest_key = un64(inner["manifest_key_b64"], 32)
        record = self._record(head["manifest_blob_id"], "MANIFEST_UNAVAILABLE")
        fields(record, "header ciphertext_b64")
        h = record["header"]
        fields(h, "schema object_id data_version manifest_version crypto_profile nonce_b64")
        if h["schema"] != "taris-encrypted-manifest/1" or h["crypto_profile"] != "AES-256-GCM":
            fail("FORMAT_UNSUPPORTED")
        if any(h[key] != head[key] for key in ("object_id", "data_version", "manifest_version")):
            fail("AUTH_FAILED")
        payload = parse(open_sealed(manifest_key, "manifest", record))
        fields(payload, "schema object_id data_version manifest_version plaintext_length object_crypto coding ciphertext_length ciphertext_digest fragments")
        if payload["schema"] != "taris-manifest-payload/1":
            fail("FORMAT_UNSUPPORTED")
        if any(payload[key] != head[key] for key in ("object_id", "data_version", "manifest_version")):
            fail("AUTH_FAILED")
        integer(payload["plaintext_length"], 0, MAX_OBJECT)
        if payload["ciphertext_length"] != payload["plaintext_length"] + 16:
            fail("INVALID_RECORD")
        blob_id(payload["ciphertext_digest"])
        c = payload["coding"]
        fields(c, "profile k n")
        if c["profile"] != "taris-rs-shards/1":
            fail("FORMAT_UNSUPPORTED")
        payload["fragments"] = validate_descriptors(payload["fragments"], c["k"], c["n"], payload["ciphertext_length"])
        oc = payload["object_crypto"]
        fields(oc, "header key_b64")
        fields(oc["header"], "schema profile object_id data_version nonce_b64")
        if oc["header"]["schema"] != "taris-object-ciphertext/1" or oc["header"]["profile"] != "AES-256-GCM":
            fail("FORMAT_UNSUPPORTED")
        if any(oc["header"][key] != head[key] for key in ("object_id", "data_version")):
            fail("AUTH_FAILED")
        un64(oc["key_b64"], 32)
        un64(oc["header"]["nonce_b64"], 12)
        return payload, manifest_key

    def store(self, plaintext, k=10, n=13):
        with writer_lock(self.private):
            return self._store(plaintext, k, n)

    def _store(self, plaintext, k, n, old=None):
        if not isinstance(plaintext, bytes) or len(plaintext) > MAX_OBJECT:
            fail("LIMIT_EXCEEDED")
        codec(k, n)
        vault = Vault(self.vault_path, self.password)
        object_id = old["object_id"] if old else secrets.token_hex(16)
        version = old["data_version"] + 1 if old else 1
        mversion = old["manifest_version"] + 1 if old else 1
        key, manifest_key = secrets.token_bytes(32), secrets.token_bytes(32)
        header = {"schema": "taris-object-ciphertext/1", "profile": "AES-256-GCM",
                  "object_id": object_id, "data_version": version, "nonce_b64": b64(secrets.token_bytes(12))}
        encrypted = un64(seal(key, "object", header, plaintext)["ciphertext_b64"])
        shards = encode(encrypted, k, n)
        descriptors = [{"fragment_index": i, "blob_id": digest(raw), "byte_length": len(raw)}
                       for i, raw in enumerate(shards)]
        self.agent.place(descriptors, shards)
        fetched = collect(self.agent, descriptors, n, time.monotonic() + 30)
        if len(fetched) != n:
            fail("INSUFFICIENT_FRAGMENTS")
        payload = {"schema": "taris-manifest-payload/1", "object_id": object_id,
                   "data_version": version, "manifest_version": mversion,
                   "plaintext_length": len(plaintext), "object_crypto": {"header": header, "key_b64": b64(key)},
                   "coding": {"profile": "taris-rs-shards/1", "k": k, "n": n},
                   "ciphertext_length": len(encrypted), "ciphertext_digest": digest(encrypted),
                   "fragments": descriptors}
        mh = {"schema": "taris-encrypted-manifest/1", "object_id": object_id,
              "data_version": version, "manifest_version": mversion,
              "crypto_profile": "AES-256-GCM", "nonce_b64": b64(secrets.token_bytes(12))}
        mid = self._put_record(seal(manifest_key, "manifest", mh, canonical(payload)))
        head = {"schema": "taris-object-head/1", "object_id": object_id,
                "head_revision": old["head_revision"] + 1 if old else 1,
                "data_version": version, "manifest_version": mversion, "manifest_blob_id": mid}
        head["envelope_blob_id"] = self._envelope(vault, manifest_key, head)
        # Verify the staged records and whole object before publishing.
        self._read_head(head)
        self._publish(old, head)
        return head

    def _read_head(self, head):
        deadline = time.monotonic() + 30
        vault = Vault(self.vault_path, self.password)
        payload, _ = self._manifest(vault, head)
        c = payload["coding"]
        shards = collect(self.agent, payload["fragments"], c["k"], deadline)
        encrypted = decode(shards, c["k"], c["n"], payload["ciphertext_length"], deadline)
        if digest(encrypted) != payload["ciphertext_digest"]:
            fail("AUTH_FAILED")
        oc = payload["object_crypto"]
        result = open_sealed(un64(oc["key_b64"], 32), "object",
                             {"header": oc["header"], "ciphertext_b64": b64(encrypted)})
        if len(result) != payload["plaintext_length"]:
            fail("INVALID_RECORD")
        if time.monotonic() > deadline:
            fail("LIMIT_EXCEEDED")
        return result

    def read_latest(self, object_id):
        head = self.head(object_id)
        result = self._read_head(head)
        if self.head(object_id) != head:
            fail("VERSION_CHANGED")
        return result

    def repair_plan(self, object_id):
        p, _ = self._manifest(Vault(self.vault_path, self.password), self.head(object_id))
        return {"fragments": p["fragments"], "k": p["coding"]["k"], "n": p["coding"]["n"],
                "ciphertext_length": p["ciphertext_length"], "ciphertext_digest": p["ciphertext_digest"]}

    def rotate_access(self):
        """Перепаковывает все текущие оболочки; старые ключи сохраняются для восстановления."""
        with writer_lock(self.private):
            vault = Vault(self.vault_path, self.password)
            heads = [self.head(i) for i in self._heads()]
            staged = [(h, self._manifest(vault, h)[1]) for h in heads]
            vault.new_active()  # previous keys decrypt-only; crash leaves old heads readable
            for old, manifest_key in staged:
                new = dict(old, head_revision=old["head_revision"] + 1)
                new["envelope_blob_id"] = self._envelope(vault, manifest_key, new)
                self._read_head(new)
                self._publish(old, new)

    def reencrypt(self, object_id):
        with writer_lock(self.private):
            old = self.head(object_id)
            plaintext = self.read_latest(object_id)
            p, _ = self._manifest(Vault(self.vault_path, self.password), old)
            return self._store(plaintext, p["coding"]["k"], p["coding"]["n"], old)

    def backup_key(self, destination):
        """Копирует только зашифрованный vault; доверенные heads резервируются отдельно."""
        with writer_lock(self.private):
            Vault(self.vault_path, self.password)  # validate before copying
            atomic(destination, read_limited(self.vault_path, MAX_RECORD, "KEY_UNAVAILABLE"))

    def restore_key(self, source):
        with writer_lock(self.private):
            if self.vault_path.exists():
                fail("VERSION_CONFLICT")
            backup = Vault(source, self.password)
            backup.path = self.vault_path
            backup.new_active()  # do NOT resume encryption under an old restored nonce state
