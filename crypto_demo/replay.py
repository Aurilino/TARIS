"""Проверка одного снимка TARIS на настоящих зашифрованных байтах.

Пробный объект не является настоящим содержимым блокчейна. Сеть задаёт только
держателей, индексы и доступность; криптографический контур остаётся клиентским.
"""
from pathlib import Path
import tempfile

from taris_sim.security import SecurityReplay, SecurityError
from .storage import Client, LocalAgent, Vault, DemoError, decode, open_sealed, b64, un64


class SnapshotAgent(LocalAgent):
    def __init__(self, root, replay, obj, t):
        super().__init__(root)
        self.replay, self.obj, self.t = replay, obj, t

    def fetch(self, descriptors):
        holders = self.replay.live_holders(self.obj, self.t)
        for d in descriptors:
            for node in holders.get(d["fragment_index"], [])[:2]:
                if self.replay.node_action("deny_node", node, self.t):
                    continue
                raw = self.path(d["blob_id"]).read_bytes()
                if self.replay.node_action("corrupt_node", node, self.t):
                    raw = bytes([raw[0] ^ 1]) + raw[1:]
                yield d["fragment_index"], raw


def verify_snapshot(result, document, obj, t, workspace):
    """Создаёт/удаляет тестовые секреты только внутри явно заданного workspace."""
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    replay = SecurityReplay(result, document)
    modeled = replay.query(obj, t)
    k, n = modeled["k"], modeled["initial_n"]
    if not 1 <= k < n <= 32:
        raise SecurityError("Байтовой демонстрации требуется 1 <= k < initial_n <= 32")
    with tempfile.TemporaryDirectory(prefix="snapshot-", dir=workspace) as directory:
        root = Path(directory).resolve()
        assert root.is_relative_to(workspace)
        # Synthetic demonstration bytes; never serialize object contents or keys.
        plaintext = f"TARIS encrypted snapshot: object {obj}; sample bytes, not blockchain data".encode()
        client = Client(root, "temporary-test-password-not-for-real-data")
        client.initialize()
        head = client.store(plaintext, k, n)
        payload, _ = client._manifest(Vault(client.vault_path, client.password), head)
        descriptors = payload["fragments"]
        key = un64(payload["object_crypto"]["key_b64"], 32)
        encrypted = {d["fragment_index"]: client.agent.path(d["blob_id"]).read_bytes()
                     for d in descriptors}
        # Evaluator models copying/exposure, not an interface available to the agent.
        attackers = []
        for state in modeled["attackers"]:
            samples = {i: encrypted[i] for i in state["copied_indices"]}
            ciphertext_ok, disclosed = False, False
            if len(samples) >= k:
                reconstructed = decode(samples, k, n, payload["ciphertext_length"])
                ciphertext_ok = True
                if state["object_key_exposed"]:
                    recovered = open_sealed(key, "object", {
                        "header": payload["object_crypto"]["header"], "ciphertext_b64": b64(reconstructed)})
                    disclosed = recovered == plaintext
            attackers.append({"attacker": state["attacker"], "ciphertext_reconstructed": ciphertext_ok,
                              "plaintext_verified": disclosed,
                              "key_injected_by_scenario": state["object_key_exposed"]})
        client.agent = SnapshotAgent(root, replay, obj, t)
        reasons = set(modeled["unavailability_reasons"])
        if "client_key_unavailable" in reasons:
            client.vault_path.unlink()
        if "manifest_unavailable" in reasons:
            client.record_path(head["manifest_blob_id"]).unlink()
        if "envelope_unavailable" in reasons:
            client.record_path(head["envelope_blob_id"]).unlink()
        try:
            verified = client.read_latest(head["object_id"]) == plaintext
            code = "OK"
        except DemoError as error:
            verified, code = False, error.code
        expected = modeled["client_readable"]
        if verified != expected or any(
            actual["plaintext_verified"] != forecast["plaintext_disclosed"]
            for actual, forecast in zip(attackers, modeled["attackers"], strict=True)
        ):
            raise AssertionError("Модель безопасности расходится с проверкой на байтах")
        return {"schema": "taris-security-byte-check/1", "obj": obj, "t": float(t),
                "synthetic_payload": True, "client_verified": verified, "client_result": code,
                "attackers": attackers, "matches_model": True}
