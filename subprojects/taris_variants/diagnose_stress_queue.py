"""Внешняя диагностика возможной блокировки проверок; не доказательство причины."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
import hashlib
import json
import numpy as np
from taris_sim.config import Config
from taris_sim.world import World
from taris_sim.chain import Chain


def main():
    folder = Path(__file__).parent
    directory = folder/"series_i51"
    audit = json.loads((directory/"audit.json").read_text(encoding="utf-8"))
    rows, cache = [], {}
    defaults = Config()
    for name, digest in audit["inputs"].items():
        path = directory/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        seed = case["seed"]
        if seed not in cache:
            cfg = Config(**{key: tuple(value) if isinstance(getattr(defaults, key), tuple) else value
                            for key, value in case["config"].items()})
            world = World(cfg)
            chain = Chain(world)
            down = {int(node): list(zip(g.start, g.end)) for node, g in chain.true_down.groupby("node")}
            cache[seed] = cfg, world, down
        cfg, world, down = cache[seed]
        t = np.nextafter(float(cfg.horizon_days), -np.inf)
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                waiting = [j for j in record["jobs"] if j["finished"] is None and j["status"] == "waiting_read_sources"]
                proofs = [j["id"] for j in record["jobs"] if j["finished"] is None and j["kind"] == "proof"]
                for job in waiting:
                    indices, available = set(), set()
                    for f in record["physical_fragments"]:
                        if f["obj"] != job["obj"] or f["start"] > t or (f["physical_end"] is not None and f["physical_end"] <= t):
                            continue
                        indices.add(f["idx"])
                        node = f["node"]
                        if world.nodes.alive_to.iloc[node] > t and not any(a <= t < b for a, b in down.get(node, [])):
                            available.add(f["idx"])
                    queue_full = record["metrics"]["queue_rejections"].get("proof_queue_full", 0)
                    rows.append({"seed": seed, "mode": case["stress_mode"], "profile": profile, "variant": variant,
                        "job": job["id"], "obj": job["obj"], "submitted": job["submitted"], "started": job["started"],
                        "physical_distinct_at_end": len(indices), "physically_available_distinct_at_end": len(available),
                        "unfinished_proof_jobs": proofs, "proof_queue_full_rejections": queue_full,
                        "possible_service_verification_block": len(available) >= cfg.k and (queue_full > 0 or bool(proofs))})
    output = folder/"stress_sensitivity_i51"
    output.mkdir(exist_ok=True)
    result = {"schema": "taris-stress-queue-diagnosis/1", "input_hashes": audit["inputs"],
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows,
        "candidate_cases": sum(r["possible_service_verification_block"] for r in rows),
        "causal_fix_tested": False, "not_agent_features": True}
    (output/"queue_diagnosis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f'I5.1: ожиданий источников{len(rows)}, кандидатов блокировки проверки{result["candidate_cases"]}; причинный тест ещё не выполнен')


if __name__ == "__main__":
    main()
