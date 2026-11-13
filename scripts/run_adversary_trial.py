"""Пилот S3/S4/S7: противник, потери, сбор шифротекста и диагностика связей."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import time
import numpy as np
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import ReservePolicy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", choices=("S3", "S4", "S7"), default="S3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--nodes", type=int, default=120)
    ap.add_argument("--share", type=float, default=.05)
    ap.add_argument("--p-leak", type=float, default=.3)
    ap.add_argument("--key-compromise-day", type=float)
    ap.add_argument("--mode", choices=("matched", "mismatched"), default="matched")
    ap.add_argument("--policies", default="uniform,agent")
    ap.add_argument("--out", type=Path, default=Path("runs/adversary/trial"))
    args = ap.parse_args()
    if args.days < (200 if args.scenario == "S7" else 10) or args.nodes < 30:
        ap.error("Слишком короткий горизонт или мало узлов; S7 требует >= 200 суток")
    policies = args.policies.split(",")
    if not policies or any(p not in ("uniform", "agent") for p in policies):
        ap.error("policies: uniform,agent")
    cfg = scenario(args.scenario, seed=args.seed, world_mode=args.mode, horizon_days=args.days,
                   train_days=max(1, args.days//2), n_initial_nodes=args.nodes, node_join_rate=.08,
                   blocks_per_epoch=1, n_dc=30, n_providers=6, n_regions=5,
                   adv_sybil_share=args.share, adv_pair_leak=args.p_leak,
                   adv_key_compromise_day=args.key_compromise_day)
    w = World(cfg)
    ch = Chain(w)
    data = DataStream(cfg, len(w.nodes))
    runs = []
    for name in policies:
        begin = time.perf_counter()
        if name == "uniform":
            # Одинаковое правило восстановления позволяет отделить размещение.
            policy = ReservePolicy(w, n=13, mode="fixed", R_fixed=2, rates="agent", placement="uniform")
        else:
            policy = ReservePolicy(w, n=13, mode="fixed", R_fixed=2, rates="agent", placement="agent_greedy")
        result = Ledger(w, ch, data).run(policy)
        threat = result.adversary_report()
        runs.append({"policy": name, "lost_objects": len(result.losses),
                     "repair_traffic": float(result.repairs.traffic.sum()),
                     "ciphertext_collected_objects": len(threat["ciphertext_exposures"]),
                     "plaintext_disclosed_objects": threat["plaintext_disclosed_objects"],
                     "adversary": threat, "seconds": time.perf_counter()-begin})
        print(name, ": потерь", len(result.losses), "; собрано шифротекстов", len(threat["ciphertext_exposures"]), flush=True)
    diagnostics = w.adversary.ownership_report(ch, np.nextafter(float(args.days), 0.))
    out = {"schema": "taris-adversary-trial/1", "config": json.loads(cfg.to_json()),
           "world_fingerprint": w.fingerprint(), "runs": runs, "ownership": diagnostics,
           "scope": "single-seed pilot; common world; fixed R_min=2; key compromise only if explicitly configured"}
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.scenario}_{args.mode}_{args.seed}.json"
    target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Отчёт:", target)


if __name__ == "__main__":
    main()
