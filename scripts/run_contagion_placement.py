"""Парное сравнение агентского размещения с затуханием риска по графу.

    python scripts/run_contagion_placement.py [workers=3] [days=365] [seed-range=1-3]

Сравнивает один и тот же мир и резерв с contagion_strength=0 и 1. Общие домены
ДЦ/провайдера/региона исключены при построении передаваемых рёбер.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "runs", "contagion_placement_bonferroni")


def run(job):
    scn, seed, linked, days = job
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    from taris_sim.policy import ReservePolicy

    cfg = scenario(scn, seed=seed, horizon_days=days, **MAIN)
    w = World(cfg)
    ledger = Ledger(w, Chain(w), DataStream(cfg, len(w.nodes)))
    pol = ReservePolicy(w, 13, mode="fixed", R_fixed=2, rates="agent", placement="agent",
                        contagion_strength=1.0 if linked else 0.0)
    started = time.time()
    res = ledger.run(pol)
    signal = pol.contagion_signal
    out = {
        "job": job, "metrics": res.metrics(), "losses": res.losses,
        "migration_traffic": float(res.migrations["traffic"].sum()) if len(res.migrations) else 0.0,
        "contagion_strength": pol.contagion_strength,
        "supported_edges": pol.contagion_edges,
        "signal_mean": float(signal.mean()), "signal_p95": float(__import__("numpy").quantile(signal, 0.95)),
        "seconds": round(time.time() - started),
    }
    os.makedirs(OUT, exist_ok=True)
    name = "linked" if linked else "base"
    path = os.path.join(OUT, f"{scn}_{name}_s{seed}_d{days}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return scn, seed, name, out["metrics"]["Потеряно объектов"], out["migration_traffic"], \
        out["supported_edges"], out["signal_p95"], out["seconds"]


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


if __name__ == "__main__":
    keep_awake()
    args = [x for x in sys.argv[1:] if x.isdigit()]
    workers = int(args[0]) if args else 3
    days = int(args[1]) if len(args) > 1 else 365
    seed_range = next((x for x in sys.argv[1:] if "-" in x and x.split("-")[0].isdigit()), "1-3")
    lo, hi = map(int, seed_range.split("-"))
    jobs = [(scn, seed, linked, days) for scn in ("S0", "S6") for seed in range(lo, hi + 1)
            for linked in (False, True)]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(
        OUT, f"{j[0]}_{'linked' if j[2] else 'base'}_s{j[1]}_d{days}.pkl"))]
    print(f"Прогонов: {len(jobs)}, дни={days}, workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            scn, seed, name, lost, traffic, edges, p95, seconds = f.result()
            print(f"{scn} s{seed} {name}: потери={lost}, трафик={traffic:.1f}, "
                  f"рёбер={edges}, сигнал p95={p95:.4f}, {seconds} с", flush=True)
