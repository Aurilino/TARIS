"""Парное сравнение размещения по оценкам агента с оракулом (пилот 5.2).

    S0/S2, основной мир, порог восстановления R_min=2. Сравниваются истинные
    веса оракула, равномерные веса и оценка агента; все используют публичные
    кошелёк/подсеть. Миры генерируются одинаково по seed.

    python scripts/run_agent_placement.py [workers=4] [days=365] [seed-range=1-3] [--force]
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "runs")


def run(job):
    scn, seed, kind, days = job
    import numpy as np
    from scipy.stats import spearmanr
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    from taris_sim.policy import OraclePolicy, ReservePolicy

    cfg = scenario(scn, seed=seed, horizon_days=days, **MAIN)
    w = World(cfg)
    L = Ledger(w, Chain(w), DataStream(cfg, len(w.nodes)))
    if kind == "oracle":
        pol = ReservePolicy(w, 13, mode="fixed", R_fixed=2, rates="true", placement="oracle")
    else:
        pol = ReservePolicy(w, 13, mode="fixed", R_fixed=2, rates="agent", placement=kind)
    started = time.time()
    res = L.run(pol)
    truth = OraclePolicy(w, 13).w
    rho = float(spearmanr(pol.w, truth).statistic) if np.std(pol.w) > 0 else float("nan")
    out = {"job": job, "metrics": res.metrics(), "losses": res.losses,
           "rho_to_oracle": rho, "fits": len(getattr(pol, "fits", [])),
           "placement": kind, "seconds": round(time.time() - started)}
    path = os.path.join(OUT, f"agent_placement_{days}", f"{scn}_{kind}_s{seed}.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return scn, seed, kind, days, out["metrics"]["Потеряно объектов"], out["metrics"]["Трафик восстановления в год"], rho, out["seconds"]


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


if __name__ == "__main__":
    keep_awake()
    workers = int(next((x for x in sys.argv[1:] if x.isdigit()), "4"))
    args = [x for x in sys.argv[1:] if x.isdigit()]
    days = int(args[1]) if len(args) > 1 else 365
    seed_range = next((x for x in sys.argv[1:] if "-" in x and x.split("-")[0].isdigit()), "1-3")
    s0, s1 = map(int, seed_range.split("-"))
    out_dir = os.path.join(OUT, f"agent_placement_{days}")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, seed, kind, days) for s in ("S0", "S2") for seed in range(s0, s1 + 1)
            for kind in ("oracle", "uniform", "agent")]
    if "--force" not in sys.argv:
        jobs = [j for j in jobs if not os.path.exists(os.path.join(out_dir, f"{j[0]}_{j[2]}_s{j[1]}.pkl"))]
    print(f"Прогонов к расчёту: {len(jobs)}, дней={days}, seeds={seed_range}, workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for f in as_completed([pool.submit(run, j) for j in jobs]):
            scn, seed, kind, days, lost, traffic, rho, seconds = f.result()
            print(f"{scn} s{seed} {kind}: потерь {lost}, трафик ремонта {traffic:.1f}/год, "
                  f"rho с оракулом {rho:.3f}, {seconds} c", flush=True)
