"""Пилот чувствительности экономического порога агента к ценности объекта W.

Три общих зерна S6, основной мир, горизонт 365 суток: отсутствие прогноза и
суточный экономический агент при W = 10^2, 10^3, 10^4. Сохраняются метки со
снимками портфеля и фактические потери, чтобы сопоставить модельную выгоду с миром.

    python scripts/run_agent_econ_sensitivity.py [workers, default 4]
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "runs", "agent_econ_sensitivity")
SEEDS = (1, 2, 3)
VALUES = (100.0, 1000.0, 10000.0)


def run(job):
    seed, W = job
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    from taris_sim.policy import AgentCrisisPolicy

    cfg = scenario("S6", seed=seed, alarm_step_hours=24, horizon_days=365, **MAIN)
    w = World(cfg)
    L = Ledger(w, Chain(w), DataStream(cfg, len(w.nodes)))
    if W is None:
        pol = AgentCrisisPolicy(w, 13, forecast=False)
        name = "none"
    else:
        pol = AgentCrisisPolicy(w, 13, kappa="econ", econ_W=W)
        name = f"W{int(W)}"
    started = time.time()
    res = L.run(pol)
    summary, table = res.crisis_report()
    econ_diag = None
    if W is not None:
        day = min(pol.fc.cum_day, pol.fc.T)
        lam = (pol.fc.cum["longst"][:, day].sum() + 0.5) / max(
            pol.fc.cum["up"][:, day].sum() / pol.fc.cpd, 1.0)
        p_bg = 1.0 - __import__("math").exp(-lam * pol.w_loss)
        econ_diag = {"lambda_bg": float(lam), "p_bg": float(p_bg),
                     "q_hit": {k: (v[0] + 1.0) / (v[1] + 2.0) for k, v in pol.q_hit.items()}}
    out = {
        "job": (name, seed), "W": W, "seconds": round(time.time() - started),
        "metrics": res.metrics(), "losses": res.losses,
        "crisis_summary": summary, "crisis_table": table,
        "migration_traffic": float(res.migrations["traffic"].sum()) if len(res.migrations) else 0.0,
        "thr": pol.thr.copy(), "econ_log": getattr(pol, "econ_log", []),
        "econ_diag": econ_diag,
        # Keep the signal episodes and their pre-migration placement snapshots.
        "labels": {k: [(float(x[0]), bool(x[1]), dict(x[2])) for x in v]
                   for k, v in getattr(pol, "labels", {}).items()},
        "q_hit": getattr(pol, "q_hit", None),
    }
    path = os.path.join(OUT, f"{name}_s{seed}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    return name, seed, len(res.losses), out["migration_traffic"], out["seconds"]


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


if __name__ == "__main__":
    keep_awake()
    os.makedirs(OUT, exist_ok=True)
    jobs = [(s, w) for s in SEEDS for w in (None, *VALUES)]
    force = "--force" in sys.argv
    if not force:
        jobs = [j for j in jobs if not os.path.exists(os.path.join(
            OUT, f"{'none' if j[1] is None else 'W'+str(int(j[1]))}_s{j[0]}.pkl"))]
    workers = int(next((x for x in sys.argv[1:] if x.isdigit()), "4"))
    print(f"Прогонов к расчёту: {len(jobs)}, workers={workers}, выход: {OUT}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for f in as_completed(futures):
            name, seed, lost, traffic, seconds = f.result()
            print(f"{name} s{seed}: потерь {lost}, переносы {traffic:.1f}, {seconds} c", flush=True)
