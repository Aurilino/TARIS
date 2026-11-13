"""Парная годовая проверка агентского и тарифного размещения на S2.

    python scripts/run_tariff_agent_pair.py [workers=4] [seed-range=1-20]
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from scripts.run_tariff_placement import PROBE

OUT_ROOT = os.path.join(ROOT, "runs")


def run(job):
    scn, seed, kind = job
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    from taris_sim.policy import ReservePolicy

    cfg = scenario(scn, seed=seed, horizon_days=365, **MAIN)
    world = World(cfg)
    ledger = Ledger(world, Chain(world), DataStream(cfg, len(world.nodes)))
    opts = {"tariff_settings": PROBE} if kind == "tariff" else {}
    policy = ReservePolicy(world, 13, mode="fixed", R_fixed=2, rates="agent",
                           placement=kind, **opts)
    t0 = time.time()
    result = ledger.run(policy)
    tv = policy.tariff_values
    out = {"job": job, "metrics": result.metrics(), "losses": result.losses,
           "tariff_eligible_share": float(tv["financial"]["new_data_eligible"].mean()) if tv else None,
           "seconds": round(time.time() - t0)}
    tag = "" if scn == "S2" else f"_{scn}"
    out_dir = os.path.join(OUT_ROOT, f"tariff_agent_pair_365{tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{kind}_s{seed}.pkl"), "wb") as f:
        pickle.dump(out, f)
    m = out["metrics"]
    return seed, kind, m["Потеряно объектов"], m["EAFDL (среднее по годам)"], \
        m["Трафик восстановления в год"], out["tariff_eligible_share"], out["seconds"]


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


if __name__ == "__main__":
    keep_awake()
    args = [x for x in sys.argv[1:] if x.isdigit()]
    workers = int(args[0]) if args else 4
    scn = next((x for x in sys.argv[1:] if x in ("S0", "S2")), "S2")
    seed_range = next((x for x in sys.argv[1:] if "-" in x and x.split("-")[0].isdigit()), "1-20")
    s0, s1 = map(int, seed_range.split("-"))
    tag = "" if scn == "S2" else f"_{scn}"
    out_dir = os.path.join(OUT_ROOT, f"tariff_agent_pair_365{tag}")
    jobs = [(scn, seed, kind) for seed in range(s0, s1 + 1) for kind in ("agent", "tariff")
            if not os.path.exists(os.path.join(out_dir, f"{kind}_s{seed}.pkl"))]
    print(f"{scn}, 365 суток; прогонов осталось {len(jobs)}, зёрна={seed_range}, workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            seed, kind, lost, eafdl, traffic, eligible, seconds = future.result()
            elig = "—" if eligible is None else f"{eligible:.1%}"
            print(f"s{seed} {kind}: потери {lost}, EAFDL {eafdl:.6g}, трафик {traffic:.1f}/год, "
                  f"допуск {elig}, {seconds} с", flush=True)
