"""Годовая чувствительность S2 к базовому доходу reward0, зёрна 1–20."""
from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from taris_sim.tariff import PriceBasis, TariffSettings

CASES = {
    "low": TariffSettings(PriceBasis(20.0, 1.0), 0.2, 2.0, 1.0, 0.15, 1.0),
    "high": TariffSettings(PriceBasis(20.0, 1.0), 0.2, 2.0, 1.0, 0.60, 1.0),
}
OUT = os.path.join(ROOT, "runs", "tariff_reward_sensitivity_365")


def run(job):
    seed, label = job
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    from taris_sim.policy import ReservePolicy

    cfg = scenario("S2", seed=seed, horizon_days=365, **MAIN)
    world = World(cfg)
    ledger = Ledger(world, Chain(world), DataStream(cfg, len(world.nodes)))
    policy = ReservePolicy(world, 13, mode="fixed", R_fixed=2, rates="agent",
                           placement="tariff", tariff_settings=CASES[label])
    t0 = time.time()
    result = ledger.run(policy)
    tv = policy.tariff_values
    out = {"seed": seed, "case": label, "reward0": CASES[label].reward0,
           "metrics": result.metrics(), "tariff_eligible_share":
               float(tv["financial"]["new_data_eligible"].mean()) if tv else None,
           "seconds": round(time.time() - t0)}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{label}_s{seed}.pkl"), "wb") as f:
        pickle.dump(out, f)
    m = out["metrics"]
    return seed, label, m["Потеряно объектов"], m["EAFDL (среднее по годам)"], \
        m["Трафик восстановления в год"], out["tariff_eligible_share"], out["seconds"]


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


if __name__ == "__main__":
    keep_awake()
    args = [x for x in sys.argv[1:] if x.isdigit()]
    workers = int(args[0]) if args else 4
    seed_range = next((x for x in sys.argv[1:] if "-" in x and x.split("-")[0].isdigit()), "1-20")
    s0, s1 = map(int, seed_range.split("-"))
    jobs = [(seed, label) for seed in range(s0, s1 + 1) for label in CASES
            if not os.path.exists(os.path.join(OUT, f"{label}_s{seed}.pkl"))]
    print(f"S2, 365 суток; прогонов осталось {len(jobs)}, зёрна={seed_range}, workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            seed, label, lost, eafdl, traffic, eligible, seconds = future.result()
            print(f"s{seed} {label}: потери {lost}, EAFDL {eafdl:.6g}, трафик {traffic:.1f}/год, "
                  f"допуск {eligible:.1%}, {seconds} с", flush=True)
