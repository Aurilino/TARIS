"""Чувствительность тарифного размещения к базовому вознаграждению.

    python scripts/run_tariff_sensitivity.py [workers=3] [days=180] [seed-range=1-3]

Мир и остальные цены фиксированы; reward0 варьируется вокруг пробного значения.
Это диагностический пилот, а не калибровка тарифа.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from taris_sim.tariff import PriceBasis, TariffSettings

SETTINGS = {
    "low": TariffSettings(PriceBasis(g=20.0, u=1.0), 0.2, 2.0, 1.0, 0.15, 1.0),
    "base": TariffSettings(PriceBasis(g=20.0, u=1.0), 0.2, 2.0, 1.0, 0.30, 1.0),
    "high": TariffSettings(PriceBasis(g=20.0, u=1.0), 0.2, 2.0, 1.0, 0.60, 1.0),
}
OUT = os.path.join(ROOT, "runs")


def run(job):
    scn, seed, label, days = job
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    from taris_sim.policy import ReservePolicy

    cfg = scenario(scn, seed=seed, horizon_days=days, **MAIN)
    world = World(cfg)
    ledger = Ledger(world, Chain(world), DataStream(cfg, len(world.nodes)))
    settings = SETTINGS[label]
    policy = ReservePolicy(world, 13, mode="fixed", R_fixed=2, rates="agent",
                           placement="tariff", tariff_settings=settings)
    started = time.time()
    result = ledger.run(policy)
    tv = policy.tariff_values
    out = {
        "job": job, "metrics": result.metrics(), "losses": result.losses,
        "placement": "tariff", "price_case": label, "reward0": settings.reward0,
        "tariff_eligible_share": float(tv["financial"]["new_data_eligible"].mean()) if tv else float("nan"),
        "mean_gross_tariff": float(tv["gross"]["gross_premium"].mean()) if tv else float("nan"),
        "settings": settings, "seconds": round(time.time() - started),
    }
    out_dir = os.path.join(OUT, f"tariff_sensitivity_{days}")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{scn}_{label}_s{seed}.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return scn, seed, label, out["metrics"]["Потеряно объектов"], \
        out["metrics"]["Трафик восстановления в год"], out["tariff_eligible_share"], out["seconds"]


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


if __name__ == "__main__":
    keep_awake()
    args = [x for x in sys.argv[1:] if x.isdigit()]
    workers = int(args[0]) if args else 3
    days = int(args[1]) if len(args) > 1 else 180
    seed_range = next((x for x in sys.argv[1:] if "-" in x and x.split("-")[0].isdigit()), "1-3")
    s0, s1 = map(int, seed_range.split("-"))
    out_dir = os.path.join(OUT, f"tariff_sensitivity_{days}")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(scn, seed, label, days) for scn in ("S0", "S2") for seed in range(s0, s1 + 1)
            for label in SETTINGS]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(out_dir, f"{j[0]}_{j[2]}_s{j[1]}.pkl"))]
    print(f"Прогонов: {len(jobs)}, дней={days}, зёрна={seed_range}, workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            scn, seed, label, lost, traffic, eligible, seconds = future.result()
            print(f"{scn} s{seed} {label}: потери {lost}, трафик {traffic:.1f}/год, "
                  f"допуск {eligible:.1%}, {seconds} c", flush=True)
