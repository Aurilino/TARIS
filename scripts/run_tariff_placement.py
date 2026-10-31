"""Короткий пилот тарифного размещения против равномерного и агентского.

    python scripts/run_tariff_placement.py [workers=3] [days=180] [seed-range=1-3]

Денежные параметры здесь — открытые пробные значения для проверки поведения механики,
не калибровка и не рекомендуемый тариф. Мир строится по неизменённой MAIN-конфигурации.
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

PROBE = TariffSettings(
    basis=PriceBasis(g=20.0, u=1.0),
    theta=0.2, eta=2.0, zeta=1.0, reward0=0.3, psi=1.0,
)
OUT = os.path.join(ROOT, "runs")


def run(job):
    scn, seed, kind, days = job
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
    opts = {"tariff_settings": PROBE} if kind == "tariff" else {}
    policy = ReservePolicy(world, 13, mode="fixed", R_fixed=2, rates="agent", placement=kind, **opts)
    started = time.time()
    result = ledger.run(policy)
    tv = policy.tariff_values
    eligibility = float(tv["financial"]["new_data_eligible"].mean()) if tv else 1.0
    gross_mean = float(tv["gross"]["gross_premium"].mean()) if tv else float("nan")
    out = {"job": job, "metrics": result.metrics(), "losses": result.losses,
           "placement": kind, "tariff_eligible_share": eligibility,
           "mean_gross_tariff": gross_mean, "probe_settings": PROBE if kind == "tariff" else None,
           "seconds": round(time.time() - started)}
    path = os.path.join(OUT, f"tariff_placement_{days}", f"{scn}_{kind}_s{seed}.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return scn, seed, kind, out["metrics"]["Потеряно объектов"], \
        out["metrics"]["Трафик восстановления в год"], eligibility, gross_mean, out["seconds"]


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
    out_dir = os.path.join(OUT, f"tariff_placement_{days}")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(scn, seed, kind, days) for scn in ("S0", "S2") for seed in range(s0, s1 + 1)
            for kind in ("uniform", "agent", "tariff")]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(out_dir, f"{j[0]}_{j[2]}_s{j[1]}.pkl"))]
    print(f"Прогонов к расчёту: {len(jobs)}, дней={days}, зёрна={seed_range}, workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            scn, seed, kind, lost, traffic, eligible, gross, seconds = future.result()
            print(f"{scn} s{seed} {kind}: потерь {lost}, ремонтный трафик {traffic:.1f}/год, "
                  f"допуск {eligible:.1%}, брутто {gross:.4f}, {seconds} c", flush=True)
