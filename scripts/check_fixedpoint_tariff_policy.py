"""Сравнение тарифного размещения с float- и fixed-point денежной частью.

    python scripts/check_fixedpoint_tariff_policy.py [days=90] [seed=1] [S0|S2]
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from scripts.run_tariff_placement import PROBE


def run(days: int, seed: int, scn: str, fixed: bool):
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
    policy = ReservePolicy(world, 13, mode="fixed", R_fixed=2, rates="agent",
                           placement="tariff", tariff_settings=PROBE,
                           fixedpoint_tariff=fixed)
    result = ledger.run(policy)
    return result, policy


def main():
    args = [x for x in sys.argv[1:] if x.isdigit()]
    days = int(args[0]) if args else 90
    seed = int(args[1]) if len(args) > 1 else 1
    scn = next((x for x in sys.argv[1:] if x in ("S0", "S2")), "S2")
    if days < 1:
        raise SystemExit("days должен быть положительным")

    floating, pf = run(days, seed, scn, False)
    fixed, pq = run(days, seed, scn, True)
    mf, mq = floating.metrics(), fixed.metrics()
    af = pf.tariff_values["financial"]["new_data_eligible"]
    aq = pq.tariff_values["financial"]["new_data_eligible"]
    elig_diff = int(np.count_nonzero(af != aq))
    elig_n = len(af)

    # Сопоставляем исходные наборы фрагментов объектов, начатых после первого обучения.
    def placements(result):
        fr = result.fragments
        fr = fr[(fr["t_from"] >= 30) & (fr["idx"] >= 0)]
        return {int(obj): tuple(sorted(g["node"].astype(int)))
                for obj, g in fr.groupby("obj", sort=False)}

    xf, xq = placements(floating), placements(fixed)
    shared = sorted(set(xf) & set(xq))
    equal = sum(xf[o] == xq[o] for o in shared)
    ldiff = int(mf["Потеряно объектов"] - mq["Потеряно объектов"])
    print(f"{scn}, seed={seed}, days={days}")
    print(f"eligibility: узлов {elig_n}, расхождений {elig_diff}")
    print(f"размещения после дня 30: общих объектов {len(shared)}, одинаковых наборов {equal}")
    print(f"float: потери={mf['Потеряно объектов']}, EAFDL={mf['EAFDL (среднее по годам)']:.9g}, "
          f"трафик={mf['Трафик восстановления в год']:.3f}, доля допуска={float(af.mean()):.6f}")
    print(f"fixed: потери={mq['Потеряно объектов']}, EAFDL={mq['EAFDL (среднее по годам)']:.9g}, "
          f"трафик={mq['Трафик восстановления в год']:.3f}, доля допуска={float(aq.mean()):.6f}")
    print(f"разность потерь float-fixed={ldiff}; policy label={pq.name}")


if __name__ == "__main__":
    main()
