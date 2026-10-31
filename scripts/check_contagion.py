"""Пилотная проверка графа заражения на синтетических S0/S6.

Агрегирует наблюдаемую/ожидаемую совместную частоту событий по доменам и
проверяет, что исторический снимок после полного прогона совпадает со снимком,
снятым вживую на той же дате (нет утечки будущих наблюдений).

    python scripts/check_contagion.py [число зёрен, default 3] [дни, default 365]
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from taris_sim.config import scenario
from taris_sim.scenario import MAIN
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import UniformPolicy
from taris_sim.contagion import ContagionGraph


class Collector(UniformPolicy):
    def __init__(self, world, t_check):
        super().__init__(n=13)
        self.cg = ContagionGraph(world, tau=1.0)
        self.t_check = float(t_check)
        self.at_check = None

    def on_epoch(self, t, view):
        self.cg.update(t, view)
        if abs(t - self.t_check) < 1e-8:
            s = self.cg.snapshot(t)
            self.at_check = (s.O.copy(), s.E.copy(), s.W.copy())
        return super().on_epoch(t, view)


def summarize(w, snap):
    nd = w.nodes
    dc = nd["dc"].to_numpy()
    prov = nd["provider"].to_numpy()
    ii, jj = np.indices(snap.O.shape)
    other_dc = (dc[:, None] != dc[None, :]) & (dc[:, None] >= 0) & (dc[None, :] >= 0)
    groups = {
        "внутри ДЦ": (snap.same_dc & ~np.eye(len(dc), dtype=bool)),
        "разные ДЦ, один провайдер": other_dc & (prov[:, None] == prov[None, :]),
        "разные провайдеры": other_dc & (prov[:, None] != prov[None, :]),
    }
    out = {"events": int(snap.O.sum()), "positive_pairs": int((snap.O > 0).sum())}
    for label, mask in groups.items():
        o, e = float(snap.O[mask].sum()), float(snap.E[mask].sum())
        out[label] = (o, e, o / e if e else float("nan"))
    return out


def run(which, seed, days):
    cfg = scenario(which, seed=seed, horizon_days=days, **MAIN)
    w = World(cfg)
    L = Ledger(w, Chain(w), DataStream(cfg, len(w.nodes)))
    t_check = min(200, days - 1)
    pol = Collector(w, t_check)
    L.run(pol)
    live = pol.at_check
    late = pol.cg.snapshot(t_check)
    no_future = bool(live and all(np.array_equal(a, b) for a, b in zip(live, (late.O, late.E, late.W))))
    final = pol.cg.snapshot(days)
    print(f"{which} seed={seed} дней={days} events={final.O.sum():.0f}; "
          f"no_future_at_{t_check}={no_future}; groups={summarize(w, final)}")
    return no_future


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    checks = [run(world, seed, days) for world in ("S0", "S6") for seed in range(1, seeds + 1)]
    print("ИТОГ: исторические снимки без утечки" if all(checks) else "ИТОГ: ошибка проверки будущих данных")
    if not all(checks):
        raise SystemExit(1)
