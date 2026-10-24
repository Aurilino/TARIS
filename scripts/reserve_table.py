"""Выбор (n*, R*_min) по актуарному резерву (шаг 3, пункт 3.11; шаг 2, пункт 2.5) на объекте основного мира.

Объект: набор из n узлов, выбранный оракулом (лимиты разнесения), в момент t0; горизонт хранения H суток.
TARIS-резерв: интенсивности меняются по эпохам (старение узлов), выбор — минимум V_n(0) при P_loss <= eps.
Марковский конкурент (H7, вариант «в»): та же цепь с постоянной интенсивностью на момент размещения,
выбор — минимум трафика восстановления при P_loss <= eps по его же модели; его выбор затем оценивается
моделью с меняющимися интенсивностями.

    python scripts/reserve_table.py [S2|S0|S6] [matched|mismatched] [t0] [H] [eps]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from taris_sim.config import scenario
from taris_sim.scenario import MAIN
from taris_sim.world import World
from taris_sim.policy import OraclePolicy
from taris_sim import reserve as RS

scn = sys.argv[1] if len(sys.argv) > 1 else "S2"
mode = sys.argv[2] if len(sys.argv) > 2 else "matched"
t0 = float(sys.argv[3]) if len(sys.argv) > 3 else 400.0
H = int(sys.argv[4]) if len(sys.argv) > 4 else 365
eps = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4
pd.set_option("display.width", 200)

w = World(scenario(scn, world_mode=mode, **MAIN))
R = RS.TrueRates(w)
k = w.cfg.k
alive = w.alive_at(np.arange(len(w.nodes)), t0)
sets, const = {}, {}
for n in range(k + 1, 2 * k + 1):
    pol = OraclePolicy(w, n)
    nodes = pol._choose_spread(np.random.default_rng(3), alive, n)
    if len(nodes) < n:
        continue
    lam, sh = RS.object_rates(R, nodes, t0, H)
    mu = RS.object_mu(R, nodes, t0, sh)
    sets[n] = (lam, sh, mu)
    const[n] = (np.full(H, lam[0]), sh, mu)
    print(f"n = {n}: интенсивность потери фрагмента {lam[0]:.4f} → {lam[-1]:.4f} в сутки, "
          f"шоков {len(sh)}, больше всего в кластере {max((m for _, m in sh), default=0)}, задержка {1 / mu:.2f} сут", flush=True)

tab = RS.evaluate_grid(k, sets)
tab_c = RS.evaluate_grid(k, const)
best = RS.choose(tab, eps, "cost")
best_c = RS.choose(tab_c, eps, "traffic")
true_c = tab[(tab["n"] == best_c["n"]) & (tab["R_min"] == best_c["R_min"])].iloc[0]
show = tab.assign(P_loss=tab["P_loss"].map("{:.2e}".format), traffic=tab["traffic"].round(2), cost=tab["cost"].round(2))
print(f"\n{scn}, {mode}, t0 = {t0:g}, H = {H} суток, eps = {eps:g}")
print(show.to_string(index=False))
print(f"\nTARIS-резерв: n* = {best['n']:.0f}, R*_min = {best['R_min']:.0f}, P_loss = {best['P_loss']:.2e}, "
      f"трафик {best['traffic']:.2f}, стоимость {best['cost']:.2f}" + ("" if best["feasible"] else " — eps недостижим"))
print(f"Марковский конкурент: n = {best_c['n']:.0f}, R_min = {best_c['R_min']:.0f}; по своей модели P_loss = "
      f"{best_c['P_loss']:.2e}, трафик {best_c['traffic']:.2f}; по модели с меняющимися интенсивностями P_loss = "
      f"{true_c['P_loss']:.2e}, трафик {true_c['traffic']:.2f}, стоимость {true_c['cost']:.2f}"
      + ("" if best_c["feasible"] else " — eps недостижим"))
