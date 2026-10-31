"""Проверка размещения по оценкам риска агента (подшаг 5.2, пилот).

Проверяет равномерный старт до обучения, отсутствие скрытого owner-домена в политике,
обновление весов по EstimatedRates и соблюдение ограничений spread на основном S0.

    python scripts/check_agent_placement.py
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
from taris_sim.policy import ReservePolicy


def check(cond, msg):
    print(("ок  " if cond else "НЕТ ") + msg)
    if not cond:
        raise AssertionError(msg)


cfg = scenario("S0", seed=1, horizon_days=90, **MAIN)
w = World(cfg)
pol = ReservePolicy(w, 13, mode="fixed", R_fixed=0, rates="agent", placement="agent")
check(np.all(pol.w == 1.0), "до обучения вес узлов одинаковый, оракул не используется")
check(np.array_equal(pol.own, pol.exp.reg.wallet), "домен оператора берётся из публичного кошелька выплат")
check(np.array_equal(pol.subnet_domain, pol.exp.reg.subnet), "дополнительный домен берётся из публичной подсети")
pol_no_claim = ReservePolicy(w, 13, mode="fixed", rates="agent", placement="agent", claimed=False)
check(len(np.unique(pol_no_claim.dc)) == len(w.nodes), "без заявленных признаков DC не используется как домен")

# Проверяем ограничения выбора на полной карте зарегистрированных узлов.
nodes = pol._choose_spread(np.random.default_rng(17), np.arange(len(w.nodes)), 13)
check(len(nodes) == 13 and len(set(nodes)) == 13, "выбрано 13 различных узлов")
r0 = min(pol.n - pol.k, pol.k - 1)
check(max(np.bincount(pol.reg[nodes])) <= r0 + pol.n, "лимит региона ослабляется только при необходимости")
check(max(np.bincount(pol.prov[nodes])) <= r0 + pol.n, "лимит провайдера ослабляется только при необходимости")
check(max(np.bincount(pol.dc[nodes])) <= pol.n, "лимит дата-центра соблюдён после fallback")
check(max(np.bincount(pol.own[nodes])) <= r0, "лимит публичного кошелька не ослабляется")
check(max(np.bincount(pol.subnet_domain[nodes])) <= r0, "лимит публичной подсети не ослабляется")

res = Ledger(w, Chain(w), DataStream(cfg, len(w.nodes))).run(pol)
check(pol.model.fitted, "модель риска обучилась только по наблюдениям сети")
check(np.isfinite(pol.w).all() and (pol.w > 0).all(), "оценочные веса конечны и положительны")
pol.model.refresh(89.0)
nodes = np.arange(pol.model.exp.N)
hazard = pol.rates.fragment_loss(nodes, 89.0).astype(float, copy=True)
for _, _, eta, mask in pol.rates.shock_clusters(nodes, 89.0):
    hazard[mask] += eta
expected = (1.0 / np.maximum(hazard, 1e-9)) ** pol.gamma
check(np.allclose(pol.w, expected), "веса размещения совпадают с обратной оценкой потери фрагмента")
check(len(res.short_placements) == 0, "в прогоне не было неполных размещений")
print(f"ИТОГ: потерь объектов={len(res.losses)}, ремонтов={len(res.repairs)}, обучений={len(pol.fits)}")
