"""Проверки ограничений защиты и отсутствия изменений в выключенном режиме."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import numpy as np
import pandas as pd
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import ReservePolicy
from taris_sim.protection import ProtectedReservePolicy


def warm_data(cfg, world, warmup=30):
    data = DataStream(cfg, len(world.nodes))
    data.objects = data.objects[data.objects.created_epoch >= warmup].reset_index(drop=True)
    data.objects.index.name = "obj"
    return data


def main():
    cfg = scenario("S3", seed=31, horizon_days=45, train_days=30, n_initial_nodes=70,
                   n_dc=20, n_providers=6, n_regions=5, blocks_per_epoch=1, k=3,
                   adv_sybil_share=.3, adv_attack_day=38.)
    w = World(cfg)
    c = Chain(w)
    data = warm_data(cfg, w)
    baseline = Ledger(w, c, data).run(ReservePolicy(w, n=5, mode="fixed", R_fixed=1, rates="agent", placement="agent_greedy"))
    disabled = Ledger(w, c, data).run(ProtectedReservePolicy(w, n=5, R_fixed=1, graph_cap=False, novice_cap=False, link_penalty=0))
    for key in ("daily", "fragments", "losses", "repairs", "disclosures"):
        pd.testing.assert_frame_equal(getattr(baseline, key), getattr(disabled, key))
    print("OK: выключенная защита побитово повторяет базового агента", flush=True)

    # Независимая маленькая задача выбора: искусственно заданы публичные домены,
    # оценки Z и веса; после выбора ограничения пересчитываются по исходным массивам.
    p = ProtectedReservePolicy(w, n=5, R_fixed=1)
    N = len(p.w)
    p.dc = np.arange(N)
    p.prov = np.arange(N)
    p.reg = np.arange(N)
    p.own = np.arange(N)
    p.subnet_domain = np.arange(N)
    p.n_dc = p.n_prov = p.n_reg = p.n_own = p.n_subnet = N
    labels = np.full(N, -1)
    labels[:4] = 0
    p._guard_labels = labels
    p._guard_conf = np.zeros((N, N))
    p._guard_t = 30.
    p._guard_snapshot = SimpleNamespace(t=30.)
    p.model.fitted = True
    Z = np.ones(N)
    Z[::2] = 0.
    p.model.Z = Z
    p.w = np.arange(N, 0, -1, dtype=float)
    selected = p._choose_spread(np.random.default_rng(1), np.arange(10), 5)
    assert len(selected) == 5
    assert sum(Z[selected] < .1) <= 2
    assert sum(labels[selected] == 0) <= 2
    print("OK: жёсткие лимиты по группе и новичкам проверены независимо от журнала политики", flush=True)
    # Ремонт также учитывает уже существующих держателей.
    selected = p._choose_spread(np.random.default_rng(1), np.arange(10), 2, held=[0, 1, 5])
    assert len(selected) == 2 and not set(selected).intersection([0, 1, 5])
    assert not set(selected).intersection([2, 3])
    print("OK: ограничения ремонта учитывают удерживаемые фрагменты", flush=True)
    p.model.Z = np.zeros(N)
    assert p._choose_spread(np.random.default_rng(1), np.arange(10), 5) == []
    print("OK: недопустимый набор не выдаётся за допустимое размещение", flush=True)
    p.novice_cap = False
    p.graph_cap = False
    p.w[:] = 1.
    p._guard_conf[0, 1] = p._guard_conf[1, 0] = 1.
    selected = p._choose_spread(np.random.default_rng(1), np.arange(5), 2)
    assert selected == [0, 2]
    print("OK: штраф изменяет совместный выбор связанных узлов", flush=True)

    # Сквозной запуск с настоящим сетевым графом, до истечения будущей атаки.
    policy = ProtectedReservePolicy(w, n=5, R_fixed=1, novice_threshold=0.)
    result = Ledger(w, c, data).run(policy)
    assert policy.guard_log
    for row in policy.guard_log:
        assert row["snapshot_t"] <= row["t"]
        assert row["max_group"] <= row["group_cap"]
        assert row["novices"] <= row["novice_cap"]
    assert (result.objects.created_epoch >= 30).all()
    print("OK: сквозной прогон, период наблюдения без объектов, снимки не из будущего", flush=True)


if __name__ == "__main__":
    main()
