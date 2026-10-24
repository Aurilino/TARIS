"""Проверки графа связей (шаг 2, пункт 2.4; визуализация, этап 5.5).

1. Корреляция: rho_hat совпадает с numpy.corrcoef по общим эпохам наблюдения на выборке пар.
2. Без истины: результат не меняется, если перемешать истинные поля узлов (владелец, истинный класс, риск).
3. Без будущего: снимок в момент t не меняется, если удалить все наблюдения после t.
4. Смысл: в конце горизонта пары из одного истинного дата-центра связаны сильнее, чем из разных (z-тест).
5. Воспроизводимость: два расчёта побитово равны.

    python scripts/check_graph.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import copy
import numpy as np
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.graph import LinkGraph

MAIN = dict(n_initial_nodes=200, node_join_rate=0.12, blocks_per_epoch=4, n_dc=16, n_providers=6, n_regions=5)
ok_all = True


def report(name, ok, extra=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{name}: {'OK' if ok else 'ОШИБКА'}{(' — ' + extra) if extra else ''}")


def same(a, b) -> bool:
    return a.pairs.equals(b.pairs) and np.array_equal(a.suspicion, b.suspicion, equal_nan=True) \
        and np.array_equal(a.clusters, b.clusters) and np.array_equal(a.clusters_infra, b.clusters_infra)


for name, over in (("основной мир", MAIN), ("полный мир", {})):
    print(f"\n=== {name}")
    w = World(scenario("S6", **over)); ch = Chain(w); nd = w.nodes
    g = LinkGraph(w, ch)
    T = float(w.cfg.horizon_days)

    # 1. корреляция против numpy.corrcoef на 300 случайных парах с определённой корреляцией
    rho_hat, n_inf, Z, rho = g.infrastructure(T)
    rng = np.random.default_rng(0)
    checked, bad = 0, 0
    while checked < 300:
        i, l = rng.integers(0, g.N, 2)
        if i == l:
            continue
        common = (g.M[i] > 0) & (g.M[l] > 0)
        yi, yl = g.Y[i, common], g.Y[l, common]
        if common.sum() < 2 or yi.std() == 0 or yl.std() == 0:
            continue
        checked += 1
        bad += abs(np.corrcoef(yi, yl)[0, 1] - rho_hat[i, l]) > 1e-4
    report("1. Корреляция совпадает с numpy.corrcoef (300 пар)", bad == 0, f"расхождений {bad}" if bad else "")

    # 2. без истины: перемешиваем истинные поля
    s0 = g.snapshot(T)
    w2 = copy.copy(w); w2.nodes = nd.copy()
    perm = rng.permutation(len(nd))
    for col in ["owner", "hw_true", "theta", "lam_avg", "lam_micro", "sev_u"]:
        w2.nodes[col] = nd[col].to_numpy()[perm]
    report("2. Граф не использует истину (перемешаны владелец, класс, риск)", same(s0, LinkGraph(w2, ch).snapshot(T)))

    # 3. без будущего: снимок в t = 400 по наблюдениям, обрезанным на t
    t = 400.0
    ch2 = copy.copy(ch)
    ob = ch.observed[ch.observed["start"] < t].copy()
    ob["end"] = ob["end"].clip(upper=t)            # конец ещё не закончившегося простоя в момент t неизвестен
    ob["dur"] = ob["end"] - ob["start"]
    ch2.observed = ob
    s_full, s_cut = g.snapshot(t), LinkGraph(w, ch2).snapshot(t)
    report("3. Снимок в момент t не зависит от наблюдений после t", same(s_full, s_cut))

    # 4. смысл: пары из одного истинного ДЦ связаны сильнее (Уэлч, без учёта зависимости пар)
    reg = np.flatnonzero(g.join <= T)
    iu, lu = np.triu_indices(len(reg), 1); I, L = reg[iu], reg[lu]
    dc = nd["dc"].to_numpy(); sd = dc[I] == dc[L]
    a, b = rho[I[sd], L[sd]], rho[I[~sd], L[~sd]]
    z = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
    report("4. В одном ДЦ связь сильнее, чем в разных", z > 3, f"средняя rho~ {a.mean():.3f} против {b.mean():.3f}, z = {z:.1f}")

    # 5. воспроизводимость
    report("5. Воспроизводимость", same(s0, LinkGraph(w, ch).snapshot(T)))

print("\nИтог:", "все проверки пройдены" if ok_all else "ЕСТЬ ОШИБКИ")
sys.exit(0 if ok_all else 1)
