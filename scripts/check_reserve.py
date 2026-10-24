"""Проверки модуля reserve (шаг 3, пункт 3.11).

1. Истинные интенсивности против мира: число простоев длиннее T_rep, уходов, тихих потерь и шоков,
   уносящих фрагменты, против ожидаемого по TrueRates (z-тесты); основной мир S2 в обоих режимах и S8.
2. Вероятность потери за окно: пуассон-биномиальная рекурсия против полного перебора; P_loss с шоками
   против Монте-Карло при фиксированном наборе узлов; отброшенная часть суммы по шокам не больше оценки.

    python scripts/check_reserve.py
"""
import sys, os
from itertools import product
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from taris_sim.config import scenario
from taris_sim.scenario import MAIN
from taris_sim.world import World
from taris_sim import reserve as RS

ok_all = True
pd.set_option("display.width", 200)


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'ok' if ok else 'ОШИБКА'}] {name}" + (f": {detail}" if detail else ""))


def ztest(name, obs, exp, var=None, lim=3.0):
    var = exp if var is None else var
    z = (obs - exp) / np.sqrt(max(var, 1e-12))
    check(name, abs(z) < lim, f"наблюдалось {obs:.0f}, ожидалось {exp:.1f}, z = {z:+.2f}")


def rates_vs_world(scn, mode):
    print(f"\n1. Интенсивности против мира: {scn}, {mode}")
    w = World(scenario(scn, world_mode=mode, **MAIN))
    c, nd = w.cfg, w.nodes
    R = RS.TrueRates(w)
    H = c.horizon_days
    a0 = nd["alive_from"].to_numpy()
    a1 = nd["alive_to"].to_numpy()
    # простои длиннее T_rep: по сырому потоку собственных простоев (до слияния), интеграл lambda_i * phi_i
    E_own = w.expected_own_counts() * R.phi
    O_own = int((w.raw_own["dur"] > c.T_rep).sum())
    ztest("собственные простои длиннее T_rep", O_own, E_own.sum())
    if R.storm:
        # справочно: вклад бурь в TrueRates — приближение (простои внутри бури считаются независимыми);
        # в мире они сливаются при перекрытии, и длинных простоев больше
        st = w.outages[(w.outages["kind"] == "storm")]
        obs, exp = int(((st["end"] - st["start"]) > c.T_rep).sum()), float((E_own * R.storm_factor).sum())
        print(f"  [справочно] простои бурь длиннее T_rep (после слияния): наблюдалось {obs}, по приближению {exp:.1f}, "
              f"отношение {obs / max(exp, 1e-9):.2f}")
    # уход: вероятность уйти за базовую жизнь по интенсивности ухода (часть churn в TrueRates)
    exp_l = var_l = 0.0
    for i in range(len(nd)):
        if a1[i] <= a0[i] and nd["alive_to_base"].iat[i] <= a0[i]:
            continue
        t = np.linspace(a0[i], H, 300)
        if t[-1] <= t[0]:
            continue
        lam = np.array([R.parts([i], x)["churn"][0] for x in t[::30]])
        lam = np.interp(t, t[::30], lam)
        p = 1.0 - np.exp(-np.trapezoid(lam, t))
        exp_l += p; var_l += p * (1 - p)
    ztest("уходы из сети (базовая жизнь)", int(np.isfinite(nd["leave_t_base"].to_numpy()).sum()), exp_l, var_l)
    expo = np.clip(a1 - a0, 0, None)
    ztest("тихие потери", len(w.silent), float((expo * R.lam_sil).sum()))
    # шоки, уносящие фрагменты (полная гибель или простой дольше T_rep), по типам кластеров
    if not R.storm and c.shocks_enabled:
        sh = w.shocks
        for ti, ct in enumerate(RS.CLUSTER_TYPES):
            n_cl = len([x for x in np.unique(nd[ct]) if x >= 0])
            obs = int(((sh["ctype"] == ct) & (sh["destroyed"] | ((sh["end"] - sh["start"]) > c.T_rep))).sum())
            ztest(f"шоки {ct}, уносящие фрагменты", obs, R.eta_type[ti] * H * n_cl)
    return w, R


def window_checks(w, R):
    print("\n2. Вероятность потери за окно")
    g = np.random.default_rng(7)
    # рекурсия против полного перебора (n = 10)
    q = g.uniform(0.01, 0.4, 10)
    G = RS.poisson_binomial(q)
    brute = np.zeros(11)
    for bits in product((0, 1), repeat=10):
        b = np.array(bits)
        brute[b.sum()] += np.prod(np.where(b, q, 1 - q))
    check("рекурсия = полный перебор (n = 10)", np.allclose(G, brute, atol=1e-14), f"макс. расхождение {abs(G - brute).max():.1e}")
    # фиксированный набор: 13 живых узлов в разных дата-центрах, окно растянуто до 60 суток, чтобы потеря была заметна
    c = w.cfg
    t = 400.0
    alive = w.alive_at(np.arange(len(w.nodes)), t)
    dc = w.nodes["dc"].to_numpy()
    pick, seen = [], set()
    for i in g.permutation(alive):
        if dc[i] not in seen:
            pick.append(int(i)); seen.add(dc[i])
        if len(pick) == 13:
            break
    pick = np.array(pick)
    tau = 60.0
    q, masks, h = RS.window_probs(R, pick, t, tau)
    # шоки сильнее, чем в мире (h = 0,02 у каждого кластера), но так, чтобы три шока сразу были редки
    h = np.full(len(h), 0.02)
    p, dropped = RS.p_loss_window_shocks(q, c.k, masks, h)
    p3, _ = RS.p_loss_window_shocks(q, c.k, masks, h, max_shocks=3)
    check("отброшенная часть мала", dropped < 0.02, f"{dropped:.1e}")
    M = 200_000
    dead = g.random((M, len(q))) < q
    for m, hc in zip(masks, h):
        dead |= (g.random(M) < hc)[:, None] & m[None, :]
    mc = (dead.sum(1) > len(q) - c.k).mean()
    se = np.sqrt(mc * (1 - mc) / M)
    print(f"  набор: {len(pick)} узлов, кластеров с шоками {len(h)}, q от {q.min():.3f} до {q.max():.3f}, h до {h.max():.3f}")
    check("P_loss с шоками против Монте-Карло", abs(p - mc) < 3 * se + dropped,
          f"формула {p:.5f}, Монте-Карло {mc:.5f} ± {se:.5f}, отброшено не больше {dropped:.1e}")
    check("три шока сразу: добавка не больше оценки отброшенного", 0 <= p3 - p <= dropped + 1e-15, f"{p3 - p:.2e} ≤ {dropped:.2e}")
    p0 = RS.p_loss_window(q, c.k)
    mc0 = ((g.random((M, len(q))) < q).sum(1) > len(q) - c.k).mean()
    se0 = np.sqrt(max(mc0 * (1 - mc0), 1e-12) / M)
    check("P_loss без шоков против Монте-Карло", abs(p0 - mc0) < 3 * se0 + 1e-6, f"формула {p0:.5f}, Монте-Карло {mc0:.5f} ± {se0:.5f}")
    # реальное окно: T_rep + восстановление, истинные h
    q, masks, h = RS.window_probs(R, pick, t, RS.window_tau(c))
    p, dropped = RS.p_loss_window_shocks(q, c.k, masks, h)
    print(f"  справочно, реальное окно {RS.window_tau(c) * 24:.0f} ч: P_loss = {p:.2e} (без шоков {RS.p_loss_window(q, c.k):.2e}), "
          f"отброшено ≤ {dropped:.1e}")


w, R = rates_vs_world("S2", "matched")
window_checks(w, R)
rates_vs_world("S2", "mismatched")
rates_vs_world("S8", "matched")

print("\nИтог:", "все проверки пройдены" if ok_all else "есть ошибки")
sys.exit(0 if ok_all else 1)
