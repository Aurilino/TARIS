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


def gillespie(n, k, lam, shocks, R_min, mu, H, M, g):
    """Монте-Карло той же цепи объекта: доля потерянных до H и средний трафик восстановления (в объёмах объекта)."""
    lost, traffic = 0, 0.0
    for _ in range(M):
        s, t = n, 0.0
        while True:
            rep = mu if (s < n and s - k <= R_min) else 0.0
            rates = np.array([s * lam] + [eta for eta, _ in shocks] + [rep])
            tot = rates.sum()
            t += g.exponential(1.0 / tot)
            if t >= H:
                break
            j = int(np.searchsorted(np.cumsum(rates), g.random() * tot, side="right"))
            if j == 0:
                s -= 1
            elif j <= len(shocks):
                s -= shocks[j - 1][1]
            else:
                traffic += (k + n - s) / k
                s = n
            if s < k:
                lost += 1
                break
    return lost / M, traffic / M


def thiele_checks(w, R):
    from scipy.linalg import expm
    print("\n3. Уравнение Тиле")
    c = w.cfg
    n, k, mu = 13, c.k, RS.repair_mu(c)
    # (а) при нулевых затратах и W = 1 — обратное уравнение Колмогорова: P(потерян к H) = [exp(Q H)]_{n, потерян}
    lam, H, shocks = 0.02, 365, [(0.004, 3), (0.002, 2)]
    for R_min in (0, 1, 2):
        V = RS.thiele_const(n, k, lam, H, shocks, R_min, mu, RS.Costs.loss_probability())
        Q, _ = RS.generator(n, k, lam, shocks, R_min, mu, RS.Costs.loss_probability())
        kol = expm(Q * H)[-1, 0]
        check(f"Тиле = Колмогоров, R_min = {R_min}", abs(V[0, -1] - kol) < 1e-10 * max(kol, 1e-30) + 1e-15,
              f"{V[0, -1]:.6e} против {kol:.6e}")
    # (б) интенсивности, меняющиеся во времени: две половины горизонта = произведение экспонент
    lam_t = np.r_[np.full(100, 0.01), np.full(100, 0.05)]
    V = RS.thiele(n, k, lam_t, [], 1, mu, RS.Costs.loss_probability())
    Q1, _ = RS.generator(n, k, 0.01, [], 1, mu, RS.Costs.loss_probability())
    Q2, _ = RS.generator(n, k, 0.05, [], 1, mu, RS.Costs.loss_probability())
    kol = (expm(Q1 * 100) @ expm(Q2 * 100))[-1, 0]
    check("меняющиеся интенсивности: Тиле = произведение экспонент", abs(V[0, -1] - kol) < 1e-10 * kol, f"{V[0, -1]:.6e} против {kol:.6e}")
    # (в) Монте-Карло той же цепи: интенсивности завышены, чтобы потери были заметны
    g = np.random.default_rng(11)
    lam, H, shocks = 0.03, 200, [(0.003, 3)]
    for R_min in (0, 2):
        m = RS.object_metrics(n, k, np.full(H, lam), shocks, R_min, mu)
        p, tr = gillespie(n, k, lam, shocks, R_min, mu, H, 20000, g)
        se = np.sqrt(p * (1 - p) / 20000)
        check(f"Монте-Карло цепи, R_min = {R_min}: вероятность потери", abs(m["P_loss"] - p) < 3 * se + 1e-4,
              f"Тиле {m['P_loss']:.4f}, Монте-Карло {p:.4f} ± {se:.4f}")
        check(f"Монте-Карло цепи, R_min = {R_min}: трафик восстановления", abs(m["traffic"] - tr) < 0.03 * m["traffic"],
              f"Тиле {m['traffic']:.2f}, Монте-Карло {tr:.2f} объёма")
    # (г) справочно: объект на 13 узлах основного мира, выбранных оракулом (лимиты разнесения), горизонт 365 суток
    from taris_sim.policy import OraclePolicy
    t0 = 400.0
    alive = w.alive_at(np.arange(len(w.nodes)), t0)
    pick = OraclePolicy(w, n)._choose_spread(np.random.default_rng(3), alive, n)
    lam_t, sh = RS.object_rates(R, pick, t0, 365)
    mu_obj = RS.object_mu(R, pick, t0, sh)
    print(f"  справочно: объект на 13 узлах оракула (S2, сутки {t0:g}), средняя интенсивность потери фрагмента "
          f"{lam_t.mean():.4f} в сутки, шоков кластеров {len(sh)}, больше всего фрагментов в одном кластере "
          f"{max(m for _, m in sh)}; средняя задержка восстановления {1 / mu_obj:.2f} сут (консервативная {1 / mu:.2f})")
    rows = []
    for R_min in range(0, n - k):
        m = RS.object_metrics(n, k, lam_t, sh, R_min, mu_obj)
        rows.append({"R_min": R_min, "P_loss за год": f"{m['P_loss']:.2e}", "трафик, объёмов": round(m["traffic"], 3),
                     "стоимость": round(m["cost"], 3)})
    print(pd.DataFrame(rows).to_string(index=False))


w, R = rates_vs_world("S2", "matched")
window_checks(w, R)
thiele_checks(w, R)
rates_vs_world("S2", "mismatched")
rates_vs_world("S8", "matched")

print("\nИтог:", "все проверки пройдены" if ok_all else "есть ошибки")
sys.exit(0 if ok_all else 1)
