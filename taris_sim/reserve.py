"""Резерв информации (шаг 2, пункт 2.5; шаг 3, пункт 3.11).

Расчётный аппарат политики: интенсивности потери фрагментов, вероятность потери объекта за окно,
модель объекта и уравнение Тиле, выбор (n, R_min), поправки IBNR и RBNS. Сам модуль ничего не размещает.

Интенсивности — через общий интерфейс (класс с методами fragment_loss и shock_clusters). Пока тарифа нет,
используются истинные интенсивности мира (TrueRates, как у оракула); агент TARIS подставит свои оценки.
Единицы среды: время — сутки (эпоха), интенсивности — в сутки.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .config import HOUR

CLUSTER_TYPES = ("dc", "provider", "region")


# ------------------------------------------------------------------ интенсивности
class TrueRates:
    """Истинные интенсивности мира (пункт 3.11): потеря фрагмента на узле и шоки кластеров, уносящие фрагменты.

    lambda_loss_i(t) = lambda_i(t) * phi_i + lambda_churn_i(t) + lambda_sil_i — простой дольше T_rep, уход, тихая потеря.
    eta_c = nu_c * P(L_c > T_rep) — шоки кластеров (согласованный мир). В несогласованном мире вместо шоков бури:
    они не уносят кластер целиком, а на время бури поднимают интенсивность простоев узлов, поэтому их средний
    вклад добавляется к lambda_loss каждого узла (приближение: связь узлов внутри бури не учитывается).
    Кризисы (пункт 3.9) сюда не входят: оракул кризисов действует по тревогам переносом.
    """

    def __init__(self, world):
        self.w = world
        c, nd = world.cfg, world.nodes
        self.phi = np.array([world.true_phi(i) for i in range(len(nd))])
        self.join = (nd["join_t_base"] if "join_t_base" in nd else nd["join_t"]).to_numpy()
        self.m_churn = nd["m_churn"].to_numpy()
        self.lam_sil = nd["lam_silent"].to_numpy()
        self.member = {ct: nd[ct].to_numpy() for ct in CLUSTER_TYPES}
        rates = np.array(c.shock_rate_per_year, float) / 365.0 if c.shocks_enabled else np.zeros(3)
        self.storm = c.world_mode != "matched"
        if self.storm:
            # средний вклад бурь: nu_c * (multiplier - 1) * длительность бури — множитель к lambda_i * phi_i
            self.storm_factor = float(rates.sum()) * (c.storm_multiplier - 1.0) * c.storm_hours * HOUR
            self.eta_type = np.zeros(3)
        else:
            self.storm_factor = 0.0
            k, d, T = c.shock_shape, c.franchise_d_hours, c.T_rep_hours
            # длительность шока — Вейбулл длиннее франшизы d; при полной гибели простой длиннее T_rep (world.shock_draw)
            sf = np.array([np.exp(-(T / th) ** k + (d / th) ** k) for th in c.shock_scale_hours])
            self.eta_type = rates * (c.shock_total_share + (1.0 - c.shock_total_share) * sf)
            # доля шоков с полной гибелью среди шоков, уносящих фрагменты (для задержки восстановления)
            p = c.shock_total_share + (1.0 - c.shock_total_share) * sf
            self.destroyed_share = float(np.average(c.shock_total_share / p, weights=np.maximum(rates, 1e-30)))

    def parts(self, nodes, t: float) -> dict:
        """Слагаемые интенсивности потери фрагмента на узлах nodes в момент t (в сутки)."""
        nodes = np.asarray(nodes, int)
        w = self.w
        lam = w.rate(nodes, np.full(len(nodes), float(t)))
        own = lam * self.phi[nodes] * (1.0 + self.storm_factor)
        age = np.maximum(float(t) - self.join[nodes], 0.0)
        c = w.cfg
        churn = c.churn_extra_new * np.exp(-age / c.churn_A_days) + c.churn_base * self.m_churn[nodes]
        return {"own": own, "churn": churn, "silent": self.lam_sil[nodes]}

    def fragment_loss(self, nodes, t: float) -> np.ndarray:
        """lambda_loss_i(t) — интенсивность потери фрагмента на каждом узле из nodes."""
        p = self.parts(nodes, t)
        return p["own"] + p["churn"] + p["silent"]

    def shock_clusters(self, nodes, t: float = 0.0) -> list[tuple[str, int, float, np.ndarray]]:
        """Кластеры, в которых лежат узлы nodes, с ненулевой интенсивностью шока: (тип, номер, eta_c, маска узлов).
        Дата-центра у устройства пользователя нет (dc = -1): такой «кластер» пропускается."""
        nodes = np.asarray(nodes, int)
        out = []
        for ti, ct in enumerate(CLUSTER_TYPES):
            if self.eta_type[ti] <= 0:
                continue
            ids = self.member[ct][nodes]
            for cl in np.unique(ids):
                if cl >= 0:
                    out.append((ct, int(cl), float(self.eta_type[ti]), ids == cl))
        return out


# ------------------------------------------------------------------ вероятность потери за окно
def poisson_binomial(q) -> np.ndarray:
    """Распределение числа погибших фрагментов: G[z] = P(D = z) при независимых гибелях с вероятностями q
    (рекурсия пункта 2.5, O(n^2))."""
    G = np.zeros(len(q) + 1)
    G[0] = 1.0
    for m, qm in enumerate(np.asarray(q, float), start=1):
        G[1:m + 1] = G[1:m + 1] * (1.0 - qm) + G[0:m] * qm
        G[0] *= 1.0 - qm
    return G


def p_loss_window(q, k: int) -> float:
    """P_loss(S) без шоков: погибло больше n - k фрагментов из n = len(q)."""
    n = len(q)
    return float(poisson_binomial(q)[n - k + 1:].sum())


def p_loss_window_shocks(q, k: int, masks, h, max_shocks: int = 2) -> tuple[float, float]:
    """P_loss(S) с шоками (пункт 2.5): полная вероятность по наборам кластеров под шоком, не больше max_shocks
    одновременно. masks — маски узлов набора по кластерам, h — вероятности шока кластера за окно.
    Возвращает (вероятность, оценку сверху отброшенной части = P(больше max_shocks шоков))."""
    q = np.asarray(q, float)
    n, h = len(q), np.asarray(h, float)
    masks = [np.asarray(m, bool) for m in masks]
    total = 0.0
    for r in range(max_shocks + 1):
        for H in combinations(range(len(h)), r):
            sel = np.zeros(len(h), bool)
            sel[list(H)] = True
            ph = float(np.prod(h[sel]) * np.prod(1.0 - h[~sel]))
            hit = np.zeros(n, bool)
            for c in H:
                hit |= masks[c]
            need = n - k + 1 - int(hit.sum())            # сколько ещё гибелей среди остальных узлов губит объект
            if need <= 0:
                total += ph
            else:
                G = poisson_binomial(q[~hit])
                total += ph * float(G[need:].sum())
    dropped = float(poisson_binomial(h)[max_shocks + 1:].sum())
    return total, dropped


def window_probs(rates, nodes, t: float, tau: float):
    """Вероятности для окна tau (сутки): q_i = 1 - exp(-lambda_loss_i * tau), маски кластеров и h_c."""
    q = 1.0 - np.exp(-rates.fragment_loss(nodes, t) * tau)
    cl = rates.shock_clusters(nodes, t)
    masks = [m for *_, m in cl]
    h = np.array([1.0 - np.exp(-eta * tau) for _, _, eta, _ in cl])
    return q, masks, h


def window_tau(cfg) -> float:
    """Окно уязвимости tau = T_rep + длительность восстановления, сутки."""
    return cfg.T_rep + cfg.repair_hours * HOUR


# ------------------------------------------------------------------ модель объекта и уравнение Тиле
class Costs:
    """Параметры уравнения Тиле (пункт 3.11): delta — сила процента в эпоху, v — хранение фрагмента за эпоху,
    c_tr — стоимость единицы трафика, W — ценность объекта, size — объём объекта."""

    def __init__(self, delta=np.log(1.05) / 365.0, v=1.0 / 365.0, c_tr=0.1, W=1e4, size=1.0):
        self.delta, self.v, self.c_tr, self.W, self.size = float(delta), float(v), float(c_tr), float(W), float(size)

    @classmethod
    def loss_probability(cls, size=1.0):
        """delta = v = c_tr = 0, W = 1: V_n(0) — вероятность потерять объект до горизонта."""
        return cls(0.0, 0.0, 0.0, 1.0, size)

    @classmethod
    def repair_traffic(cls, size=1.0):
        """delta = v = 0, W = 0, c_tr = 1: V_n(0) — ожидаемый трафик восстановления за срок."""
        return cls(0.0, 0.0, 1.0, 0.0, size)


def repair_mu(cfg) -> float:
    """Консервативная интенсивность завершения восстановления: задержка T_rep + t_rep (в сутки)."""
    return 1.0 / window_tau(cfg)


def path_delays(cfg) -> dict:
    """Задержка от исчезновения фрагмента до конца его восстановления по путям потери (учёт, пункт 3.6), сутки.
    Политика решает раз в эпоху (в среднем полэпохи ожидания), операция длится t_rep.
    - простой дольше T_rep без гибели: фрагмент исчезает в момент признания (T_rep от начала простоя);
    - уход узла и шок с полной гибелью: исчезает сразу, признан через T_rep;
    - тихая потеря: исчезает сразу, признана при следующем успешном аудите (в среднем полинтервала)."""
    base = 0.5 + cfg.repair_hours * HOUR
    aud = 1.0 / max(cfg.audits_per_epoch, 1)
    return {"own": base, "churn": cfg.T_rep + base, "silent": 0.5 * aud + base,
            "shock": base, "shock_destroyed": cfg.T_rep + base}


def object_mu(rates, nodes, t: float, shocks=()) -> float:
    """Интенсивность завершения восстановления объекта: обратная величина средней задержки, взвешенной
    по интенсивностям путей потери его фрагментов (пункт 3.11)."""
    c = rates.w.cfg
    D = path_delays(c)
    p = rates.parts(nodes, t)
    w_own, w_ch, w_sil = p["own"].sum(), p["churn"].sum(), p["silent"].sum()
    num = w_own * D["own"] + w_ch * D["churn"] + w_sil * D["silent"]
    den = w_own + w_ch + w_sil
    ds = getattr(rates, "destroyed_share", 0.0)
    for eta, mj in shocks:
        num += eta * mj * ((1 - ds) * D["shock"] + ds * D["shock_destroyed"])
        den += eta * mj
    return den / num if num > 0 else 1.0 / D["own"]


REPAIR_PHASES = 3      # фаз Эрланга у задержки восстановления (пункт 3.11; выбор — check_reserve.py, раздел 4)


def chain_states(n: int, k: int, R_min: int, phases: int) -> list[tuple[int, int]]:
    """Состояния цепи объекта: (0, -1) — «потерян»; (s, 0) — s фрагментов, восстановления нет;
    (s, j), j = 1..phases — s фрагментов, идёт восстановление, фаза j задержки. Последнее состояние — (n, 0)."""
    st = [(0, -1)]
    for s in range(k, n + 1):
        if s < n and s - k <= R_min:
            st += [(s, j) for j in range(1, phases + 1)]
        else:
            st.append((s, 0))
    return st


def generator(n: int, k: int, lam: float, shocks, R_min: int, mu: float, costs: Costs, phases: int = REPAIR_PHASES):
    """Генератор цепи объекта и платежи на одну эпоху с постоянными интенсивностями.

    lam — средняя интенсивность потери одного фрагмента (Lambda_s = s * lam); shocks — [(eta_c, m_jc)]: шок кластера
    уносит m_jc фрагментов. Восстановление начинается, как только запас s - k опускается до R_min, и через
    задержку (Эрланг с phases фазами и средним 1 / mu) возвращает объект в n фрагментов; потери во время
    восстановления его не прерывают. phases = 1 — экспоненциальная задержка.
    Возвращает (Q, b): Q — интенсивности переходов (строка — откуда), b — интенсивность платежей в состоянии:
    хранение v * s плюс ожидаемые платежи при переходах (W при потере, g(r) при завершении восстановления).
    """
    st = chain_states(n, k, R_min, phases)
    ix = {x: a for a, x in enumerate(st)}
    m = len(st)
    Q = np.zeros((m, m))
    b = np.zeros(m)
    c = costs

    def target(s2, j):
        if s2 < k:
            return 0
        if j >= 1:
            return ix[(s2, j)]
        return ix[(s2, 1)] if (s2, 1) in ix else ix[(s2, 0)]

    for a, (s, j) in enumerate(st):
        if j < 0:
            continue
        b[a] += c.v * s
        moves = [(s * lam, s - 1)] + [(eta, s - mj) for eta, mj in shocks if eta > 0 and mj > 0]
        for r, s2 in moves:
            to = target(s2, j)
            Q[a, to] += r
            if to == 0:
                b[a] += r * c.W
        if j >= 1 and mu > 0:                       # фаза задержки восстановления
            r = phases * mu
            if j < phases:
                Q[a, ix[(s, j + 1)]] += r
            else:
                Q[a, ix[(n, 0)]] += r
                b[a] += r * c.c_tr * (k + n - s) * c.size / k
    np.fill_diagonal(Q, 0.0)
    Q[np.diag_indices(m)] = -Q.sum(1)
    return Q, b


def thiele(n: int, k: int, lam, shocks=(), R_min: int = 0, mu: float = 1.0, costs: Costs | None = None,
           dt: float = 1.0, phases: int = REPAIR_PHASES) -> np.ndarray:
    """Решение уравнения Тиле назад по эпохам (пункт 3.11).

    dV/dt = delta * V - b(t) - Q(t) V, V(H) = 0. lam — интенсивность потери фрагмента по эпохам (массив длины E,
    горизонт H = E * dt; при постоянных интенсивностях — thiele_const); shocks — [(eta_c, m_jc)], где eta_c —
    число или массив по эпохам.
    Внутри эпохи коэффициенты постоянны, и шаг точный: экспонента расширенной матрицы [[-A, b], [0, 0]],
    A = delta * I - Q. Возвращает V размера (E + 1, число состояний chain_states): V[e, состояние] в начале
    эпохи e, V[E] = 0; последний столбец — состояние (n, 0), первый — «потерян».
    """
    from scipy.linalg import expm
    costs = costs or Costs()
    lam = np.atleast_1d(np.asarray(lam, float))
    E = len(lam)
    m = len(chain_states(n, k, R_min, phases))
    V = np.zeros((E + 1, m))
    cache = {}
    for e in range(E - 1, -1, -1):
        sh = tuple((float(np.atleast_1d(eta)[min(e, np.size(eta) - 1)]), int(mj)) for eta, mj in shocks)
        key = (float(lam[e]), sh)
        if key not in cache:
            Q, b = generator(n, k, lam[e], sh, R_min, mu, costs, phases)
            A = costs.delta * np.eye(m) - Q
            M = np.zeros((m + 1, m + 1))
            M[:m, :m] = -A
            M[:m, m] = b
            X = expm(M * dt)
            cache[key] = (X[:m, :m], X[:m, m])
        P, f = cache[key]
        V[e] = P @ V[e + 1] + f
    return V


def thiele_const(n, k, lam: float, H: float, shocks=(), R_min=0, mu=1.0, costs=None, dt=1.0,
                 phases: int = REPAIR_PHASES) -> np.ndarray:
    """То же при постоянных интенсивностях на горизонте H (сутки)."""
    E = int(round(H / dt))
    return thiele(n, k, np.full(E, float(lam)), [(float(eta), mj) for eta, mj in shocks], R_min, mu, costs, dt, phases)


def object_rates(rates, nodes, t0: float, H: int):
    """Интенсивности объекта на узлах nodes на горизонте H эпох от t0: средняя интенсивность потери фрагмента
    по эпохам (в середине эпохи; при восстановлении новые узлы — как в среднем текущие) и шоки кластеров
    [(eta_c, m_jc)]."""
    nodes = np.asarray(nodes, int)
    lam = np.array([rates.fragment_loss(nodes, t0 + e + 0.5).mean() for e in range(H)])
    shocks = [(eta, int(mask.sum())) for _, _, eta, mask in rates.shock_clusters(nodes, t0)]
    return lam, shocks


def object_metrics(n, k, lam, shocks=(), R_min=0, mu=1.0, costs=None, phases: int = REPAIR_PHASES) -> dict:
    """Три величины таблицы пункта 2.5 из состояния n в начале горизонта: вероятность потери, ожидаемый трафик
    восстановления (в объёмах объекта) и ожидаемая приведённая стоимость."""
    costs = costs or Costs()
    out = {}
    for name, cst in (("P_loss", Costs.loss_probability(costs.size)), ("traffic", Costs.repair_traffic(costs.size)),
                      ("cost", costs)):
        out[name] = float(thiele(n, k, lam, shocks, R_min, mu, cst, phases=phases)[0, -1])
    return out
