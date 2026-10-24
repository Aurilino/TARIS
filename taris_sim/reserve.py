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
