"""Граф заражения узлов агентом TARIS (шаг 5, подшаг С1): связь отказов с запаздыванием.

Граф связей пункта 2.4 (graph.py) ищет одновременные отказы. Здесь — направленная связь j -> i: после начала
простоя узла j простой узла i начинается чаще, чем по его фону. Считается только по данным сети: начала
наблюдаемых простоев длиннее франшизы d (страховые случаи) из view.step и моменты регистрации узлов.

Для пары (j -> i) и окна tau:
    O_ij — число пар событий (t_j, t_i) с 0 < t_i - t_j <= tau;
    E_ij — ожидаемое при независимости: сумма по событиям узла j фоновой частоты узла i, умноженной на время
           наблюдения узла i в окне (t_j, t_j + tau];
    Z_ij = E_ij / (E_ij + K) — кредибилити пары;
    w_ij = Z_ij * max(0, (O_ij - E_ij) / E_ij).
Фон узла — его частота событий на время в доступе (в 30-суточном блоке, сглаженная к его частоте за всё
время, а та — к средней по сети на c0 суток). Пары внутри одного заявленного
дата-центра считаются отдельно (буря дата-центра и так видна прогнозу): в граф заражения W они не входят.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HOUR
from .experience import Registry


@dataclass
class ContagionSnapshot:
    t: float
    tau: float
    pairs: pd.DataFrame          # j, i, O, E, Z, w, same_dc — только пары с O > 0
    W: np.ndarray                # N x N: W[i, j] = w_ij (влияние j на i), без пар внутри дата-центра
    rate: np.ndarray             # фоновая частота событий узла, в сутки
    O: np.ndarray                # N x N: O[i, j]
    E: np.ndarray                # N x N: E[i, j]
    same_dc: np.ndarray          # N x N: пара внутри одного заявленного дата-центра


class ContagionGraph:
    """Накопление страховых случаев по view.step и расчёт связей с запаздыванием.

    Экспозиция узла — время в доступе по пингам (зарегистрирован и не в наблюдаемом простое), по часам:
    ушедший узел (простой без конца) экспозиции не даёт. Фон узла i в окне после события узла j — его частота
    в 30-суточном блоке события, сглаженная к его частоте за всё время (а та — к средней по сети)."""

    SLOT = 24                                   # ячеек экспозиции в сутках (по часу)

    def __init__(self, world, tau: float = 1.0, K: float = 1.0, c0: float = 30.0, c1: float = 30.0,
                 block: int = 30):
        c = world.cfg
        self.reg = Registry(world)
        self.N, self.T = self.reg.N, int(c.horizon_days)
        self.d = c.franchise_d_hours * HOUR
        self.tau, self.K, self.c0, self.c1, self.block = float(tau), float(K), float(c0), float(c1), int(block)
        self.ev_node, self.ev_t, self.ev_seen = [], [], []  # узел, начало простоя, момент когда длительность стала известна
        self.seen = set()                         # (узел, начало) уже записанные
        self.up = np.zeros((self.N, self.T * self.SLOT), np.float32)   # доля часа в доступе
        self.join = np.full(self.N, np.inf)
        self.t = 0.0

    def update(self, t: float, view):
        """Забирает из view.step простои за [t0, t1): события длиннее d и часы в доступе."""
        st = view.step
        self.join = view.join_times()
        if st is None:
            return
        t0, t1 = (st["t0"], st["t1"]) if "t0" in st else (float(st["epoch"]), float(st["epoch"]) + 1.0)
        self.t = max(self.t, float(t1))
        cand = []
        e = st["outages_ended"]
        if len(e):
            m = e["dur"].to_numpy() > self.d
            cand += list(zip(e["node"].to_numpy()[m].tolist(), e["start"].to_numpy()[m].tolist()))
        og = st["outages_ongoing"]
        if len(og):
            m = (t1 - og["start"].to_numpy()) > self.d
            cand += list(zip(og["node"].to_numpy()[m].tolist(), og["start"].to_numpy()[m].tolist()))
        for n, s in cand:
            if (n, s) not in self.seen:
                self.seen.add((n, s))
                self.ev_node.append(int(n)); self.ev_t.append(float(s)); self.ev_seen.append(float(t1))
        # экспозиция по часам отрезка: зарегистрирован, минус наблюдаемые простои
        S = self.SLOT
        k0, k1 = int(round(t0 * S)), min(int(round(t1 * S)), self.T * S)
        if k1 <= k0:
            return
        g = (np.arange(k0, k1 + 1) / S)                               # границы часов
        lo, hi = g[:-1], g[1:]
        reg = np.clip((hi[None, :] - np.maximum(self.join[:, None], lo[None, :])) * S, 0.0, 1.0)
        down = np.zeros_like(reg)
        iv = []
        if len(e):
            iv += list(zip(e["node"].to_numpy(), e["start"].to_numpy(), e["end"].to_numpy()))
        if len(og):
            iv += [(n, s_, t1) for n, s_ in zip(og["node"].to_numpy(), og["start"].to_numpy())]
        for n, s_, e_ in iv:
            a_, b_ = max(s_, t0), min(e_, t1)
            if b_ > a_:
                down[int(n)] += np.clip((np.minimum(hi, b_) - np.maximum(lo, a_)) * S, 0.0, 1.0)
        self.up[:, k0:k1] = np.clip(reg - down, 0.0, 1.0)

    def _cum(self, k_end):
        C = np.zeros((self.N, k_end + 1))
        np.cumsum(self.up[:, :k_end], axis=1, out=C[:, 1:])
        return C / self.SLOT                                           # сутки в доступе

    def snapshot(self, t: float | None = None) -> ContagionSnapshot:
        """Связи по событиям, начавшимся до t (по умолчанию — до последних данных)."""
        t = self.t if t is None else float(t)
        S = self.SLOT
        node, ts, seen = (np.array(self.ev_node, int), np.array(self.ev_t, float),
                          np.array(self.ev_seen, float))
        # Исторический снимок видит только события, распознанные к моменту t.
        # Начало простоя может быть раньше: долгий простой подтверждается с задержкой.
        keep = (ts < t) & (seen <= t)
        node, ts = node[keep], ts[keep]
        o = np.argsort(ts, kind="stable")
        node, ts = node[o], ts[o]
        N, tau = self.N, self.tau
        kt = min(int(np.floor(t * S + 1e-9)), self.T * S)
        C = self._cum(kt)

        def up_between(a, b):                                          # сутки в доступе на (a, b]
            def at(x):
                x = min(max(x, 0.0), kt / S)
                k = min(int(np.floor(x * S)), kt)
                fr = x * S - k
                return C[:, k] + (fr * self.up[:, k] / S if k < kt else 0.0)
            return at(b) - at(a)

        # фон: за всё время и по блокам
        cnt = np.bincount(node, minlength=N).astype(float)
        expo = C[:, -1]
        m = cnt.sum() / max(expo.sum(), 1e-9)
        rate = (cnt + m * self.c0) / (expo + self.c0)
        nb = int(np.ceil(t / self.block)) or 1
        blk = np.minimum((ts / self.block).astype(int), nb - 1)
        cnt_b = np.zeros((N, nb)); np.add.at(cnt_b, (node, blk), 1.0)
        edges = [min(int(b * self.block * S), kt) for b in range(nb + 1)]
        up_b = np.stack([C[:, edges[b + 1]] - C[:, edges[b]] for b in range(nb)], 1)
        rate_b = (cnt_b + rate[:, None] * self.c1) / (up_b + self.c1)
        # O: пары событий j -> i в окне (0, tau]
        O = np.zeros((N, N))
        hi = np.searchsorted(ts, ts + tau, side="right")
        for a in range(len(ts)):
            b0, b1 = a + 1, hi[a]
            if b1 > b0:
                later = node[b0:b1]
                later = later[ts[b0:b1] > ts[a]]
                np.add.at(O[:, node[a]], later, 1.0)
        # E: по каждому событию узла j — фон узла i в блоке события на его время в доступе в (t_j, t_j + tau]
        E = np.zeros((N, N))
        for a in range(len(ts)):
            E[:, node[a]] += rate_b[:, blk[a]] * up_between(ts[a], min(ts[a] + tau, t))
        np.fill_diagonal(O, 0.0); np.fill_diagonal(E, 0.0)
        Z = E / (E + self.K)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(E > 0, Z * np.maximum(0.0, (O - E) / E), 0.0)
        dc = self.reg.dc
        same = (dc[:, None] == dc[None, :]) & (dc[:, None] >= 0)
        W = np.where(same, 0.0, w)
        ii, jj = np.nonzero(O > 0)
        pairs = pd.DataFrame({"j": jj, "i": ii, "O": O[ii, jj], "E": E[ii, jj], "Z": Z[ii, jj],
                              "w": w[ii, jj], "same_dc": same[ii, jj]})
        return ContagionSnapshot(t, tau, pairs, W, rate, O, E, same)
