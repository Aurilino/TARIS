"""Граф связей узлов (шаг 2, пункт 2.4) — первая часть модели TARIS.

Считается только по «взгляду сети» и только по данным до момента расчёта t:
- наблюдаемые простои длиннее франшизы d (страховые случаи) — по пингам;
- реестр узлов: регистрация, кошелёк выплат, подсеть /24.

Инфраструктурная связь — корреляция индикаторов страхового случая по общим эпохам наблюдения пары,
сжатая к нулю по кредибилити: rho~ = Z * rho^, Z = n / (n + K), где n — эпохи, в которых случай был
хотя бы у одного узла пары. Отрицательная корреляция — нулевая связь.
Связь по владению — noisy-OR сигналов: o = 1 - prod(1 - b_m * f_m).
Итог для конфиденциальности — max(rho~, o); подозрительность узла — его самая сильная связь.
Кластеры — иерархическая кластеризация (средняя связь) по расстоянию 1 - rho_conf; детерминирована.
Хранится только разреженная часть графа (пары не слабее graph_store_min).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass
class Snapshot:
    t: float                  # момент расчёта, сутки
    pairs: pd.DataFrame       # i < l: rho_hat, n_inf, Z, rho, own, conf — только пары не слабее порога
    suspicion: np.ndarray     # подозрительность узла (самая сильная связь); NaN — узел ещё не зарегистрирован
    clusters: np.ndarray      # номер найденного кластера (-1 — не зарегистрирован или одиночка)
    clusters_infra: np.ndarray  # то же только по инфраструктурной связи (уровень 2 из 2.4: общий источник отказов)


class LinkGraph:
    def __init__(self, world, chain):
        c = world.cfg
        self.cfg = c
        nd = world.nodes
        self.N, self.E = len(nd), int(c.horizon_days)
        cancelled = nd["cancelled"].to_numpy() if "cancelled" in nd else np.zeros(self.N, bool)
        # реестр (наблюдаемо): регистрация, кошелёк, подсеть
        self.join = np.where(cancelled, np.inf, nd["join_t"].to_numpy())
        self.wallet = nd["wallet"].to_numpy()
        self.subnet = pd.factorize(nd["subnet"])[0]
        # страховые случаи по эпохам и эпохи наблюдения: узел зарегистрирован и не простаивает всю эпоху
        ob = chain.observed[chain.observed["dur"] > c.d]
        Y = np.zeros((self.N, self.E), np.float32)
        e0 = np.floor(ob["start"].to_numpy()).astype(int)
        ok = e0 < self.E
        Y[ob["node"].to_numpy()[ok], e0[ok]] = 1.0
        covered = np.zeros((self.N, self.E), bool)
        for i, s, e in zip(ob["node"].to_numpy(), ob["start"].to_numpy(), ob["end"].to_numpy()):
            a, b = int(np.ceil(s)), int(min(np.floor(e), self.E))       # эпохи [a, b) целиком внутри простоя
            if b > a:
                covered[i, a:b] = True
        reg = np.arange(self.E)[None, :] >= np.ceil(self.join)[:, None]
        self.M = (reg & ~covered).astype(np.float32)
        self.Y = Y * self.M

    # ---------------------------------------------------------------- связи
    def infrastructure(self, t: float):
        """Инфраструктурная связь по эпохам [0, floor(t)): rho_hat, n_inf, Z и rho~ для всех пар (матрицы N x N)."""
        T = int(np.floor(t))
        Y, M = self.Y[:, :T], self.M[:, :T]
        n = M @ M.T                                   # общие эпохи наблюдения
        A = Y @ M.T                                   # A[i, l] — случаи узла i в общих эпохах пары
        C = Y @ Y.T                                   # совместные случаи
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = A / n, A.T / n
            cov = C / n - a * b
            rho_hat = np.where((n > 1) & (a > 0) & (b > 0) & (a < 1) & (b < 1),
                               cov / np.sqrt(a * (1 - a) * b * (1 - b)), 0.0)
        n_inf = A + A.T - C                           # эпохи, где случай был хотя бы у одного узла пары
        Z = n_inf / (n_inf + self.cfg.graph_K_rho)
        rho = np.clip(Z * rho_hat, 0.0, 1.0)
        np.fill_diagonal(rho, 0.0)
        return rho_hat, n_inf, Z, rho

    def ownership(self, t: float) -> np.ndarray:
        """Связь по владению (noisy-OR) для узлов, зарегистрированных к моменту t."""
        c = self.cfg
        reg = self.join <= t
        f_wallet = (self.wallet[:, None] == self.wallet[None, :])
        f_subnet = (self.subnet[:, None] == self.subnet[None, :])
        j = np.where(reg, self.join, np.nan)
        f_join = np.abs(j[:, None] - j[None, :]) < c.graph_join_window_days
        o = 1.0 - (1.0 - c.graph_b_wallet * f_wallet) * (1.0 - c.graph_b_subnet * f_subnet) * (1.0 - c.graph_b_join * f_join)
        o = o * (reg[:, None] & reg[None, :])
        np.fill_diagonal(o, 0.0)
        return o

    def snapshot(self, t: float) -> Snapshot:
        c = self.cfg
        rho_hat, n_inf, Z, rho = self.infrastructure(t)
        own = self.ownership(t)
        conf = np.maximum(rho, own)
        reg = self.join <= t
        iu, lu = np.triu_indices(self.N, 1)
        keep = conf[iu, lu] >= c.graph_store_min
        i, l = iu[keep], lu[keep]
        pairs = pd.DataFrame({"i": i, "l": l, "rho_hat": rho_hat[i, l], "n_inf": n_inf[i, l], "Z": Z[i, l],
                              "rho": rho[i, l], "own": own[i, l], "conf": conf[i, l]})
        susp = np.where(reg, conf.max(axis=1), np.nan)
        return Snapshot(t=t, pairs=pairs, suspicion=susp, clusters=self.clusters(conf, reg),
                        clusters_infra=self.clusters(rho, reg))

    def clusters(self, conf: np.ndarray, reg: np.ndarray) -> np.ndarray:
        """Иерархическая кластеризация зарегистрированных узлов (средняя связь, расстояние 1 - conf),
        разрез на расстоянии 1 - graph_cluster_cut. Кластеры из одного узла помечаются -1."""
        idx = np.flatnonzero(reg)
        lab = np.full(self.N, -1)
        if len(idx) < 2:
            return lab
        D = 1.0 - conf[np.ix_(idx, idx)]
        np.fill_diagonal(D, 0.0)
        Zl = linkage(squareform(D, checks=False), method="average")
        f = fcluster(Zl, t=1.0 - self.cfg.graph_cluster_cut, criterion="distance")
        sizes = np.bincount(f)
        f = np.where(sizes[f] > 1, f, -1)
        # номера кластеров — по первому узлу кластера (детерминированно и не зависит от нумерации scipy)
        first = {}
        for k, v in zip(idx, f):
            if v > 0 and v not in first:
                first[v] = k
        lab[idx] = [first[v] if v > 0 else -1 for v in f]
        return lab

    def snapshots(self, step_days: int | None = None) -> list[Snapshot]:
        step = step_days or self.cfg.graph_snapshot_days
        return [self.snapshot(float(t)) for t in range(step, self.E + 1, step)]
