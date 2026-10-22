"""Интерфейс политики и две граничные политики (шаг 3; шаг 4, пункт 4.1).

UniformPolicy — равномерное размещение со статическим кодом (нижняя граница).
OraclePolicy — знает истинный риск узлов и скрытые кластеры (верхняя граница).
Политика видит мир только через NetworkView; оракул дополнительно получает объект World.
"""
from __future__ import annotations

import numpy as np


class Policy:
    name = "base"

    def place(self, obj: int, t: float, view) -> list[int]:
        """Узлы для фрагментов нового объекта (их число = n_j)."""
        raise NotImplementedError

    def on_epoch(self, t: float, view) -> list[tuple[int, list[int], list[int]]]:
        """Решения о восстановлении: (объект, номера фрагментов, узлы-получатели)."""
        return []

    def migrate(self, t: float, view) -> list[tuple[int, int, int]]:
        """Переносы заранее (пункт 3.9): (объект, номер фрагмента, узел-получатель)."""
        return []

    def alarm(self, t: float, view) -> list[dict]:
        """Тревоги о кризисах (пункт 3.9): словари с ключами type (тип кризиса), ctype и cluster
        (кластер-цель в тех же обозначениях, что world.crises: dc — номер, region — номер,
        batch — «dc/класс», version — номер версии), prob (вероятность) и horizon (горизонт, сутки)."""
        return []

    def predict(self, t: float, view):
        return None

    def explain(self, obj: int, t: float, view):
        return None

    @staticmethod
    def _rng(view):
        return np.random.default_rng(int.from_bytes(view.seed[:8], "big"))


class UniformPolicy(Policy):
    """Равномерный выбор среди доступных узлов, статический код (n, k), немедленное восстановление."""

    def __init__(self, n: int = 15):
        self.n = n
        self.name = f"Равномерное, n={n}"

    @staticmethod
    def _choose(rng, cand, m, exclude=()):
        """m различных узлов из cand, исключая exclude (быстрый выбор с отбраковкой)."""
        out, seen = [], set(exclude)
        if len(cand) - len(seen) < m:
            return []
        while len(out) < m:
            for j in rng.integers(0, len(cand), 2 * (m - len(out)) + 2):
                c = int(cand[j])
                if c not in seen:
                    seen.add(c); out.append(c)
                    if len(out) == m:
                        break
        return out

    def place(self, obj, t, view):
        return self._choose(self._rng(view) if not hasattr(self, "_g") else self._g, view.candidates(), self.n)

    def on_epoch(self, t, view):
        self._g = self._rng(view)
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose(self._g, cand, len(idx), exclude=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class OraclePolicy(UniformPolicy):
    """Знает истинную интенсивность полной гибели узлов и скрытые кластеры.
    Вес узла обратно пропорционален истинному риску; в одном дата-центре — не больше одного фрагмента объекта,
    у одного провайдера, в одном регионе и у одного владельца — не больше min(n-k, k-1)."""

    def __init__(self, world, n: int = 15, gamma: float = 1.0):
        super().__init__(n)
        self.name = f"Оракул, n={n}"
        nd = world.nodes
        phi = np.array([world.true_phi(i) for i in range(len(nd))])
        q = nd["lam_avg"].to_numpy() * phi
        self.w = (1.0 / np.maximum(q, 1e-9)) ** gamma
        self.dc, self.prov, self.reg, self.own = (nd[c].to_numpy() for c in ("dc", "provider", "region", "owner"))
        self.k = world.cfg.k
        self.n_dc, self.n_prov, self.n_reg = world.cfg.n_dc, world.cfg.n_providers, world.cfg.n_regions

    def _choose_spread(self, rng, cand, m, held=()):
        r = min(self.n - self.k, self.k - 1)
        held = np.fromiter(held, int) if len(held) else np.empty(0, int)
        used_dc = np.zeros(self.n_dc, bool); used_dc[self.dc[held]] = True
        cp = np.bincount(self.prov[held], minlength=self.n_prov)
        cr = np.bincount(self.reg[held], minlength=self.n_reg)
        co = {}
        for h in held:
            co[self.own[h]] = co.get(self.own[h], 0) + 1
        cand = cand[~np.isin(cand, held)] if len(held) else cand
        w = self.w[cand].astype(float)
        out = []
        for _ in range(m):
            ok = (~used_dc[self.dc[cand]]) & (cp[self.prov[cand]] < r) & (cr[self.reg[cand]] < r)
            if co:
                ok &= np.array([co.get(o, 0) < r for o in self.own[cand]]) if len(co) else ok
            ww = w * ok
            tot = ww.sum()
            if tot <= 0:
                return []
            j = int(np.searchsorted(np.cumsum(ww), rng.random() * tot, side="right"))
            j = min(j, len(cand) - 1)
            c = int(cand[j]); out.append(c)
            used_dc[self.dc[c]] = True; cp[self.prov[c]] += 1; cr[self.reg[c]] += 1
            co[self.own[c]] = co.get(self.own[c], 0) + 1
            w[j] = 0.0
        return out

    def place(self, obj, t, view):
        return self._choose_spread(self._g if hasattr(self, "_g") else self._rng(view), view.candidates(), self.n)

    def on_epoch(self, t, view):
        self._g = self._rng(view)
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose_spread(self._g, cand, len(idx), held=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs
