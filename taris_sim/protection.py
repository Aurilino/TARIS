"""Опциональная защита размещения от коалиций; не меняет стандартный ReservePolicy."""
from __future__ import annotations
import numpy as np
from .policy import ReservePolicy
from .graph import LinkGraph


class ProtectedReservePolicy(ReservePolicy):
    """Графовый лимит, лимит новичков и штраф за совместный выбор связанных узлов.

    Ограничения относятся к целевому набору текущей операции. Они не отзывают
    прежние копии у противника и не исправляют автоматически уже возникшие
    нарушения после изменения графа. При недопустимом наборе операция отклоняется.
    """
    def __init__(self, world, n=13, *, graph_cap=True, novice_cap=True,
                 link_penalty=1., novice_threshold=.1, snapshot_days=30, **kwargs):
        if not np.isfinite(link_penalty) or link_penalty < 0:
            raise ValueError("link_penalty должен быть конечным и >= 0")
        if not np.isfinite(novice_threshold) or not 0 <= novice_threshold <= 1:
            raise ValueError("novice_threshold должен быть в [0, 1]")
        if type(snapshot_days) is not int or snapshot_days < 1:
            raise ValueError("snapshot_days должен быть положительным целым")
        kwargs.setdefault("rates", "agent")
        kwargs.setdefault("placement", "agent_greedy")
        kwargs.setdefault("mode", "fixed")
        kwargs.setdefault("R_fixed", 2)
        if kwargs["rates"] != "agent" or kwargs["placement"] != "agent_greedy":
            raise ValueError("Защищённый режим поддерживает только agent_greedy с rates='agent'")
        super().__init__(world, n=n, **kwargs)
        self.graph_cap = bool(graph_cap)
        self.novice_cap = bool(novice_cap)
        self.link_penalty = float(link_penalty)
        self.novice_threshold = float(novice_threshold)
        self.snapshot_days = snapshot_days
        self._link_graph = None
        self._guard_snapshot = None
        self._guard_conf = None
        self._guard_labels = None
        self._guard_t = None
        self.guard_log = []
        self.guard_stats = {"attempts": 0, "rejected_attempts": 0, "snapshots": 0}
        self.name = f"TARIS: защита коалиций, n={n}"

    @property
    def guard_enabled(self):
        return self.graph_cap or self.novice_cap or self.link_penalty > 0

    def on_epoch(self, t, view):
        self._guard_t = float(t)
        if self.guard_enabled and (self.graph_cap or self.link_penalty):
            if self._link_graph is None:
                self._link_graph = LinkGraph(self.world, view.L.ch)
            if self._guard_snapshot is None or t - self._guard_snapshot.t >= self.snapshot_days:
                snapshot = self._link_graph.snapshot(float(t))
                self._guard_snapshot = snapshot
                self._guard_labels = snapshot.clusters.copy()
                # Только сеть: общие наблюдённые отказы и публичные сигналы владения.
                self._guard_conf = np.zeros((len(self.w), len(self.w)), dtype=float)
                for row in snapshot.pairs.itertuples(index=False):
                    self._guard_conf[row.i, row.l] = self._guard_conf[row.l, row.i] = row.conf
                self.guard_stats["snapshots"] += 1
        return super().on_epoch(t, view)

    def _novices(self):
        if not self.model.fitted:
            return np.ones(len(self.w), dtype=bool)
        return np.asarray(self.model.Z) < self.novice_threshold

    def _choose_spread(self, rng, cand, m, held=()):
        # Глобально невозможный набор не становится возможным от ослабления
        # инфраструктурных лимитов; не повторяем полный поиск для каждого шага.
        if self.guard_enabled:
            held = np.asarray(list(held), dtype=int)
            cand = np.asarray(cand, dtype=int)
            cand = cand[(self.w[cand] > 0) & ~np.isin(cand, held)]
            possible = len(cand)
            if self.novice_cap:
                novice = self._novices()
                allowance = self.k - 1 - int(novice[held].sum())
                possible = min(possible, int((~novice[cand]).sum()) + min(int(novice[cand].sum()), max(allowance, 0)))
                if allowance < 0:
                    possible = -1
            if self.graph_cap and self._guard_labels is not None:
                labels = self._guard_labels
                cap = min(self.n-self.k, self.k-1)
                capacity = int((labels[cand] < 0).sum())
                for label in np.unique(labels[cand][labels[cand] >= 0]):
                    capacity += min(int((labels[cand] == label).sum()),
                                    max(0, cap - int((labels[held] == label).sum())))
                if any(int((labels[held] == label).sum()) > cap for label in np.unique(labels[held][labels[held] >= 0])):
                    capacity = -1
                possible = min(possible, capacity)
            if possible < m:
                self.guard_stats["attempts"] += 1
                self.guard_stats["rejected_attempts"] += 1
                return []
        return super()._choose_spread(rng, cand, m, held)

    def _choose_spread_r(self, rng, cand, m, held, r_prov, r_reg, r_own, dc_cap=1, greedy=False):
        if not self.guard_enabled:
            return super()._choose_spread_r(rng, cand, m, held, r_prov, r_reg, r_own, dc_cap, greedy)
        if self._guard_t is None:
            raise RuntimeError("Перед размещением требуется on_epoch с наблюдениями сети")
        self.guard_stats["attempts"] += 1
        held = np.asarray(list(held), dtype=int)
        cand = np.asarray(cand, dtype=int)
        cand = cand[~np.isin(cand, held)]
        selected = list(map(int, held))
        novice = self._novices() if self.novice_cap else np.zeros(len(self.w), bool)
        labels = self._guard_labels if self.graph_cap else None
        cap = min(self.n - self.k, self.k - 1)

        def admissible(nodes):
            if self.novice_cap and int(novice[nodes].sum()) > self.k - 1:
                return False
            if labels is not None:
                group = labels[nodes]
                group = group[group >= 0]
                if len(group) and np.unique(group, return_counts=True)[1].max() > cap:
                    return False
            return True

        if not admissible(held):
            self.guard_stats["rejected_attempts"] += 1
            return []
        cd = np.bincount(self.dc[held], minlength=self.n_dc)
        cp = np.bincount(self.prov[held], minlength=self.n_prov)
        cr = np.bincount(self.reg[held], minlength=self.n_reg)
        co = np.bincount(self.own[held], minlength=self.n_own)
        cs = np.bincount(self.subnet_domain[held], minlength=self.n_subnet) if self.subnet_domain is not None else None
        out = []
        for _ in range(m):
            scores = np.zeros(len(cand), dtype=float)
            for pos, node in enumerate(cand):
                if node in selected or self.w[node] <= 0:
                    continue
                if (cd[self.dc[node]] >= dc_cap or cp[self.prov[node]] >= r_prov or
                        cr[self.reg[node]] >= r_reg or co[self.own[node]] >= r_own or
                        (cs is not None and cs[self.subnet_domain[node]] >= r_own)):
                    continue
                if not admissible(np.asarray(selected + [int(node)], dtype=int)):
                    continue
                penalty = (float(self._guard_conf[node, selected].sum())
                           if self.link_penalty and self._guard_conf is not None and selected else 0.)
                scores[pos] = self.w[node] / (1. + self.link_penalty * penalty)
            if not scores.any():
                self.guard_stats["rejected_attempts"] += 1
                return []
            if greedy:
                pos = int(np.argmax(scores))
            else:
                pos = min(int(np.searchsorted(np.cumsum(scores), rng.random()*scores.sum(), side="right")), len(cand)-1)
            node = int(cand[pos])
            out.append(node)
            selected.append(node)
            cd[self.dc[node]] += 1
            cp[self.prov[node]] += 1
            cr[self.reg[node]] += 1
            co[self.own[node]] += 1
            if cs is not None:
                cs[self.subnet_domain[node]] += 1
        indices = np.asarray(selected, dtype=int)
        groups = labels[indices] if labels is not None else np.empty(0, int)
        groups = groups[groups >= 0]
        max_group = int(np.unique(groups, return_counts=True)[1].max()) if len(groups) else 0
        self.guard_log.append({"t": self._guard_t, "snapshot_t": self._guard_snapshot.t if self._guard_snapshot else None,
                               "target_size": len(selected), "max_group": max_group, "group_cap": cap,
                               "novices": int(novice[indices].sum()), "novice_cap": self.k - 1})
        return out
