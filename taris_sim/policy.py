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

    def release(self, t: float, view) -> list[tuple[int, int]]:
        """Высвобождение лишнего фрагмента: (объект, номер фрагмента)."""
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


class FilecoinInspiredPolicy(UniformPolicy):
    """Упрощённый Filecoin-inspired baseline: равномерное размещение,
    общий залог оператора и штраф за провал аудита доступности.

    Денежная механика нормирована и служит только для сравнительного эксперимента;
    она не эмулирует сектора, storage deals или WindowPoSt Filecoin.
    """

    def __init__(self, world, n: int = 13, collateral_per_capacity: float = 10.0,
                 fault_penalty: float = 1.0):
        super().__init__(n)
        nd = world.nodes
        self.owner = nd["owner"].to_numpy(dtype=int)
        capacity = nd["cap_scale"].to_numpy(dtype=float)
        self.balance = {}
        for owner in np.unique(self.owner):
            self.balance[int(owner)] = float(collateral_per_capacity * capacity[self.owner == owner].sum())
        self.fault_penalty = float(fault_penalty)
        self.collateral_per_capacity = float(collateral_per_capacity)
        self.faults = 0
        self.slashed = 0.0
        self.exhausted_owners = set()
        self.empty_eligible_fallbacks = 0
        self.name = f"Filecoin-inspired, n={n}"

    def _observe_faults(self, view):
        step = getattr(view, "step", None)
        audits = step.get("audits") if step else None
        if audits is None or not len(audits):
            return
        failed = audits.loc[~audits["passed_avail"].astype(bool), "node"].to_numpy(dtype=int)
        for node in failed:
            owner = int(self.owner[node])
            before = self.balance.get(owner, 0.0)
            after = max(0.0, before - self.fault_penalty)
            self.balance[owner] = after
            self.slashed += before - after
            self.faults += 1
            if after <= 0.0:
                self.exhausted_owners.add(owner)

    def _eligible(self, candidates):
        candidates = np.asarray(candidates, dtype=int)
        if not len(candidates):
            return candidates
        ok = np.fromiter((self.balance.get(int(self.owner[i]), 0.0) > 0.0 for i in candidates),
                         dtype=bool, count=len(candidates))
        eligible = candidates[ok]
        if len(eligible):
            return eligible
        # Keep storage service operating if the stylized collateral pool is exhausted network-wide.
        self.empty_eligible_fallbacks += 1
        return candidates

    def place(self, obj, t, view):
        return self._choose(self._rng(view), self._eligible(view.candidates()), self.n)

    def on_epoch(self, t, view):
        self._observe_faults(view)
        self._g = self._rng(view)
        cand = self._eligible(view.candidates())
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose(self._g, cand, len(idx), exclude=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class SiaInspiredPolicy(UniformPolicy):
    """Упрощённый Sia-inspired host score по наблюдаемому uptime, возрасту и ёмкости."""

    def __init__(self, world, n: int = 13, history_days: int = 30, mature_days: float = 83.5):
        super().__init__(n)
        nd = world.nodes
        self.join_t = nd["join_t"].to_numpy(dtype=float)
        self.capacity = np.maximum(nd["cap_scale"].to_numpy(dtype=float), 1e-9)
        self.capacity /= max(float(np.median(self.capacity)), 1e-9)
        self.history_days = int(history_days)
        self.mature_days = float(mature_days)
        self.audit_history = [[] for _ in range(len(nd))]
        self._scored_t = None
        self._weights = np.ones(len(nd), dtype=float)
        self.name = f"Sia-inspired, n={n}"

    @staticmethod
    def _uptime_multiplier(u):
        # Узловая кривая упрощает legacy host-score ступени из документации Sia.
        return np.interp(u, [0.0, 0.80, 0.85, 0.90, 0.95, 0.98, 1.0],
                         [0.001, 0.03, 0.16, 0.51, 0.91, 1.0, 1.0])

    def _observe_audits(self, view):
        step = getattr(view, "step", None)
        audits = step.get("audits") if step else None
        if audits is None or not len(audits):
            return
        for row in audits.itertuples(index=False):
            self.audit_history[int(row.node)].append((float(row.epoch), bool(row.passed_avail)))

    def _score(self, t):
        if self._scored_t == t:
            return self._weights
        cutoff = t - self.history_days
        good = np.zeros(len(self.audit_history), dtype=float)
        total = np.zeros_like(good)
        for i, events in enumerate(self.audit_history):
            recent = [(day, ok) for day, ok in events if day >= cutoff]
            if len(recent) != len(events):
                self.audit_history[i] = recent
            total[i] = len(recent)
            good[i] = sum(ok for _, ok in recent)
        uptime = np.divide(good, total, out=np.ones_like(good), where=total > 0)
        reliability = self._uptime_multiplier(uptime)
        age = np.maximum(0.0, t - self.join_t)
        age_score = np.clip(age / max(self.mature_days, 1e-9), 0.01, 1.0)
        # Collateral and price are constant across hosts because TARIS has no
        # node-level Sia contract prices; this score therefore stays neutral.
        self._weights = np.maximum(reliability * age_score * self.capacity, 1e-12)
        self._scored_t = t
        return self._weights

    def _weighted_choose(self, rng, cand, m, t, exclude=()):
        cand = np.asarray(cand, dtype=int)
        excluded = set(exclude)
        if excluded:
            cand = cand[~np.isin(cand, list(excluded))]
        if len(cand) < m:
            return []
        weights = self._score(t)[cand].copy()
        out = []
        for _ in range(m):
            total = weights.sum()
            if total <= 0:
                return []
            j = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
            j = min(j, len(cand) - 1)
            out.append(int(cand[j]))
            weights[j] = 0.0
        return out

    def place(self, obj, t, view):
        return self._weighted_choose(self._rng(view), view.candidates(), self.n, t)

    def on_epoch(self, t, view):
        self._observe_audits(view)
        self._scored_t = None
        self._g = self._rng(view)
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._weighted_choose(self._g, cand, len(idx), t, exclude=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class StorjInspiredPolicy(UniformPolicy):
    """Упрощённый Storj-inspired выбор: репутация аудитов, vetting и diversity по подсети."""

    def __init__(self, world, n: int = 13, threshold: float = 0.96,
                 prior_success: float = 24.0, prior_failure: float = 1.0,
                 vetting_audits: int = 100, vetting_days: float = 14.0,
                 unvetted_weight: float = 0.05):
        super().__init__(n)
        nd = world.nodes
        self.join_t = nd["join_t"].to_numpy(dtype=float)
        self.subnet = nd["subnet"].astype(str).to_numpy()
        _, self.subnet_id = np.unique(self.subnet, return_inverse=True)
        self.n_subnets = int(self.subnet_id.max()) + 1
        self.n_nodes = len(nd)
        self.threshold = float(threshold)
        self.prior_success = float(prior_success)
        self.prior_failure = float(prior_failure)
        self.vetting_audits = int(vetting_audits)
        self.vetting_days = float(vetting_days)
        self.unvetted_weight = float(unvetted_weight)
        self.good = np.zeros(self.n_nodes, dtype=float)
        self.bad = np.zeros(self.n_nodes, dtype=float)
        self.audit_count = np.zeros(self.n_nodes, dtype=float)
        self._weights = np.ones(self.n_nodes, dtype=float)
        self._scored_t = None
        self.disqualified = 0
        self.subnet_relaxations = 0
        self.name = f"Storj-inspired, n={n}"

    def _observe_audits(self, view):
        step = getattr(view, "step", None)
        audits = step.get("audits") if step else None
        if audits is None or not len(audits):
            return
        for row in audits.itertuples(index=False):
            node = int(row.node)
            passed = bool(row.passed_avail)
            self.audit_count[node] += 1.0
            if passed:
                self.good[node] += 1.0
            else:
                self.bad[node] += 1.0

    def _score(self, t):
        if self._scored_t == t:
            return self._weights
        den = self.prior_success + self.prior_failure + self.audit_count
        rep = (self.prior_success + self.good) / den
        vetted = ((self.audit_count >= self.vetting_audits)
                  & ((t - self.join_t) >= self.vetting_days))
        self._weights = np.where(rep >= self.threshold, np.where(vetted, rep, self.unvetted_weight * rep), 0.0)
        self._scored_t = t
        return self._weights

    def _choose(self, rng, cand, m, t, exclude=()):
        cand = np.asarray(cand, dtype=int)
        if len(set(exclude)):
            cand = cand[~np.isin(cand, np.asarray(list(set(exclude)), dtype=int))]
        if len(cand) < m:
            return []
        weights = self._score(t)
        eligible = cand[weights[cand] > 0.0]
        selection_weight = weights.copy()
        if len(eligible) < m:
            self.disqualified += max(0, len(cand) - len(eligible))
            selection_weight = np.where(weights > 0.0, weights, 1e-12)
            eligible = cand                         # emergency admission to avoid placement failure
        for subnet_cap in range(1, m + 1):
            available = np.isin(cand, eligible)
            subnet_counts = np.zeros(self.n_subnets, dtype=int)
            out = []
            while len(out) < m:
                opts = cand[available & (subnet_counts[self.subnet_id[cand]] < subnet_cap)]
                if not len(opts):
                    break
                # Power-of-two choices approximation: sample two eligible nodes and prefer higher reputation.
                pair = rng.choice(opts, size=min(2, len(opts)), replace=False)
                node = int(pair[np.argmax(selection_weight[pair])])
                out.append(node)
                available[np.flatnonzero(cand == node)[0]] = False
                subnet_counts[self.subnet_id[node]] += 1
            if len(out) == m:
                if subnet_cap > 1:
                    self.subnet_relaxations += 1
                return out
        return []

    def place(self, obj, t, view):
        return self._choose(self._rng(view), view.candidates(), self.n, t)

    def on_epoch(self, t, view):
        self._observe_audits(view)
        self._scored_t = None
        self._g = self._rng(view)
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose(self._g, cand, len(idx), t, exclude=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class DynamicECPolicy(UniformPolicy):
    """Global redundancy tiers selected from recent observed audit-failure rates."""

    def __init__(self, world, base_n: int = 13, middle_n: int = 15, high_n: int = 17,
                 window_days: int = 30, medium_threshold: float = 0.02,
                 high_threshold: float = 0.05):
        super().__init__(base_n)
        self.k = int(world.cfg.k)
        self.levels = tuple(sorted({max(self.k, int(x)) for x in (base_n, middle_n, high_n)}))
        if len(self.levels) < 2:
            raise ValueError("Dynamic-EC требует не менее двух уровней избыточности")
        self.window_days = int(window_days)
        self.medium_threshold = float(medium_threshold)
        self.high_threshold = float(high_threshold)
        self.audit_history = []
        self.audit_fail_rate = 0.0
        self.target_n = self.levels[0]
        self.name = f"Dynamic-EC-inspired, уровни={self.levels}"

    def _observe(self, view, t):
        step = getattr(view, "step", None)
        audits = step.get("audits") if step else None
        if audits is not None and len(audits):
            self.audit_history.extend((float(row.epoch), not bool(row.passed_avail))
                                      for row in audits.itertuples(index=False))
        cutoff = t - self.window_days
        self.audit_history = [(day, fail) for day, fail in self.audit_history if day >= cutoff]
        self.audit_fail_rate = (sum(fail for _, fail in self.audit_history) / len(self.audit_history)
                                if self.audit_history else 0.0)
        if self.audit_fail_rate >= self.high_threshold:
            self.target_n = self.levels[-1]
        elif self.audit_fail_rate >= self.medium_threshold:
            self.target_n = self.levels[min(1, len(self.levels) - 1)]
        else:
            self.target_n = self.levels[0]

    def target_fragments(self, obj, t, view):
        return self.target_n

    def place(self, obj, t, view):
        return self._choose(self._rng(view), view.candidates(), self.target_n)

    def on_epoch(self, t, view):
        self._observe(view, t)
        self.n = self.target_n
        return super().on_epoch(t, view)


class AECInspiredPolicy(UniformPolicy):
    """Упрощённая audit-reputation policy с tier-based proactive migration."""

    def __init__(self, world, n: int = 13, initial_reputation: float = 0.96,
                 audit_step: float = 0.10, low_tier: float = 0.85,
                 preferred_tier: float = 0.96, migration_interval: int = 7,
                 max_migrations: int = 200):
        super().__init__(n)
        self.reputation = np.full(len(world.nodes), float(initial_reputation), dtype=float)
        self.audit_step = float(audit_step)
        self.low_tier = float(low_tier)
        self.preferred_tier = float(preferred_tier)
        self.migration_interval = int(migration_interval)
        self.max_migrations = int(max_migrations)
        self.audit_count = 0
        self.migration_candidates = 0
        self.name = f"AEC-DS-inspired, n={n}"

    def _observe(self, view):
        step = getattr(view, "step", None)
        audits = step.get("audits") if step else None
        if audits is None or not len(audits):
            return
        for row in audits.itertuples(index=False):
            node = int(row.node)
            passed = 1.0 if bool(row.passed_avail) else 0.0
            self.reputation[node] = ((1.0 - self.audit_step) * self.reputation[node]
                                     + self.audit_step * passed)
            self.audit_count += 1

    def _weighted_choose(self, rng, cand, m, exclude=()):
        cand = np.asarray(cand, dtype=int)
        excluded = set(exclude)
        if excluded:
            cand = cand[~np.isin(cand, list(excluded))]
        if len(cand) < m:
            return []
        weights = np.maximum(self.reputation[cand], 1e-9).copy()
        out = []
        for _ in range(m):
            tot = weights.sum()
            if tot <= 0:
                return []
            j = int(np.searchsorted(np.cumsum(weights), rng.random() * tot, side="right"))
            j = min(j, len(cand) - 1)
            out.append(int(cand[j])); weights[j] = 0.0
        return out

    def place(self, obj, t, view):
        return self._weighted_choose(self._rng(view), view.candidates(), self.n)

    def on_epoch(self, t, view):
        self._observe(view)
        self._g = self._rng(view)
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._weighted_choose(self._g, cand, len(idx), exclude=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs

    def migrate(self, t, view):
        if t <= 0 or int(t) % self.migration_interval != 0 or self.max_migrations <= 0:
            return []
        low = np.flatnonzero(self.reputation < self.low_tier)
        if not len(low):
            return []
        good = view.candidates()
        good = good[self.reputation[good] >= self.preferred_tier]
        if not len(good):
            return []
        rng = self._rng(view)
        out, moved = [], set()
        for node in low:
            for obj, idx in view.on_node(int(node)):
                if len(out) >= self.max_migrations:
                    return out
                if (obj in moved or idx in view.migrating(obj)
                        or view.L.obj_lost_t[obj] != np.inf):
                    continue
                held = view.holders(obj) | {to for o, _, to in out if o == obj}
                receivers = self._weighted_choose(rng, good, 1, exclude=held)
                if receivers:
                    out.append((obj, idx, receivers[0])); moved.add(obj)
        self.migration_candidates += len(out)
        return out


class _PoissonStumpBoost:
    """Небольшой детерминированный Poisson gradient booster с деревьями-гранями."""

    def __init__(self, rounds=15, learning_rate=0.08, l2=1.0, bins=7):
        self.rounds, self.learning_rate, self.l2, self.bins = rounds, learning_rate, l2, bins
        self.trees = []
        self.base = 0.0
        self.fitted = False

    def fit(self, X, y, exposure):
        X, y, exposure = np.asarray(X, float), np.asarray(y, float), np.asarray(exposure, float)
        keep = np.isfinite(X).all(1) & np.isfinite(y) & np.isfinite(exposure) & (exposure > 0)
        X, y, exposure = X[keep], y[keep], exposure[keep]
        if len(y) < 100 or y.sum() < 10:
            return False
        self.base = float(np.log(max(y.sum(), 0.5) / exposure.sum()))
        eta = np.full(len(y), self.base)
        self.trees = []
        splits = []
        for j in range(X.shape[1]):
            cuts = np.unique(np.quantile(X[:, j], np.linspace(0.1, 0.9, self.bins)))
            order = np.argsort(X[:, j], kind="mergesort")
            positions = np.searchsorted(X[order, j], cuts, side="right")
            valid = (positions > 0) & (positions < len(y))
            if valid.any():
                splits.append((j, cuts[valid], order, positions[valid]))
        offset = np.log(exposure)
        for _ in range(self.rounds):
            mu = np.exp(np.clip(offset + eta, -30.0, 30.0))
            grad, hess = y - mu, mu
            sum_g, sum_h = grad.sum(), hess.sum()
            best_gain, best = 0.0, None
            for j, cuts, order, positions in splits:
                cg = np.cumsum(grad[order]); ch = np.cumsum(hess[order])
                gl, hl = cg[positions - 1], ch[positions - 1]
                gr, hr = sum_g - gl, sum_h - hl
                gains = 0.5 * (gl * gl / (hl + self.l2) + gr * gr / (hr + self.l2)
                               - sum_g * sum_g / (sum_h + self.l2))
                ix = int(np.argmax(gains))
                if gains[ix] > best_gain:
                    best_gain = float(gains[ix])
                    best = (j, float(cuts[ix]), self.learning_rate * float(gl[ix]) / (float(hl[ix]) + self.l2),
                            self.learning_rate * float(gr[ix]) / (float(hr[ix]) + self.l2))
            if best is None:
                break
            j, cut, left, right = best
            left, right = float(np.clip(left, -1.0, 1.0)), float(np.clip(right, -1.0, 1.0))
            eta += np.where(X[:, j] <= cut, left, right)
            self.trees.append((j, cut, left, right))
        self.fitted = bool(self.trees)
        return self.fitted

    def predict_rate(self, X):
        X = np.asarray(X, float)
        eta = np.full(len(X), self.base)
        for j, cut, left, right in self.trees:
            eta += np.where(X[:, j] <= cut, left, right)
        return np.exp(np.clip(eta, -30.0, 20.0))


class GradientBoostingPolicy(UniformPolicy):
    """Gradient-boosted Poisson frequency baseline trained only on network observations."""

    def __init__(self, world, n: int = 13, block: int = 30,
                 rounds: int = 15, learning_rate: float = 0.08):
        super().__init__(n)
        from .experience import Experience
        self.world = world
        self.exp = Experience(world, block=block)
        self.block = int(block)
        self.model = _PoissonStumpBoost(rounds=rounds, learning_rate=learning_rate)
        self.last_fit = -1
        self.weights = np.ones(self.exp.N, dtype=float)
        self.fit_rows = 0
        self.name = f"Градиентный бустинг Пуассона, n={n}"

    def _features(self, region, provider, hw, device, age, micro):
        r = self.exp.reg
        cols = [np.log1p(np.asarray(age, float) / 30.0),
                np.log(np.maximum(np.asarray(micro, float), 1e-12))]
        for vals, levels in ((region, r.n_region), (provider, int(np.max(r.provider)) + 1),
                             (hw, r.n_hw), (device, 2)):
            vals = np.asarray(vals, int)
            cols.extend([(vals == j).astype(float) for j in range(1, levels)])
        return np.column_stack(cols)

    def _fit(self, t):
        if t <= 0 or int(t) % self.block != 0 or int(t) == self.last_fit:
            return
        tab = self.exp.table(t)
        tab = tab[tab["e"] > 0]
        if len(tab) < 100 or tab["N"].sum() < 10:
            self.last_fit = int(t)
            return
        X = self._features(tab.region, tab.provider, tab.hw, tab.device, tab.age, tab.micro_rate)
        if self.model.fit(X, tab["N"].to_numpy(), tab["e"].to_numpy()):
            self.fit_rows = len(tab)
        self.last_fit = int(t)

    def _refresh_weights(self, t, view):
        if not self.model.fitted:
            self.weights = np.ones(self.exp.N, dtype=float)
            return
        r = self.exp.reg
        join = view.join_times()
        age = np.where(np.isfinite(join), np.maximum(t - join, 0.0), 0.0)
        micro = self.exp.micro_rate_now(t)
        X = self._features(r.region, r.provider, r.hw, r.is_device.astype(int), age, micro)
        rates = self.model.predict_rate(X)
        self.weights = 1.0 / np.maximum(rates, 1e-12)
        self.weights /= max(float(np.mean(self.weights)), 1e-12)

    def _weighted_choose(self, rng, cand, m, exclude=()):
        cand = np.asarray(cand, dtype=int)
        excluded = set(exclude)
        if excluded:
            cand = cand[~np.isin(cand, list(excluded))]
        if len(cand) < m:
            return []
        weights = self.weights[cand].copy()
        out = []
        for _ in range(m):
            total = weights.sum()
            if total <= 0:
                return []
            j = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
            j = min(j, len(cand) - 1)
            out.append(int(cand[j])); weights[j] = 0.0
        return out

    def place(self, obj, t, view):
        return self._weighted_choose(self._rng(view), view.candidates(), self.n)

    def on_epoch(self, t, view):
        self.exp.update(t, view)
        self._fit(t)
        self._refresh_weights(t, view)
        self._g = self._rng(view)
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._weighted_choose(self._g, cand, len(idx), exclude=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class K1PoissonGammaPolicy(UniformPolicy):
    """Эмпирическая Poisson-Gamma модель с маргинальной подгонкой гиперпараметров."""

    def __init__(self, world, n: int = 13, block: int = 30,
                 min_group_nodes: int = 30, min_group_events: int = 5):
        super().__init__(n)
        from .experience import Experience
        self.exp = Experience(world, block=block)
        self.block = int(block)
        self.min_group_nodes = int(min_group_nodes)
        self.min_group_events = int(min_group_events)
        self.shape_global, self.rate_global = 2.0, 2.0
        self.group_prior = {}
        self.node_rate = np.full(self.exp.N, 1.0)
        self.fitted = False
        self.last_fit = -1
        self.fit_groups = 0
        self.name = f"K1 Пуассон–гамма, n={n}"

    @staticmethod
    def _fit_gamma_prior(y, e):
        from scipy.optimize import minimize
        from scipy.special import gammaln
        y, e = np.asarray(y, float), np.asarray(e, float)
        keep = (e > 0) & np.isfinite(e) & np.isfinite(y)
        y, e = y[keep], e[keep]
        if len(y) < 5:
            rate = max(float(y.sum() / max(e.sum(), 1e-12)), 1e-8)
            return 2.0, 2.0 / rate

        def objective(z):
            a, b = np.exp(z)
            ll = (gammaln(a + y) - gammaln(a) - gammaln(y + 1.0)
                  + a * np.log(b) + y * np.log(e) - (a + y) * np.log(b + e))
            return -float(ll.sum())

        rate = max(float(y.sum() / e.sum()), 1e-8)
        init = np.log([2.0, 2.0 / rate])
        fit = minimize(objective, init, method="L-BFGS-B", bounds=[(-7.0, 8.0), (-16.0, 12.0)])
        if not fit.success or not np.isfinite(fit.fun):
            return 2.0, 2.0 / rate
        return tuple(np.exp(fit.x))

    def _fit(self, t):
        if t <= 0 or int(t) % self.block != 0 or int(t) == self.last_fit:
            return
        tab = self.exp.table(t)
        tab = tab[tab["e"] > 0]
        self.last_fit = int(t)
        if len(tab) < 100 or tab["N"].sum() < 10:
            return
        N = self.exp.N
        y = np.bincount(tab["node"].to_numpy(), weights=tab["N"].to_numpy(), minlength=N)
        e = np.bincount(tab["node"].to_numpy(), weights=tab["e"].to_numpy(), minlength=N)
        self.shape_global, self.rate_global = self._fit_gamma_prior(y, e)
        reg = self.exp.reg
        keys = list(zip(reg.region, reg.hw, reg.is_device.astype(int)))
        groups = {}
        for node, key in enumerate(keys):
            groups.setdefault(tuple(map(int, key)), []).append(node)
        self.group_prior = {}
        for key, ids in groups.items():
            ids = np.asarray(ids, dtype=int)
            if len(ids) >= self.min_group_nodes and y[ids].sum() >= self.min_group_events:
                self.group_prior[key] = self._fit_gamma_prior(y[ids], e[ids])
            else:
                self.group_prior[key] = (self.shape_global, self.rate_global)
        rates = np.empty(N, dtype=float)
        for node, key in enumerate(keys):
            a, b = self.group_prior[tuple(map(int, key))]
            rates[node] = (a + y[node]) / (b + e[node])
        self.node_rate = np.maximum(rates, 1e-12)
        self.fitted = True
        self.fit_groups = len(self.group_prior)

    def _choose(self, rng, cand, m, exclude=()):
        cand = np.asarray(cand, dtype=int)
        excluded = set(exclude)
        if excluded:
            cand = cand[~np.isin(cand, list(excluded))]
        if len(cand) < m:
            return []
        weights = 1.0 / self.node_rate[cand]
        out = []
        for _ in range(m):
            total = weights.sum()
            j = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
            j = min(j, len(cand) - 1)
            out.append(int(cand[j])); weights[j] = 0.0
        return out

    def place(self, obj, t, view):
        return self._choose(self._rng(view), view.candidates(), self.n)

    def on_epoch(self, t, view):
        self.exp.update(t, view)
        self._fit(t)
        self._g = self._rng(view)
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose(self._g, cand, len(idx), exclude=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class K2LaplacePolicy(UniformPolicy):
    """Poisson-lognormal node/cell random-intercept model with a Laplace/MAP fit.

    This is an executable approximation to the planned K2 model; it is not NUTS
    and reports no full posterior uncertainty.
    """

    def __init__(self, world, n: int = 13, block: int = 30, iterations: int = 25):
        super().__init__(n)
        from .experience import Experience
        self.exp = Experience(world, block=block)
        self.block = int(block)
        self.iterations = int(iterations)
        r = self.exp.reg
        cell_keys = list(zip(r.hw.tolist(), r.region.tolist()))
        self.node_cell, self.cell_keys = np.unique(np.asarray(cell_keys), axis=0, return_inverse=True)
        self.cell_effect = np.zeros(len(self.node_cell))
        self.node_effect = np.zeros(self.exp.N)
        self.beta0 = 0.0
        self.sigma_node = 0.5
        self.sigma_cell = 0.25
        self.rates = np.ones(self.exp.N)
        self.fitted = False
        self.last_fit = -1
        self.name = f"K2 Пуассон–логнормальная Laplace, n={n}"

    @staticmethod
    def _newton_random(y, base_exposure, effect, variance, groups, n_groups):
        counts = np.bincount(groups, weights=y, minlength=n_groups)
        exposure = np.bincount(groups, weights=base_exposure, minlength=n_groups)
        precision = 1.0 / max(variance, 1e-4)
        z = effect.copy()
        for _ in range(5):
            mu = exposure * np.exp(np.clip(z, -15, 15))
            grad = counts - mu - z * precision
            hess = mu + precision
            z += np.clip(grad / np.maximum(hess, 1e-9), -1.0, 1.0)
        post_var = 1.0 / np.maximum(exposure * np.exp(np.clip(z, -15, 15)) + precision, 1e-9)
        new_var = float(np.clip(np.mean(z * z - post_var), 0.01, 2.25))
        return z, new_var

    def _fit(self, t):
        if t <= 0 or int(t) % self.block != 0 or int(t) == self.last_fit:
            return
        self.last_fit = int(t)
        tab = self.exp.table(t)
        tab = tab[tab["e"] > 0]
        if len(tab) < 100 or tab["N"].sum() < 10:
            return
        y = tab["N"].to_numpy(dtype=float)
        e = tab["e"].to_numpy(dtype=float)
        node = tab["node"].to_numpy(dtype=int)
        cell = self.cell_keys[node]
        beta0 = float(np.log(max(y.sum() / e.sum(), 1e-8)))
        u, v = self.node_effect.copy(), self.cell_effect.copy()
        su2, sv2 = self.sigma_node ** 2, self.sigma_cell ** 2
        for _ in range(self.iterations):
            eta = np.clip(beta0 + u[node] + v[cell], -20.0, 10.0)
            mu = e * np.exp(eta)
            beta0 += float(np.clip((y - mu).sum() / max(mu.sum(), 1e-9), -0.5, 0.5))
            base_node = e * np.exp(np.clip(beta0 + v[cell], -20.0, 10.0))
            u, su2 = self._newton_random(y, base_node, u, su2, node, self.exp.N)
            base_cell = e * np.exp(np.clip(beta0 + u[node], -20.0, 10.0))
            v, sv2 = self._newton_random(y, base_cell, v, sv2, cell, len(v))
        self.beta0, self.node_effect, self.cell_effect = beta0, u, v
        self.sigma_node, self.sigma_cell = float(np.sqrt(su2)), float(np.sqrt(sv2))
        self.rates = np.exp(np.clip(beta0 + u + v[self.cell_keys], -25.0, 10.0))
        self.fitted = True

    def _choose(self, rng, cand, m, exclude=()):
        cand = np.asarray(cand, dtype=int)
        excluded = set(exclude)
        if excluded:
            cand = cand[~np.isin(cand, list(excluded))]
        if len(cand) < m:
            return []
        weights = 1.0 / np.maximum(self.rates[cand], 1e-12)
        out = []
        for _ in range(m):
            total = weights.sum()
            j = int(np.searchsorted(np.cumsum(weights), rng.random() * total, side="right"))
            j = min(j, len(cand) - 1)
            out.append(int(cand[j])); weights[j] = 0.0
        return out

    def place(self, obj, t, view):
        return self._choose(self._rng(view), view.candidates(), self.n)

    def on_epoch(self, t, view):
        self.exp.update(t, view)
        self._fit(t)
        self._g = self._rng(view)
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose(self._g, cand, len(idx), exclude=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class K2NUTSPolicy(K2LaplacePolicy):
    """Poisson-lognormal GLMM fitted by PyMC NUTS at a configurable interval.

    NUTS is substantially more expensive than the Laplace baseline. The default
    annual refit cadence is intentional and is recorded as part of this baseline.
    """

    def __init__(self, world, n: int = 13, fit_interval: int = 365,
                 draws: int = 500, tune: int = 500, chains: int = 4,
                 target_accept: float = 0.99):
        super().__init__(world, n=n, block=30, iterations=1)
        self.world = world
        self.fit_interval = max(1, int(fit_interval))
        self.draws, self.tune, self.chains = int(draws), int(tune), int(chains)
        self.target_accept = float(target_accept)
        self.fit_count = 0
        self.fit_seconds = 0.0
        self.divergences = 0
        self.max_rhat = float("nan")
        self.min_ess_bulk = float("nan")
        self.rhat_beta = float("nan")
        self.rhat_scales = float("nan")
        self.rhat_node = float("nan")
        self.ess_node_bulk = float("nan")
        self.name = f"K2 Пуассон–логнормальная NUTS, refit={self.fit_interval}d"

    def _fit_nuts(self, t):
        if t <= 0 or int(t) % self.fit_interval != 0 or int(t) == self.last_fit:
            return
        import time
        import pymc as pm
        import pytensor.tensor as pt

        self.last_fit = int(t)
        tab = self.exp.table(t)
        tab = tab[tab["e"] > 0]
        if len(tab) < 100 or tab["N"].sum() < 10:
            return

        from .risk import RiskModel
        risk = RiskModel(self.exp, claimed=True, micro=True)
        risk.m0 = float(np.average(tab["micro_rate"], weights=tab["e"]))
        X = risk._cols(tab["region"].to_numpy(), tab["age"].to_numpy(),
                       tab["micro_rate"].to_numpy(), tab["hw"].to_numpy(),
                       tab["device"].to_numpy())
        keep = risk._keep(X)
        X = np.asarray(X[:, keep], dtype=np.float64)
        y = tab["N"].to_numpy(dtype=np.int64)
        exposure = tab["e"].to_numpy(dtype=np.float64)
        node_ids = np.unique(tab["node"].to_numpy(dtype=int))
        node_idx = np.searchsorted(node_ids, tab["node"].to_numpy(dtype=int)).astype("int32")
        obs_cell_idx = self.cell_keys[tab["node"].to_numpy(dtype=int)].astype("int32")
        node_total = np.bincount(node_idx, weights=y, minlength=len(node_ids)).astype(np.int64)
        gh_x, gh_w = np.polynomial.hermite.hermgauss(32)
        log_gh_w = np.log(gh_w) - 0.5 * np.log(np.pi)

        coords = {"obs": np.arange(len(y)), "coef": np.arange(X.shape[1]),
                  "cell": np.arange(len(self.node_cell))}
        started = time.perf_counter()
        seed = int((int(self.world.cfg.seed) * 1_000_003 + int(t) * 97) % (2**31 - 1))
        with pm.Model(coords=coords) as model:
            x_data = pm.Data("X", X, dims=("obs", "coef"))
            e_data = pm.Data("exposure", exposure, dims="obs")
            ni_data = pm.Data("node_idx", node_idx, dims="obs")
            ci_data = pm.Data("cell_idx", obs_cell_idx, dims="obs")
            beta = pm.Normal("beta", mu=0.0, sigma=2.0, dims="coef")
            sigma_node = pm.HalfNormal("sigma_node", sigma=1.0)
            sigma_cell = pm.HalfNormal("sigma_cell", sigma=1.0)
            z_cell = pm.Normal("z_cell", mu=0.0, sigma=1.0, dims="cell")
            u = pm.Deterministic("u", sigma_cell * z_cell, dims="cell")
            eta_base = pt.dot(x_data, beta) + u[ci_data]
            base_mu = e_data * pt.exp(eta_base)
            node_a = pt.extra_ops.bincount(ni_data, weights=base_mu, minlength=len(node_ids))
            y_data = pt.as_tensor_variable(y.astype(np.float64))
            node_linear = pt.extra_ops.bincount(
                ni_data, weights=y_data * (pt.log(e_data) + eta_base), minlength=len(node_ids))
            bq = np.sqrt(2.0) * sigma_node * gh_x
            log_lik_node = (node_total[:, None] * bq[None, :]
                            - node_a[:, None] * pt.exp(bq)[None, :]
                            + log_gh_w[None, :])
            # Exact marginalization of each Gaussian node effect; NUTS samples
            # the population/cell posterior, and quadrature recovers node posteriors.
            pm.Potential("node_marginal_likelihood",
                         node_linear.sum() + pt.logsumexp(log_lik_node, axis=1).sum())
            idata = pm.sample(draws=self.draws, tune=self.tune, chains=self.chains,
                              cores=1, target_accept=self.target_accept,
                              random_seed=seed, progressbar=False,
                              nuts_sampler="nutpie",
                              compute_convergence_checks=True, return_inferencedata=True)

        beta_s = np.asarray(idata.posterior["beta"], dtype=float).reshape(-1, X.shape[1])
        u_s = np.asarray(idata.posterior["u"], dtype=float).reshape(-1, len(self.node_cell))
        sig_s = np.asarray(idata.posterior["sigma_node"], dtype=float).reshape(-1)
        samples = len(sig_s)

        # Recover each node's posterior E[exp(b_i)] by quadrature conditional
        # on each population draw. This avoids sampling thousands of weakly
        # identified node effects while preserving their marginal posterior.
        a_s = np.empty((samples, len(node_ids)), dtype=float)
        for s in range(samples):
            base = exposure * np.exp(np.clip(beta_s[s] @ X.T + u_s[s, obs_cell_idx], -25.0, 10.0))
            a_s[s] = np.bincount(node_idx, weights=base, minlength=len(node_ids))
        node_mult = np.empty_like(a_s)
        for s in range(samples):
            bq = np.sqrt(2.0) * sig_s[s] * gh_x
            exp_bq = np.exp(bq)
            for start in range(0, len(node_ids), 512):
                stop = min(start + 512, len(node_ids))
                lp = (node_total[start:stop, None] * bq[None, :]
                      - a_s[s, start:stop, None] * exp_bq[None, :]
                      + log_gh_w[None, :])
                log_den = np.logaddexp.reduce(lp, axis=1)
                log_num = np.logaddexp.reduce(lp + bq[None, :], axis=1)
                node_mult[s, start:stop] = np.exp(log_num - log_den)

        # Current, network-only covariates; unseen nodes integrate over the node effect prior.
        reg = self.exp.reg
        join = np.where(np.isfinite(self.exp.join), self.exp.join, float(t))
        age = np.maximum(float(t) - join, 0.0)
        micro = self.exp.micro_rate_now(float(t))
        X_now = risk._cols(reg.region, age, micro, reg.hw, reg.is_device.astype(int))[:, keep]
        active_lookup = np.full(self.exp.N, -1, dtype=int)
        active_lookup[node_ids] = np.arange(len(node_ids))
        self.rates = np.empty(self.exp.N, dtype=float)
        chunk = 512
        for start in range(0, self.exp.N, chunk):
            stop = min(start + chunk, self.exp.N)
            ids = np.arange(start, stop)
            eta_s = beta_s @ X_now[start:stop].T + u_s[:, self.cell_keys[ids]]
            ai = active_lookup[ids]
            active = ai >= 0
            if active.any():
                eta_s[:, active] = (np.exp(np.clip(eta_s[:, active], -25.0, 10.0))
                                    * node_mult[:, ai[active]])
            if (~active).any():
                eta_s[:, ~active] = (np.exp(np.clip(eta_s[:, ~active], -25.0, 10.0))
                                     * np.exp(0.5 * sig_s[:, None] ** 2))
            self.rates[start:stop] = eta_s.mean(axis=0)

        self.beta0 = float(np.mean(beta_s[:, 0]))
        self.sigma_node = float(np.mean(sig_s))
        self.sigma_cell = float(np.asarray(idata.posterior["sigma_cell"], dtype=float).mean())
        self.node_effect[:] = 0.0
        self.node_effect[node_ids] = np.log(node_mult.mean(axis=0))
        self.cell_effect[:] = u_s.mean(axis=0)
        self.fitted = True
        self.fit_count += 1
        self.fit_seconds += time.perf_counter() - started
        self.divergences += int(np.asarray(idata.sample_stats["diverging"], dtype=bool).sum())
        try:
            import arviz as az
            vars_ = ["beta", "sigma_node", "sigma_cell", "z_cell"]
            rh_ds = az.rhat(idata, var_names=vars_).ds
            ess_ds = az.ess(idata, var_names=vars_, method="bulk").ds
            rh = rh_ds.to_array().values
            cur_rhat = float(np.nanmax(rh))
            self.max_rhat = max(self.max_rhat, cur_rhat) if np.isfinite(self.max_rhat) else cur_rhat
            ess = ess_ds.to_array().values
            cur_ess = float(np.nanmin(ess))
            self.min_ess_bulk = min(self.min_ess_bulk, cur_ess) if np.isfinite(self.min_ess_bulk) else cur_ess
            cur_beta = float(np.nanmax(rh_ds["beta"].values))
            self.rhat_beta = max(self.rhat_beta, cur_beta) if np.isfinite(self.rhat_beta) else cur_beta
            cur_scales = float(max(np.nanmax(rh_ds["sigma_node"].values),
                                   np.nanmax(rh_ds["sigma_cell"].values),
                                   np.nanmax(rh_ds["z_cell"].values)))
            self.rhat_scales = max(self.rhat_scales, cur_scales) if np.isfinite(self.rhat_scales) else cur_scales
            self.rhat_node = float("nan")
            self.ess_node_bulk = float("nan")
        except Exception:
            self.max_rhat = float("nan")
            self.min_ess_bulk = float("nan")

    def on_epoch(self, t, view):
        self.exp.update(t, view)
        self._fit_nuts(t)
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

    def __init__(self, world, n: int = 15, gamma: float = 1.0, oracle: bool = True, claimed: bool = True):
        super().__init__(n)
        self.gamma = float(gamma)
        nd = world.nodes
        if oracle:
            phi = np.array([world.true_phi(i) for i in range(len(nd))])
            q = nd["lam_avg"].to_numpy() * phi
            # Устройства чаще уходят: добавить повышенный риск ухода относительно серверов.
            q = q + (nd["m_churn"].to_numpy() - 1.0) * world.cfg.churn_base
            self.w = (1.0 / np.maximum(q, 1e-9)) ** gamma
            self.own = nd["owner"].to_numpy()
            self.subnet_domain = None
            self.name = f"Оракул, n={n}"
        else:
            # Только сетевые признаки и, если разрешено, заявленный узлом DC.
            # Владельца агент не знает; подсеть — доступный сети прокси-домен концентрации.
            from .experience import Registry
            reg = Registry(world)
            self.w = np.ones(len(nd), dtype=float)     # до первой оценки — равномерный выбор
            self.own = reg.wallet
            self.subnet_domain = reg.subnet
            self.name = f"Политика без оракула, n={n}"
            self._agent_domains = reg
            if claimed:
                self.dc = reg.dc.copy()
            else:
                # Не использовать заявленный DC даже как ограничение spread.
                self.dc = world.cfg.n_dc + np.arange(len(nd))
        if oracle:
            self._agent_domains = None
            self.dc = nd["dc"].to_numpy()
        self.n_subnet = int(self.subnet_domain.max()) + 1 if self.subnet_domain is not None else 0
        self.prov, self.reg = (nd[c].to_numpy() for c in ("provider", "region"))
        # у устройства нет дата-центра: оно — свой отдельный домен отказа (номера после дата-центров)
        dev = self.dc < 0
        self.dc = np.where(dev, world.cfg.n_dc + np.arange(len(nd)), self.dc)
        self.k = world.cfg.k
        self.n_dc, self.n_prov, self.n_reg = int(self.dc.max()) + 1, int(self.prov.max()) + 1, world.cfg.n_regions
        self.n_own = int(self.own.max()) + 1

    def _choose_spread(self, rng, cand, m, held=()):
        """Выбор с разнесением: в дата-центре — один фрагмент объекта, у провайдера, в регионе и у владельца —
        не больше r = min(n - k, k - 1). Если узлов не набрать (например, кластер-цель кризиса исключён),
        лимиты ослабляются по одному, начиная с самого безобидного: сначала регион (+1, +2, ...), потом
        дата-центр (2, 3, ... фрагмента), в крайнем случае — провайдер. Дата-центр — меньший домен отказа, чем
        провайдер: при немногих живых провайдерах лишний фрагмент лучше положить вторым в дата-центр, чем
        четвёртым к провайдеру, — простой провайдера дольше T_rep иначе губит объекты разом.
        Лимит на владельца не ослабляется (общий владелец — это риск раскрытия)."""
        r0, n = min(self.n - self.k, self.k - 1), self.n
        steps = ([(r0, r0 + rr, 1) for rr in range(n + 1)]                 # регион
                 + [(r0, r0 + n, dc) for dc in range(2, n + 1)]            # дата-центр
                 + [(r0 + rp, r0 + n, n) for rp in range(1, n + 1)])       # провайдер
        for r_prov, r_reg, dc_cap in steps:
            out = self._choose_spread_r(rng, cand, m, held, r_prov, r_reg, r0, dc_cap,
                                        greedy=getattr(self, "greedy_placement", False))
            if out:
                return out
        return []

    def _choose_spread_r(self, rng, cand, m, held, r_prov, r_reg, r_own, dc_cap=1, greedy=False):
        held = np.fromiter(held, int) if len(held) else np.empty(0, int)
        cd = np.bincount(self.dc[held], minlength=self.n_dc)
        cp = np.bincount(self.prov[held], minlength=self.n_prov)
        cr = np.bincount(self.reg[held], minlength=self.n_reg)
        co = np.bincount(self.own[held], minlength=self.n_own)          # фрагменты объекта у каждого владельца
        cs = (np.bincount(self.subnet_domain[held], minlength=self.n_subnet)
              if self.subnet_domain is not None else None)
        cand = cand[~np.isin(cand, held)] if len(held) else cand
        w = self.w[cand].astype(float)
        dc_c, prov_c, reg_c, own_c = self.dc[cand], self.prov[cand], self.reg[cand], self.own[cand]
        sub_c = self.subnet_domain[cand] if self.subnet_domain is not None else None
        out = []
        for _ in range(m):
            ok = (cd[dc_c] < dc_cap) & (cp[prov_c] < r_prov) & (cr[reg_c] < r_reg) & (co[own_c] < r_own)
            if cs is not None:
                ok &= cs[sub_c] < r_own
            ww = w * ok
            tot = ww.sum()
            if tot <= 0:
                return []
            if greedy:
                j = int(np.argmax(ww))          # стабильный выбор узла с наименьшим оценённым риском
            else:
                j = int(np.searchsorted(np.cumsum(ww), rng.random() * tot, side="right"))
                j = min(j, len(cand) - 1)
            c = int(cand[j]); out.append(c)
            cd[self.dc[c]] += 1; cp[self.prov[c]] += 1; cr[self.reg[c]] += 1
            co[self.own[c]] += 1
            if cs is not None:
                cs[self.subnet_domain[c]] += 1
            w[j] = 0.0
        return out

    def _cand(self, t, view):
        return view.candidates()

    def place(self, obj, t, view):
        return self._choose_spread(self._g if hasattr(self, "_g") else self._rng(view), self._cand(t, view), self.n)

    def on_epoch(self, t, view):
        self._g = self._rng(view)
        cand = self._cand(t, view)
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose_spread(self._g, cand, len(idx), held=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class CrushLikePolicy(OraclePolicy):
    """Детерминированное размещение в стиле CRUSH (шаг 4.2а).

    Использует стабильное ранжирование rendezvous-hash по паре «объект, узел»,
    вес узла пропорционален его ёмкости, а ограничения доменов отказа наследует
    от политики размещения. Это структурный ориентир, а не полная реализация
    CRUSH: нет иерархических bucket-алгоритмов straw2 и гарантий минимального
    перемещения при изменении топологии.
    """

    name = "CRUSH-подобное размещение"

    def __init__(self, world, n: int = 13):
        super().__init__(world, n=n, oracle=False, claimed=True)
        self.capacity = np.maximum(world.nodes["cap_scale"].to_numpy(dtype=float), 1e-9)
        self._rank_cache = {}
        self.greedy_placement = True
        self.name = f"CRUSH-подобное, n={n}"

    def _ranked_candidates(self, cand, obj):
        cand = np.asarray(cand, dtype=int)
        key = int(obj)
        if key not in self._rank_cache:
            ids = np.arange(len(self.capacity), dtype=np.uint64)
            obj_mix = np.uint64((key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)
            x = ids + obj_mix + np.uint64(0xD1B54A32D192ED03)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x ^= x >> np.uint64(31)
            u = ((x >> np.uint64(11)).astype(float) + 0.5) / 9007199254740992.0
            score = -np.log(u) / self.capacity
            self._rank_cache[key] = np.argsort(score, kind="mergesort")
        live = np.zeros(len(self.capacity), dtype=bool)
        live[cand] = True
        order = self._rank_cache[key]
        return order[live[order]]

    def _choose_spread(self, rng, cand, m, held=(), obj=0):
        return self._fast_spread(cand, m, held, obj)

    def _fast_spread(self, cand, m, held, obj):
        """Single-rank deterministic spread selection; same relaxation order as OraclePolicy."""
        cand = np.asarray(cand, dtype=int)
        held = np.asarray(list(held), dtype=int) if len(held) else np.empty(0, int)
        if len(held):
            cand = cand[~np.isin(cand, held)]
        if len(cand) < m:
            return []
        order = self._ranked_candidates(cand, obj)
        r0 = min(self.n - self.k, self.k - 1)
        steps = ([(r0, r0 + rr, 1) for rr in range(self.n + 1)]
                 + [(r0, r0 + self.n, dc) for dc in range(2, self.n + 1)]
                 + [(r0 + rp, r0 + self.n, self.n) for rp in range(1, self.n + 1)])
        for r_prov, r_reg, dc_cap in steps:
            cd = np.bincount(self.dc[held], minlength=self.n_dc)
            cp = np.bincount(self.prov[held], minlength=self.n_prov)
            cr = np.bincount(self.reg[held], minlength=self.n_reg)
            co = np.bincount(self.own[held], minlength=self.n_own)
            if m == 1:
                ok = ((cd[self.dc[order]] < dc_cap) & (cp[self.prov[order]] < r_prov)
                      & (cr[self.reg[order]] < r_reg) & (co[self.own[order]] < r0))
                found = np.flatnonzero(ok)
                if len(found):
                    return [int(order[found[0]])]
                continue
            out = []
            for node in order:
                if (cd[self.dc[node]] < dc_cap and cp[self.prov[node]] < r_prov
                        and cr[self.reg[node]] < r_reg and co[self.own[node]] < r0):
                    out.append(int(node))
                    cd[self.dc[node]] += 1; cp[self.prov[node]] += 1
                    cr[self.reg[node]] += 1; co[self.own[node]] += 1
                    if len(out) == m:
                        return out
        return []

    def place(self, obj, t, view):
        return self._choose_spread(None, view.candidates(), self.n, obj=obj)

    def on_epoch(self, t, view):
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose_spread(None, cand, len(idx), held=view.holders(obj), obj=obj)
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class CopysetsPolicy(CrushLikePolicy):
    """Ограниченный пул copysets с небольшим запасом узлов (шаг 4.2б).

    Объекты закрепляются за одним пулом, а при размещении выбирают узлы только
    из него. Если в пуле временно/постоянно недостаточно доступных узлов, включается
    безопасный CRUSH-подобный spillover; такие случаи учитываются в spill_count.
    """

    def __init__(self, world, n: int = 13, scatter: int = 4, slack: int = 6):
        super().__init__(world, n=n)
        self.scatter = max(1, int(scatter))
        self.slack = max(0, int(slack))
        self.pool_size = self.n + self.slack
        all_nodes = np.arange(len(world.nodes), dtype=int)
        count = max(1, int(np.ceil(self.scatter * len(all_nodes) / self.pool_size)))
        self.copysets = []
        for j in range(count):
            pool = self._build_copyset(all_nodes, 0x4350590000000000 + j)
            if len(pool) == self.pool_size:
                self.copysets.append(np.asarray(pool, dtype=int))
        if not self.copysets:
            raise ValueError("Не удалось построить ни одного copyset")
        self._members = [set(x.tolist()) for x in self.copysets]
        self.object_copyset = {}
        self.spill_count = 0
        self.pool_probes = 4
        self.name = f"Copysets, n={n}, scatter={self.scatter}, slack={self.slack}"

    def _build_copyset(self, candidates, key):
        return self._fast_spread(candidates, self.pool_size, (), key)

    @staticmethod
    def _mix64(v):
        mask = 0xFFFFFFFFFFFFFFFF
        x = int(v) & mask
        x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
        x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
        return x ^ (x >> 31)

    def _pool_order(self, obj):
        start = self._mix64(obj) % len(self.copysets)
        return [(start + i) % len(self.copysets) for i in range(len(self.copysets))]

    def _choose_copyset(self, cand, m, held, obj):
        held = set(held)
        assigned = self.object_copyset.get(obj)
        order = [assigned] if assigned is not None else self._pool_order(obj)[:self.pool_probes]
        cset = set(np.asarray(cand, dtype=int).tolist()) - held
        for j in order:
            subset = np.asarray(sorted(self._members[j] & cset), dtype=int)
            if len(subset) < m:
                continue
            # Every pool was built under the same spread constraints; any subset
            # of its members preserves them, so no second constraint search is needed.
            out = self._ranked_candidates(subset, obj)[:m].astype(int).tolist()
            if len(out) == m:
                self.object_copyset[obj] = j
                return out
        self.spill_count += 1
        return CrushLikePolicy._choose_spread(self, None, cand, m, held=held, obj=obj)

    def place(self, obj, t, view):
        return self._choose_copyset(view.candidates(), self.n, (), obj)

    def on_epoch(self, t, view):
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose_copyset(cand, len(idx), view.holders(obj), obj)
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class CrisisOraclePolicy(OraclePolicy):
    """Оракул кризисов (пункт 3.9): оракул, который с начала окна предвестника знает тип, кластер-цель
    и момент каждого предсказуемого кризиса. Пока кризис известен и не закончился, он поднимает тревогу,
    не кладёт новые фрагменты на узлы-цели, а до момента кризиса переносит с них фрагменты.
    О внезапных кризисах не знает, ложные тревоги отличает от кризисов. Верхняя граница для H8."""

    def __init__(self, world, n: int = 15, gamma: float = 1.0, fast: bool = False):
        super().__init__(world, n, gamma)
        self.name = f"Оракул кризисов, n={n}" + (", тревоги каждый подшаг" if fast else "")
        self.fast = fast              # тревоги и перенос с шагом alarm_step_hours (ledger, Ч2)
        cr = world.crises
        self.known = cr[cr["predictable"] & ~cr["false_alarm"]].reset_index(drop=True)
        self.targets = {e["id"]: e["targets"] for e in world.crisis.episodes}

    def _active(self, t):
        k = self.known
        return k[(k["w0"] <= t) & (k["end"] > t)]

    def _avoid(self, t) -> np.ndarray:
        a = self._active(t)
        if a.empty:
            return np.empty(0, int)
        return np.unique(np.concatenate([self.targets[i] for i in a["id"]]))

    def _cand(self, t, view):
        cand = view.candidates()
        avoid = self._avoid(t)
        return cand[~np.isin(cand, avoid)] if len(avoid) else cand

    def alarm(self, t, view):
        a = self._active(t)
        a = a[a["tau"] > t]
        return [dict(type=r.type, ctype=r.ctype, cluster=r.cluster, prob=1.0, horizon=r.tau - t + 1e-6)
                for r in a.itertuples()]

    def migrate(self, t, view):
        a = self._active(t)
        a = a[a["tau"] > t]
        if a.empty:
            return []
        cand = self._cand(t, view)
        out = []
        for node in np.unique(np.concatenate([self.targets[i] for i in a["id"]])):
            for obj, idx in view.on_node(int(node)):
                if idx in view.migrating(obj):
                    continue
                held = view.holders(obj) | {to for o, _, to in out if o == obj}
                to = self._choose_spread(self._g, cand, 1, held=held - {int(node)})
                if to:
                    out.append((obj, idx, to[0]))
        return out


class AgentCrisisPolicy(OraclePolicy):
    """Агент с прогнозом кризисов (шаг 5, Ч5): размещение как у оракула, тревоги — по счёту аномалий агента
    (forecast.Forecaster) только по данным сети. Пока тревога по группе узлов действует, новые фрагменты в группу
    не кладутся, а с её узлов фрагменты переносятся заранее — как у оракула кризисов, но без знания истины.

    Тревога поднимается, когда счёт группы выше порога своего типа, и держится hold суток после последнего
    превышения. Порог агент учит сам (П3): каждый всплеск счёта выше floor становится эпизодом; через
    горизонт разметки агент смотрит, случился ли в группе кризис (массовый простой дата-центра — для бури
    и каскада; не меньше двух узлов партии, ушедших в простой дольше суток, — для износа; уходы в регионе
    втрое выше фона — для массового ухода). Порог — максимум (верные - kappa * ложные) по эпизодам с пиком
    не ниже порога; пока верных эпизодов меньше min_pos — начальный порог c_init. Для сбоя ПО (эпизодов
    мало) порог постоянный — c_init.
    fast=True — тревоги и перенос каждый подшаг (ledger, alarm_step_hours); forecast=False — та же политика
    без прогноза (нижняя граница для H8). За один вызов — не больше max_moves переносов; фрагмент, которому
    некуда переехать из-за лимитов на кластеры, до конца тревог больше не пробуется. Тревога держится не меньше
    шага решений; группу больше max_share узлов в реестре (например, версию ПО, на которую обновилась почти вся
    сеть) агент отмечает тревогой, но не избегает; если без групп в тревоге кандидатов меньше 3n, размещение и
    перенос идут по всем кандидатам. fixed_thr — постоянные пороги (без обучения): показывает, что даёт сам прогноз.

    kappa="econ" — экономический критерий (Э1): эпизод весит не 1 и kappa, а деньгами по ценам резерва
    (reserve.Costs: c_tr за единицу трафика, W за объект). В начале всплеска агент снимает по своей карте
    размещения, сколько объектов держат в группе f фрагментов. Перенос стоит C = c_tr * F / k (F — фрагменты в
    группе) и в верном, и в ложном эпизоде; верный эпизод ещё и приносит B = c_tr * R + W * dL — избегнутый трафик
    восстановления и избегнутые потери при доле q узлов группы, выбитых кризисом (q агент оценивает по своей
    разметке). Порог — максимум суммы (y * B - C) по эпизодам с пиком не ниже порога; «не тревожить» (порог inf)
    тоже допустим."""

    GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
    HOLD = {"storm": 0.25, "cascade": 0.25, "software": 0.25, "wear": 7.0, "exodus": 7.0}   # сутки
    LABEL = {"storm": 1.0, "cascade": 1.0, "software": 1.0, "wear": 30.0, "exodus": 14.0}  # горизонт разметки
    LEARN = {"storm": "dc", "cascade": "dc", "wear": "wear", "exodus": "exodus", "software": None}

    def __init__(self, world, n: int = 13, gamma: float = 1.0, fast: bool = False, forecast: bool = True,
                 c_init: float = 10.0, floor: float = 2.0, kappa: float = 0.5, min_pos: int = 3,
                 max_moves: int = 2000, fixed_thr: dict | None = None, econ_W: float = 1e4,
                 confidence_q: float = 0.10):
        from .experience import Experience
        from .forecast import Forecaster
        super().__init__(world, n, gamma)
        self.fast, self.forecast = fast, forecast
        hours = world.cfg.alarm_step_hours if fast else 24
        self.fc = Forecaster(world, Experience(world), step_hours=hours)
        self.c_init, self.floor, self.kappa, self.min_pos = c_init, floor, kappa, min_pos
        if not 0.0 < confidence_q < 0.5:
            raise ValueError("confidence_q должен быть в интервале (0, 0.5)")
        self.confidence_q = float(confidence_q)  # нижняя односторонняя граница posterior precision
        self.thr = {k: c_init for k in ("dc", "wear", "exodus", "software")}
        self.learn = fixed_thr is None          # fixed_thr — постоянные пороги по группам, без обучения
        if fixed_thr:
            self.thr.update(fixed_thr)
        self.active = {}          # (тип, ctype, кластер) -> [до какого момента, узлы]
        self.runs = {}            # группа -> открытый всплеск: [начало, пик, последнее превышение floor, узлы, тип]
        self.pending = []         # закрытые всплески, ждущие разметки
        self.labels = {k: [] for k in ("dc", "wear", "exodus")}      # (пик, был ли кризис)
        self.thr_log = []
        self.calibration_log = []  # момент, тип, порог, n, positives, posterior mean, lower bound, utility
        self._cal_mark = {}
        self.max_moves = max_moves  # переносов за один вызов (пропускная способность сети)
        self.step = hours / 24.0    # шаг решений о тревогах, сутки: тревога держится не меньше шага
        self.max_share = 0.25       # группу больше этой доли узлов в реестре агент не избегает (только тревога)
        self.nomove = set()         # фрагменты, которым при текущих тревогах некуда переехать
        self.econ = kappa == "econ"
        if self.econ:
            from .reserve import Costs
            from .config import HOUR
            self.costs = Costs(W=econ_W)
            self.w_loss = world.cfg.T_rep_hours * HOUR + world.cfg.repair_hours * HOUR   # окно до конца восстановления
            self.q_hit = {k: [0.0, 0.0] for k in self.labels}     # выбито узлов и всего узлов в верных эпизодах
            self.econ_log = []                                    # (t, группа, порог, kappa_экв, B, C)
            self._coef_memo = {}
        self.name = ("Агент кризисов" if forecast else "Оракул без прогноза") + f", n={n}" + \
            (", шаг тревог 1 ч" if fast and hours < 24 else "") + (", экономический порог" if self.econ else "")

    # ---- разметка своих всплесков (П3)
    def _crisis_seen(self, lab, nodes, t0, t1):
        fc = self.fc
        d0, d1 = int(max(t0, 0)), int(min(t1, fc.T - 1))
        if lab == "dc":
            return bool(fc.mass[nodes, d0:d1 + 1].any())
        c0, c1 = int(t0 * fc.cpd), int(min(t1, fc.T) * fc.cpd)
        n_long = float(fc.longst[nodes, c0:c1].sum())
        if lab == "wear":
            return n_long >= 2
        cut = max(int(t0) - 30, 1)
        rate = (fc.cum["longst"][nodes, min(cut, fc.cum_day)].sum() + 0.5) / max(fc.cum["up"][nodes, min(cut, fc.cum_day)].sum() / fc.cpd, 1.0)
        return n_long >= max(3.0, 3.0 * rate * len(nodes) * (t1 - t0))

    def _hit_share(self, lab, nodes, t0, t1):
        """Сколько узлов группы кризис выбил надолго (по данным сети): для дата-центра — узлы массового простоя,
        для партии и региона — узлы, чей простой стал длиннее суток."""
        fc = self.fc
        if lab == "dc":
            d0, d1 = int(max(t0, 0)), int(min(t1, fc.T - 1))
            return int(fc.mass[nodes, d0:d1 + 1].any(1).sum())
        c0, c1 = int(t0 * fc.cpd), int(min(t1, fc.T) * fc.cpd)
        return int((fc.longst[nodes, c0:c1].sum(1) > 0).sum())

    def _snapshot(self, view, g):
        """Число объектов по числу f своих фрагментов в группе g (по карте размещения): {f: объектов}."""
        cnt = {}
        for node in g:
            for obj, _ in view.on_node(int(node)):
                cnt[obj] = cnt.get(obj, 0) + 1
        out = {}
        for f in cnt.values():
            out[f] = out.get(f, 0) + 1
        return out

    def _p_loss(self, m, p):
        """Вероятность потерять объект с m фрагментами за окно восстановления: больше m - k гибелей из m."""
        from scipy.stats import binom
        return float(binom.sf(m - self.k, m, p)) if m >= self.k else 1.0

    def _coef(self, lab, f):
        """Выгода b_f и цена c_f на один объект с f фрагментами в группе (Э1); кэш по (q, p) текущей оценки."""
        from scipy.stats import binom
        c, k, n, fc = self.costs, self.k, self.n, self.fc
        m, N = self.q_hit[lab]
        q = (m + 1.0) / (N + 2.0)
        day = min(fc.cum_day, fc.T)
        lam = (fc.cum["longst"][:, day].sum() + 0.5) / max(fc.cum["up"][:, day].sum() / fc.cpd, 1.0)
        p = 1.0 - np.exp(-lam * self.w_loss)              # фрагмент гибнет за окно восстановления (фон)
        key = (lab, q, p, f)
        if key not in self._coef_memo:
            base = self._p_loss(n, p)
            R = c.size * ((1.0 - (1.0 - q) ** f) + q * f / k)
            dL = sum(binom.pmf(r, f, q) * self._p_loss(n - r, p) for r in range(f + 1)) - base
            self._coef_memo[key] = (c.c_tr * R + c.W * dL, c.c_tr * c.size * f / k)
        return self._coef_memo[key]

    def _value(self, lab, snap):
        """Выгода верной тревоги B и цена переноса C для эпизода со снимком snap (Э1)."""
        B = C = 0.0
        for f, cnt in snap.items():
            b, cc = self._coef(lab, f)
            B += cnt * b; C += cnt * cc
        return B, C

    def _learn(self, t):
        keep = []
        for r in self.pending:
            start, peak, last, nodes, typ, lab = r[:6]
            due = last + self.LABEL[typ]
            if t < due:
                keep.append(r)
                continue
            y = self._crisis_seen(lab, nodes, start, due)
            if self.econ:
                if y:
                    self.q_hit[lab][0] += self._hit_share(lab, nodes, start, due)
                    self.q_hit[lab][1] += len(nodes)
                self.labels[lab].append((peak, y, r[6]))
            else:
                self.labels[lab].append((peak, y))
        self.pending = keep
        for lab, L in self.labels.items():
            if not self.learn or sum(v[1] for v in L) < self.min_pos:
                continue
            mark = (len(L), int(t) if self.econ else None)
            if self._cal_mark.get(lab) == mark:
                continue
            self._cal_mark[lab] = mark
            pk = np.array([v[0] for v in L]); y = np.array([v[1] for v in L], bool)
            from scipy.stats import beta as beta_dist
            if self.econ:
                self._coef_memo = {}
                BC = np.array([self._value(lab, v[2]) for v in L])
                grid = self.GRID + (np.inf,)
                gain, posterior = [], []
                for c0 in grid:
                    selected = pk >= c0
                    n_sel = int(selected.sum())
                    if not n_sel:
                        p_mean = p_low = 0.0
                        utility = 0.0
                    else:
                        pos = int(y[selected].sum())
                        a, b = pos + 0.5, n_sel - pos + 0.5  # Jeffreys Beta prior
                        p_mean = a / (a + b)
                        p_low = float(beta_dist.ppf(self.confidence_q, a, b))
                        utility = p_low * float(BC[selected, 0].sum()) - float(BC[selected, 1].sum())
                    gain.append(utility)
                    posterior.append((n_sel, int(y[selected].sum()) if n_sel else 0, p_mean, p_low))
                gain = np.asarray(gain)
                c = grid[int(np.flatnonzero(gain >= gain.max() - 1e-12)[-1])]   # при равенстве — больший порог
                Bm, Cm = float(BC[y, 0].mean()), float(BC[:, 1].mean())
                kap = Cm / (Bm - Cm) if Bm > Cm else np.inf
                n_sel, n_pos, p_mean, p_low = posterior[int(np.flatnonzero(gain >= gain.max() - 1e-12)[-1])]
                if not self.econ_log or self.econ_log[-1][1:3] != (lab, c) or t - self.econ_log[-1][0] >= 30:
                    self.econ_log.append((t, lab, c, kap, Bm, Cm))
            else:
                grid = self.GRID + (np.inf,)
                gain, posterior = [], []
                for c0 in grid:
                    selected = pk >= c0
                    n_sel = int(selected.sum())
                    if not n_sel:
                        p_mean = p_low = utility = 0.0
                        n_pos = 0
                    else:
                        n_pos = int(y[selected].sum())
                        a, b = n_pos + 0.5, n_sel - n_pos + 0.5  # Jeffreys Beta prior
                        p_mean = a / (a + b)
                        p_low = float(beta_dist.ppf(self.confidence_q, a, b))
                        utility = n_sel * (p_low - self.kappa * (1.0 - p_low))
                    gain.append(utility)
                    posterior.append((n_sel, n_pos, p_mean, p_low))
                gain = np.asarray(gain)
                best = int(np.flatnonzero(gain >= gain.max() - 1e-12)[-1])
                c = grid[best]
                n_sel, n_pos, p_mean, p_low = posterior[best]
            self.calibration_log.append((float(t), lab, float(c), int(n_sel), int(n_pos),
                                         float(p_mean), float(p_low), float(gain.max())))
            if c != self.thr[lab]:
                self.thr[lab] = c
                self.thr_log.append((t, lab, c, len(L), int(y.sum())))

    # ---- интерфейс политики
    def _avoid(self, t):
        nodes = [v[1] for v in self.active.values() if v[0] > t and v[2]]
        return np.unique(np.concatenate(nodes)) if nodes else np.empty(0, int)

    def _cand(self, t, view):
        cand = view.candidates()
        avoid = self._avoid(t)
        if not len(avoid):
            return cand
        left = cand[~np.isin(cand, avoid)]
        # без группы в тревоге узлов не хватает на разнесённое размещение — тревога не должна губить объекты
        return left if len(left) >= 3 * self.n else cand

    def alarm(self, t, view):
        if not self.forecast:
            return []
        self.fc.update(t, view)
        self._learn(t)
        for typ, ct, cl, S, O, E, g in self.fc.scores():
            lab = self.LEARN[typ] or "software"
            gk = (lab, ct, str(cl))
            run = self.runs.get(gk)
            if S > self.floor:
                if run is None:
                    self.runs[gk] = [t, S, t, g, typ, lab]
                    if self.econ and lab != "software":
                        self.runs[gk].append(self._snapshot(view, g))     # до переносов этого вызова
                else:
                    run[1] = max(run[1], S); run[2] = t; run[4] = typ
            elif run is not None and t - run[2] > self.HOLD[typ]:
                if lab != "software":
                    self.pending.append(tuple(run))
                del self.runs[gk]
            if S > self.thr[lab]:
                n_reg = max(int((view.join_times() <= t).sum()), 1)
                act = len(g) <= self.max_share * n_reg       # слишком большую группу не избегать
                self.active[(typ, ct, cl)] = [t + max(self.HOLD[typ], self.step), g, act]
        self.active = {k: v for k, v in self.active.items() if v[0] > t}
        return [dict(type=k[0], ctype=k[1], cluster=k[2], prob=1.0, horizon=v[0] - t) for k, v in self.active.items()]

    def migrate(self, t, view):
        if not self.active:
            self.nomove = set()
            return []
        cand = self._cand(t, view)
        out = []
        for node in self._avoid(t):
            for obj, idx in view.on_node(int(node)):
                if len(out) >= self.max_moves:
                    return out
                if (obj, idx) in self.nomove or idx in view.migrating(obj):
                    continue
                held = view.holders(obj) | {to for o, _, to in out if o == obj}
                to = self._choose_spread(self._g, cand, 1, held=held - {int(node)})
                if to:
                    out.append((obj, idx, to[0]))
                else:
                    self.nomove.add((obj, idx))      # места нет (лимиты на кластеры) — до конца тревог не пробовать
        return out


class ReservePolicy(OraclePolicy):
    """Политика резерва информации (шаг 3, пункт 3.11): восстановление — по порогу запаса.

    placement="oracle" (default) выбирает узлы по истинному риску; placement="agent" требует rates="agent"
    и ранжирует их по оценённой потере фрагмента плюс наблюдаемой частоте шоков доменов. placement="agent_capped"
    дополнительно перебирает несколько допустимых наборов и применяет лимит вероятности потери за окно уязвимости;
    если среди них нет допустимого набора, берёт наименее рискованный и считает нарушение. До первой подгонки
    риск-веса равны. Агент ограничивает концентрацию по региону, провайдеру, заявленному DC (если claimed=True)
    и публичных кошелька выплат и подсети вместо скрытого поля владельца.

    Восстановление объекта (всех признанных утраченными фрагментов разом) запускается, когда запас
    не выше порога R_min. Три варианта для H7:
    - mode="thiele" — TARIS-резерв: порог свой у каждого объекта, из уравнения Тиле по меняющимся во времени
      истинным интенсивностям его узлов (минимум приведённой стоимости при P_loss <= eps за горизонт H);
      запас — фактический, с поправками IBNR и RBNS; порог пересчитывается при смене набора узлов и раз в 30 суток;
    - mode="markov" — марковский конкурент: порог на объект при размещении по постоянной интенсивности
      (минимум трафика при P_loss <= eps), запас — по карте, без поправок;
    - mode="fixed" — эвристический порог R_fixed для всех объектов (R_fixed = n - k - 1 — немедленное восстановление).
    Если eps недостижим, берётся немедленное восстановление.
    Абляции TARIS-резерва (H7): corr=False — запас по карте, без IBNR и RBNS; time_varying=False — интенсивности
    на горизонте постоянны (берутся на момент расчёта порога, без старения узлов).
    """

    def __init__(self, world, n: int = 15, mode: str = "thiele", eps: float = 1e-3, H: int = 365,
                 R_fixed: int | None = None, gamma: float = 1.0, block: int = 30, corr: bool = True,
                 time_varying: bool = True, rates: str = "true", claimed: bool = True,
                 placement: str = "oracle", tariff_settings=None, contagion_strength: float = 0.0,
                 contagion_half_life_days: float = 1.0, contagion_window_days: float = 7.0,
                 contagion_min_events: int = 3, contagion_min_credibility: float = 0.5,
                 contagion_family_alpha: float = 0.05, placement_max_n: int | None = None,
                 placement_trials: int = 6, placement_cluster_opt: bool = False,
                 release_surplus: bool = False, release_min_credibility: float = 0.5,
                 release_cap_fraction: float = 1.0, fixedpoint_tariff: bool = False):
        from . import reserve as RS
        if placement not in ("oracle", "uniform", "agent", "agent_greedy", "agent_capped", "tariff"):
            raise ValueError("неизвестный placement; доступны oracle, uniform, agent, agent_greedy, agent_capped, tariff")
        if placement in ("agent", "agent_greedy", "agent_capped", "tariff") and rates != "agent":
            raise ValueError(f"placement='{placement}' требует rates='agent'")
        if placement_cluster_opt and placement != "agent_capped":
            raise ValueError("placement_cluster_opt доступен только для placement='agent_capped'")
        if release_surplus and (placement != "agent_capped" or rates != "agent"):
            raise ValueError("release_surplus доступен только для agent_capped с оценками агента")
        if not 0.0 <= release_min_credibility <= 1.0:
            raise ValueError("release_min_credibility должен быть в [0, 1]")
        if not 0.0 < release_cap_fraction <= 1.0:
            raise ValueError("release_cap_fraction должен быть в (0, 1]")
        if placement == "tariff":
            from .tariff import TariffSettings
            if not isinstance(tariff_settings, TariffSettings):
                raise ValueError("placement='tariff' требует явный tariff_settings типа TariffSettings")
        elif tariff_settings is not None:
            raise ValueError("tariff_settings задаётся только для placement='tariff'")
        if fixedpoint_tariff and placement != "tariff":
            raise ValueError("fixedpoint_tariff доступен только для placement='tariff'")
        super().__init__(world, n, gamma, oracle=placement == "oracle", claimed=claimed)
        self.world = world
        self.greedy_placement = placement == "agent_greedy"
        self.capped_placement = placement == "agent_capped"
        self.placement_cluster_opt = bool(placement_cluster_opt)
        self.release_surplus = bool(release_surplus)
        self.release_min_credibility = float(release_min_credibility)
        self.release_cap_fraction = float(release_cap_fraction)
        self.release_stats = {"checked": 0, "released": 0, "cap_blocked": 0,
                              "credibility_blocked": 0, "unhealthy_blocked": 0}
        self.placement_trials = int(placement_trials)
        if self.placement_trials < 1:
            raise ValueError("placement_trials должен быть положительным")
        self.placement_max_n = int(placement_max_n if placement_max_n is not None
                                   else (n + 5 if self.capped_placement else n))
        if self.placement_max_n < n:
            raise ValueError("placement_max_n не может быть меньше базового n")
        self.placement_risk = {"checked": 0, "feasible": 0, "violations": 0, "unscored": 0,
                               "extra_fragments": 0, "max_placed_n": 0,
                               "max_best_upper": 0.0, "max_dropped": 0.0}
        self.RS, self.mode, self.eps, self.H, self.block = RS, mode, eps, H, block
        self.corr, self.time_varying = corr and mode == "thiele", time_varying
        self.agent = rates == "agent"
        contagion_params = np.asarray([contagion_strength, contagion_half_life_days,
                                       contagion_window_days, contagion_min_credibility,
                                       contagion_family_alpha], dtype=float)
        if (not np.isfinite(contagion_params).all() or contagion_strength < 0
                or contagion_half_life_days <= 0 or contagion_window_days <= 0
                or not 0.0 <= contagion_min_credibility <= 1.0 or contagion_min_events < 1
                or not 0.0 < contagion_family_alpha < 1.0):
            raise ValueError("некорректные параметры передачи риска по графу")
        if contagion_strength > 0 and not self.agent:
            raise ValueError("передача риска по графу доступна только rates='agent'")
        self.contagion_strength = float(contagion_strength)
        self.contagion_half_life_days = float(contagion_half_life_days)
        self.contagion_window_days = float(contagion_window_days)
        self.contagion_min_events = int(contagion_min_events)
        self.contagion_min_credibility = float(contagion_min_credibility)
        self.contagion_family_alpha = float(contagion_family_alpha)
        self.contagion_graph = None
        self.contagion_W = None
        self.contagion_signal = np.zeros(len(world.nodes), dtype=float)
        self.contagion_edges = 0
        if self.contagion_strength > 0:
            from .contagion import ContagionGraph
            self.contagion_graph = ContagionGraph(world, tau=1.0, K=1.0, block=block)
        self.placement = placement
        self.tariff_settings = tariff_settings
        self.fixedpoint_tariff = bool(fixedpoint_tariff)
        self.tariff_snapshot = None
        self.tariff_values = None
        if self.agent:
            # оценки агента (шаг 5, пункт 5.1): обучение раз в block эпох; до первого обучения — немедленное восстановление
            from .experience import Experience
            from .risk import RiskModel, EstimatedRates
            self.exp = Experience(world, block)
            self.model = RiskModel(self.exp, claimed=claimed)
            self.rates = EstimatedRates(world, self.model)
            self.fits = []
        else:
            self.rates = RS.TrueRates(world)
        self.R_fixed = n - self.k - 1 if R_fixed is None else R_fixed
        self.costs = RS.Costs()
        self.thr = {}          # объект -> (ключ набора узлов, порог)
        self.memo = {}         # округлённые интенсивности -> порог
        self.name = {"thiele": f"Резерв (Тиле), n={n}", "markov": f"Марковский порог, n={n}",
                     "fixed": f"Порог R_min={self.R_fixed}, n={n}"}[mode]
        if mode == "thiele" and not corr:
            self.name += ", без IBNR/RBNS"
        if mode == "thiele" and not time_varying:
            self.name += ", постоянные интенсивности"
        if self.agent:
            self.name += ", оценки агента" + ("" if claimed else " (только проверяемые признаки)")
        if placement in ("agent", "agent_greedy", "agent_capped"):
            self.name += ", размещение по оценке агента"
            if self.greedy_placement:
                self.name += " (жадное)"
            if self.capped_placement:
                self.name += " (с лимитом риска)"
        elif placement == "tariff":
            self.name += ", размещение по тарифу"
            if self.fixedpoint_tariff:
                self.name += " (fixed point денежных решений)"
        elif placement == "uniform":
            self.name += ", равномерное размещение"

    def _threshold(self, obj, nodes, t, view, n_obj: int | None = None):
        if self.mode == "fixed":
            return self.R_fixed
        if self.agent and not self.model.fitted:
            return self.n - self.k - 1
        T_end = view.L.T
        n = self.n if n_obj is None else int(n_obj)
        key = (n, frozenset(nodes)) if self.mode == "markov" else (n, frozenset(nodes), int(t // self.block))
        old = self.thr.get(obj)
        if old is not None and (self.mode == "markov" or old[0] == key):
            return old[1]
        RS, k = self.RS, self.k
        Hrem = int(min(self.H, max(T_end - t, self.block)))
        nb = max(1, int(np.ceil(Hrem / self.block)))
        nodes = np.asarray(sorted(nodes), int)
        cl = self.rates.shock_clusters(nodes, t)
        shocks = [(eta, int(m.sum())) for _, _, eta, m in cl]
        mu = RS.object_mu(self.rates, nodes, t, shocks)
        if self.mode == "markov" or not self.time_varying:
            lam = np.full(nb, self.rates.fragment_loss(nodes, t).mean())
        else:
            lam = np.array([self.rates.fragment_loss(nodes, t + (b + 0.5) * self.block).mean() for b in range(nb)])
        # кэш порогов: интенсивности округляются ступенями 5% (логарифмическая сетка), задержка — до 0,05 суток
        q = lambda x: float(np.exp(np.round(np.log(max(x, 1e-12)) * 20) / 20))
        lam = np.array([q(x) for x in lam])
        shocks = sorted((q(e), m) for e, m in shocks)
        mu = 1.0 / (round(20.0 / mu) / 20.0)
        mkey = (self.mode, tuple(lam), tuple(shocks), mu)       # порог зависит только от этих величин
        if mkey not in self.memo:
            # P_loss убывает с ростом порога (восстановление раньше): допустимые пороги — от некоторого R и выше.
            # Перебор сверху вниз до первого недопустимого проверяет все допустимые — выбор тот же, что при полном.
            best, R_best = None, n - k - 1
            for R_min in range(n - k - 1, -1, -1):
                m = RS.object_metrics(n, k, lam, shocks, R_min, mu, self.costs, dt=self.block)
                if m["P_loss"] > self.eps:
                    break
                v = m["cost"] if self.mode == "thiele" else m["traffic"]
                if best is None or v < best:
                    best, R_best = v, R_min
            self.memo[mkey] = R_best
        R = self.memo[mkey]
        self.thr[obj] = (key, R)
        return R

    def _agent_placement_weights(self, t):
        """Обратный ожидаемый риск фрагмента: индивидуальные потери плюс наблюдаемая частота шоков домена.

        Общие шоки оцениваются только по Experience/RiskModel. Каждый узел получает частоту
        тех оценённых шоков, в домен которых он входит; скрытый владелец сюда не попадает.
        """
        nodes = np.arange(self.model.exp.N)
        hazard = self.rates.fragment_loss(nodes, t).astype(float, copy=True)
        for _, _, eta, mask in self.rates.shock_clusters(nodes, t):
            hazard[mask] += eta
        if self.contagion_strength > 0:
            hazard *= 1.0 + self.contagion_strength * self.contagion_signal[nodes]
        self.w = (1.0 / np.maximum(hazard, 1e-9)) ** self.gamma

    def place(self, obj, t, view):
        if not self.capped_placement:
            return super().place(obj, t, view)
        if not self.model.fitted:
            self.placement_risk["unscored"] += 1
            return super().place(obj, t, view)

        from .reserve import placement_risk_metrics, window_probs, p_loss_window_shocks, window_tau
        cand = self._cand(t, view)
        if len(cand) < self.n:
            return self._choose_spread(self._g, cand, self.n)
        tau = window_tau(self.world.cfg)
        eps_tau = self.eps * tau / max(float(self.H), tau)
        candidates = []
        feasible = []
        for n_try in range(self.n, self.placement_max_n + 1):
            for _ in range(self.placement_trials):
                nodes = self._choose_spread(self._g, cand, n_try)
                if len(nodes) != n_try:
                    continue
                p_upper, dropped, cost = placement_risk_metrics(self.rates, nodes, t, tau, self.k)
                row = (p_upper, cost, nodes, dropped, n_try)
                candidates.append(row)
                if p_upper <= eps_tau:
                    feasible.append(row)
            if feasible:
                break
        if not candidates:
            return self._choose_spread(self._g, cand, self.n)

        if self.placement_cluster_opt:
            # Один локальный swap для выбранного набора: заменяем самый рискованный узел
            # на первого допустимого низкорискового кандидата и сохраняем spread/cap.
            base = min(feasible, key=lambda x: (x[4], x[1], x[0])) if feasible else min(candidates, key=lambda x: (x[0], x[4], x[1]))
            chosen_nodes = list(base[2])
            all_nodes = np.arange(self.model.exp.N)
            q_all, masks_all, h = window_probs(self.rates, all_nodes, t, tau)
            hazard = self.rates.fragment_loss(all_nodes, t).astype(float, copy=True)
            clusters = self.rates.shock_clusters(all_nodes, t)
            for _, _, eta, mask in clusters:
                hazard[mask] += eta
            pool = cand[np.argsort(self.w[cand], kind="stable")[-min(8, len(cand)):]]
            r0 = min(self.n - self.k, self.k - 1)

            def valid(ns):
                dc = np.bincount(self.dc[ns], minlength=self.n_dc)
                prov = np.bincount(self.prov[ns], minlength=self.n_prov)
                reg = np.bincount(self.reg[ns], minlength=self.n_reg)
                own = np.bincount(self.own[ns], minlength=self.n_own)
                if own.max(initial=0) > r0:
                    return False
                if self.subnet_domain is not None and np.bincount(self.subnet_domain[ns], minlength=self.n_subnet).max(initial=0) > r0:
                    return False
                return ((dc.max(initial=0) <= 1 and prov.max(initial=0) <= r0 and reg.max(initial=0) <= r0 + self.n)
                        or (dc.max(initial=0) <= self.n and prov.max(initial=0) <= r0 and reg.max(initial=0) <= r0 + self.n)
                        or (dc.max(initial=0) <= self.n and prov.max(initial=0) <= r0 + self.n and reg.max(initial=0) <= r0 + self.n))

            pos = int(np.argmax(hazard[np.asarray(chosen_nodes)]))
            for new in pool:
                new = int(new)
                if new in chosen_nodes:
                    continue
                trial = chosen_nodes.copy(); trial[pos] = new
                if not valid(trial):
                    continue
                q = q_all[trial]
                masks = [m[trial] for m in masks_all]
                p, dropped = p_loss_window_shocks(q, self.k, masks, h, max_shocks=2)
                p = min(1.0, p + dropped)
                cost = float(hazard[trial].sum())
                base_ok = base[0] <= eps_tau
                if ((base_ok and p <= eps_tau and cost < base[1]) or (not base_ok and p < base[0])):
                    optimized = (p, cost, trial, dropped, len(trial))
                    candidates.append(optimized)
                    if p <= eps_tau:
                        feasible.append(optimized)
                break

        self.placement_risk["checked"] += 1
        best_upper = min(x[0] for x in candidates)
        self.placement_risk["max_best_upper"] = max(self.placement_risk["max_best_upper"], best_upper)
        self.placement_risk["max_dropped"] = max(self.placement_risk["max_dropped"], max(x[3] for x in candidates))
        if feasible:
            self.placement_risk["feasible"] += 1
            chosen = min(feasible, key=lambda x: (x[4], x[1], x[0]))
            self.placement_risk["extra_fragments"] += chosen[4] - self.n
            self.placement_risk["max_placed_n"] = max(self.placement_risk["max_placed_n"], chosen[4])
            return chosen[2]
        # Если даже при максимальной избыточности cap недостижим, размещаем лучший найденный набор
        # и записываем нарушение цели, как предусмотрено спецификацией.
        self.placement_risk["violations"] += 1
        chosen = min(candidates, key=lambda x: (x[0], x[4], x[1]))
        self.placement_risk["extra_fragments"] += chosen[4] - self.n
        self.placement_risk["max_placed_n"] = max(self.placement_risk["max_placed_n"], chosen[4])
        return chosen[2]

    def release(self, t, view):
        """Снимает максимум один лишний фрагмент с объекта, если текущая оценка cap это допускает.

        Требуется обученная модель, доверие не ниже порога у всех оставшихся узлов, отсутствие
        простоя/потерь/переносов и успешное прохождение cap после удаления. Базовые n фрагментов
        не затрагиваются; функция выключена по умолчанию.
        """
        if (not self.release_surplus or not self.model.fitted
                or t <= 0 or int(t) % self.block != 0):
            return []
        from .reserve import placement_loss_probability, window_tau

        tau = window_tau(self.world.cfg)
        eps_tau = self.eps * tau / max(float(self.H), tau)
        down = view.down_since()
        reqs = []
        for obj in range(len(view.L.obj_frags)):
            if view.L.obj_n[obj] <= self.n or view.L.obj_lost_t[obj] != np.inf:
                continue
            if view.L.missing[obj] or view.L.pending[obj] or view.migrating(obj):
                continue
            mapping = view.placement(obj)
            if len(mapping) <= self.n:
                continue
            frag_idx = sorted(mapping)
            nodes = np.asarray([mapping[i] for i in frag_idx], int)
            if np.isfinite(down[nodes]).any():
                self.release_stats["unhealthy_blocked"] += 1
                continue
            self.release_stats["checked"] += 1
            risk = self.rates.fragment_loss(nodes, t).astype(float, copy=True)
            for _, _, eta, mask in self.rates.shock_clusters(nodes, t):
                risk[mask] += eta
            remove_pos = int(np.argmax(risk))
            keep = np.delete(nodes, remove_pos)
            if float(np.min(self.model.Z[keep])) < self.release_min_credibility:
                self.release_stats["credibility_blocked"] += 1
                continue
            p_upper, _ = placement_loss_probability(self.rates, keep, t, tau, self.k)
            if p_upper > eps_tau * self.release_cap_fraction:
                self.release_stats["cap_blocked"] += 1
                continue
            reqs.append((obj, frag_idx[remove_pos]))
        self.release_stats["released"] += len(reqs)
        return reqs

    def _tariff_placement_weights(self, t, view):
        """Ранжирование по стоимости хранения и чистой премии; тариф выше базы исключает новые данные."""
        from .graph import LinkGraph
        from .tariff import (cluster_volume_share, gross_premium, node_financial_terms,
                             pure_premium)
        if self.tariff_snapshot is None or int(t) % self.block == 0:
            self.tariff_snapshot = LinkGraph(self.world, view.L.ch).snapshot(t)
        volumes = view.stored_volume()
        snapshot = self.tariff_snapshot
        concentration = cluster_volume_share(volumes, (snapshot.clusters, snapshot.clusters_infra))
        nodes = view.registered()
        suspicion = snapshot.suspicion[nodes].copy()
        # Узлы, зарегистрированные после последнего месячного снимка, ещё не имеют
        # графовой оценки; до обновления графа их связь считается ненаблюдаемой.
        suspicion[~np.isfinite(suspicion)] = 0.0
        s = self.tariff_settings
        pure = pure_premium(self.model, nodes, t, s.basis)
        if not np.isfinite(pure["total_pure_premium"]).all():
            raise ValueError("тарифное размещение пока не оценивает тяжесть: нет наблюдённых частичных случаев")
        risk_adjusted = pure["total_pure_premium"] * (1.0 + self.contagion_strength * self.contagion_signal[nodes])
        if self.fixedpoint_tariff:
            from .fixedpoint import (SCALE, decode, encode, financial_terms_q,
                                     gross_premium_q)
            pure_q = np.fromiter((encode(x) for x in risk_adjusted), dtype=np.int64, count=len(nodes))
            conc_q = np.fromiter((encode(x) for x in concentration[nodes]), dtype=np.int64, count=len(nodes))
            susp_q = np.fromiter((encode(x) for x in suspicion), dtype=np.int64, count=len(nodes))
            theta_q, eta_q, zeta_q = encode(s.theta), encode(s.eta), encode(s.zeta)
            gross_q = np.fromiter((gross_premium_q(int(p), int(c), int(ss), theta_q, eta_q, zeta_q)
                                   for p, c, ss in zip(pure_q, conc_q, susp_q)),
                                  dtype=np.int64, count=len(nodes))
            gross_values = gross_q.astype(float) / SCALE
            reward_q, eligible, collateral_q = [], [], []
            reward0_q, psi_q, g_q = encode(s.reward0), encode(s.psi), encode(s.basis.g)
            for gval, volume, trust in zip(gross_q, volumes[nodes], self.model.Z[nodes]):
                reward, ok, collateral = financial_terms_q(int(gval), encode(volume), encode(trust),
                                                            g_q, reward0_q, psi_q)
                reward_q.append(reward); eligible.append(ok); collateral_q.append(collateral)
            gross = {"gross_premium": gross_values}
            terms = {"reward_per_volume": np.asarray(reward_q, float) / SCALE,
                     "new_data_eligible": np.asarray(eligible, bool),
                     "collateral": np.asarray(collateral_q, float) / SCALE}
            cost = (reward0_q + pure_q).astype(float) / SCALE
        else:
            gross = gross_premium(risk_adjusted, concentration[nodes], suspicion,
                                  theta=s.theta, eta=s.eta, zeta=s.zeta)
            terms = node_financial_terms(gross["gross_premium"], volumes[nodes], self.model.Z[nodes], s.basis,
                                         reward0=s.reward0, psi=s.psi)
            cost = s.reward0 + risk_adjusted
        weights = np.zeros_like(self.w)
        weights[nodes] = np.where(terms["new_data_eligible"], 1.0 / np.maximum(cost, 1e-12), 0.0)
        self.w = weights
        self.tariff_values = {"t": float(t), "nodes": nodes.copy(), "pure": pure,
                              "risk_adjusted_pure": risk_adjusted,
                              "gross": gross, "financial": terms, "concentration": concentration[nodes],
                              "suspicion": suspicion, "contagion_signal": self.contagion_signal[nodes].copy(),
                              "fixedpoint": self.fixedpoint_tariff}

    def on_epoch(self, t, view):
        self._g = self._rng(view)
        if self.agent:
            self.exp.update(t, view)
            if self.contagion_graph is not None:
                self.contagion_graph.update(t, view)
                if self.contagion_W is None or (t > 0 and int(t) % self.block == 0):
                    snap = self.contagion_graph.snapshot(t)
                    self.contagion_W = self.contagion_graph.propagation_weights(
                        snap, self.contagion_min_events, self.contagion_min_credibility,
                        self.contagion_family_alpha)
                    self.contagion_edges = int(np.count_nonzero(self.contagion_W))
                self.contagion_signal = self.contagion_graph.decayed_signal(
                    self.contagion_W, t, self.contagion_half_life_days, self.contagion_window_days)
            if t > 0 and int(t) % self.block == 0:
                self.model.fit(t)
                if self.model.fitted:
                    self.model.refresh(t)
                    self.fits.append((t, self.model.a_hat, self.model.pearson, self.model.churn_share))
            if self.placement in ("agent", "agent_greedy", "agent_capped", "tariff"):
                if self.model.fitted:
                    # Актуализируем микропрерывания по доступным наблюдениям; до обучения веса равны.
                    self.model.refresh(t)
                    if self.placement in ("agent", "agent_greedy", "agent_capped"):
                        self._agent_placement_weights(t)
                    else:
                        self._tariff_placement_weights(t, view)
                else:
                    self.w.fill(1.0)
        cand = self._cand(t, view)
        reqs = []
        need = view.needs_repair()
        if self.corr and need and (not self.agent or self.model.fitted):
            last_ok, down = view.last_audit_ok(), view.down_since()
        for obj in need:
            holders = view.holders(obj)
            if len(holders) < self.k:
                continue
            n_obj = int(view.L.obj_n[obj])
            thr = self._threshold(obj, holders, t, view, n_obj=n_obj)
            if self.corr and (not self.agent or self.model.fitted):
                R_eff, _ = self.RS.effective_reserve(self.rates, view, obj, t, self.k, last_ok, down)
            else:
                R_eff = len(holders) - self.k
            if R_eff > thr:
                continue
            idx = sorted(view.missing(obj))
            nodes = self._choose_spread(self._g, cand, len(idx), held=holders)
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs
