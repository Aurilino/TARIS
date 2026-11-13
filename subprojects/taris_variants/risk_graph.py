"""Полный граф из конечной смеси наблюдений и публичной базовой модели.

Не перечисляет 2**N состояний и не читает истинные параметры мира.
"""
from collections import deque
import numpy as np


class ObservedRiskGraph:
    def __init__(self, registry, *, window_days=30, prior_hours=48, samples=128, seed=17001):
        if window_days < 1 or prior_hours <= 0 or samples < 1:
            raise ValueError("Некорректные параметры графа")
        self.registry = registry
        self.N = registry.N
        self.window_days, self.prior_hours, self.samples, self.seed = window_days, prior_hours, samples, seed
        self.rows = deque()
        self.last_end = 0.0
        self.fitted_at = None
        self.own = np.full(self.N, np.nan)
        self.pairs = np.full((self.N, self.N), np.nan)
        self.states = np.empty((0, self.N), bool)
        self.weights = np.empty(0)
        self.updates = 0

    def update(self, t, view):
        step = view.step
        if step is not None:
            a = float(step.get("t0", step.get("epoch", 0)))
            b = float(step.get("t1", a + 1))
            if b > t + 1e-9:
                raise ValueError("Графу переданы будущие наблюдения")
            if b > self.last_end + 1e-9:
                a = max(a, self.last_end)
                intervals = [(int(r.node), float(r.start), min(float(r.end), b))
                             for r in step["outages_ended"].itertuples()]
                intervals += [(int(r.node), float(r.start), b)
                              for r in step["outages_ongoing"].itertuples()]
                joins = view.join_times()
                for hour in range(int(round(a * 24)), int(round(b * 24))):
                    sample = (hour + 1) / 24 - 1e-10
                    down = np.zeros(self.N, bool)
                    for node, start, end in intervals:
                        if start <= sample < end:
                            down[node] = True
                    self.rows.append(((hour + 1) / 24, down, joins <= sample))
                self.last_end = b
        while self.rows and self.rows[0][0] <= t - self.window_days:
            self.rows.popleft()
        registered = np.zeros(self.N, bool)
        registered[view.registered()] = True
        # Свой поток политики; не использует и не сдвигает rng.stream мира.
        prior = np.zeros((self.samples, self.N), bool)
        for node in np.flatnonzero(registered):
            rng = np.random.default_rng([self.seed, 0, int(node)])
            prior[:, node] = rng.random(self.samples) < 0.02
        for dc in np.unique(self.registry.dc[registered]):
            if dc < 0:
                continue
            members = registered & (self.registry.dc == dc)
            shocks = np.random.default_rng([self.seed, 1, int(dc)]).random(self.samples) < 0.01
            prior |= shocks[:, None] & members[None, :]
        # Положительная базовая масса одиночного отказа даже без попадания в случайную выборку.
        prior = np.vstack([prior, np.eye(self.N, dtype=bool)[registered]])
        states, weights = [prior], [np.full(len(prior), self.prior_hours / len(prior))]
        for observed_at, down, known in self.rows:
            # У неизвестных тогда нод отсутствует опыт, а не нулевой риск.
            completed = np.where(known, down, prior[int(round(observed_at * 24)) % len(prior)])
            states.append(completed[None, :])
            weights.append(np.ones(1))
        self.states = np.vstack(states)
        self.weights = np.concatenate(weights)
        self.weights /= self.weights.sum()
        numeric = self.states.astype(float)
        self.own = self.weights @ numeric
        self.pairs = numeric.T @ (numeric * self.weights[:, None])
        self.own[~registered] = np.nan
        self.pairs[~registered, :] = np.nan
        self.pairs[:, ~registered] = np.nan
        self.fitted_at = float(t)
        self.updates += 1

    def score(self, nodes):
        nodes = np.asarray(sorted(nodes), int)
        if len(nodes) < 2:
            return 0.0
        matrix = self.pairs[np.ix_(nodes, nodes)]
        return float(np.triu(matrix, 1).sum())

    def object_unavailability(self, nodes, k):
        nodes = np.asarray(sorted(nodes), int)
        if len(nodes) < k:
            return 1.0
        failures = self.states[:, nodes].sum(axis=1)
        return float(self.weights[failures > len(nodes) - k].sum())
