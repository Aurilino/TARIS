"""Р3.2: ограничения объекта для конечного ансамбля допустимых моделей.

Это устойчивость к заданному ансамблю, не доверительная гарантия вне него.
"""
from __future__ import annotations

from fractions import Fraction
import random

from exact_model import ExactModel, positive_integer
from learning import CONTEXTS, LAYERS, fit, group_distribution, rational_rows


class RiskEnvelope:
    def __init__(self, layers):
        self.layers = tuple(layers)
        self._own_cache, self._pair_cache, self._risk_cache = {}, {}, {}

    def own(self, node):
        if node not in self._own_cache:
            self._own_cache[node] = max(layer.own(node) for layer in self.layers)
        return self._own_cache[node]

    def pair(self, i, j):
        key = tuple(sorted((i, j)))
        if key not in self._pair_cache:
            self._pair_cache[key] = max(layer.pair(i, j) for layer in self.layers)
        return self._pair_cache[key]

    def risk(self, nodes, k):
        key = tuple(sorted(nodes)), k
        if key not in self._risk_cache:
            self._risk_cache[key] = max(layer.risk(key[0], k) for layer in self.layers)
        return self._risk_cache[key]


class RobustModel(ExactModel):
    """Интерфейс поиска сохранён; риск объекта — максимум по целым моделям."""

    def __init__(self, models):
        self.models = tuple(models)
        if not self.models:
            raise ValueError("Нужна хотя бы одна модель")
        self.nodes = tuple(sorted(self.models[0].nodes))
        if any(set(model.nodes) != set(self.nodes) for model in self.models):
            raise ValueError("Набор нод всех моделей должен совпадать")
        self.availability = RiskEnvelope(model.availability for model in self.models)
        self.loss = RiskEnvelope(model.loss for model in self.models)


def bootstrap_weights(store, cutoff, *, seed, block_size=8):
    positive_integer(block_size, "block_size")
    times = sorted({row.time for row in store.known(cutoff)})
    blocks = {}
    for t in times:
        blocks.setdefault(t // block_size, []).append(t)
    if not blocks:
        return {}
    keys = sorted(blocks)
    rng = random.Random(seed)
    weights = {t: 0 for t in times}
    for _ in keys:
        for t in blocks[rng.choice(keys)]:
            weights[t] += 1
    return weights


def stress_prior(specs, multiplier=2):
    if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)) or not 1 <= multiplier <= 10:
        raise ValueError("Множитель стресса должен быть конечным числом от 1 до 10")
    nodes = tuple(spec.name for spec in specs)
    layers = []
    for layer in LAYERS:
        contextual = []
        for context in CONTEXTS:
            own = 0.005 if layer == "loss" else 0.06 if context == "day" else 0.18
            shock = 0.008 if layer == "loss" else 0.03 if context == "day" else 0.07
            states, probabilities = group_distribution(specs, layer, context,
                individual=min(0.95, own * multiplier), group_risk=min(0.95, shock * multiplier))
            contextual.append(probabilities)
        mixture = [(a + b) / 2 for a, b in zip(*contextual)]
        layers.append(rational_rows(states, mixture))
    return ExactModel(nodes, *layers)


def build_ensembles(store, registry, cutoff, *, strength, seed=0, replicas=8, block_size=8):
    positive_integer(replicas, "replicas")
    point = fit(store, registry, cutoff, strength=strength)
    point_model = point.exact_model()
    replicas_models = []
    resamples = []
    for replica in range(replicas):
        weights = bootstrap_weights(store, cutoff, seed=seed + replica, block_size=block_size)
        fitted = fit(store, registry, cutoff, strength=strength, window_weights=weights)
        replicas_models.append(fitted.exact_model())
        resamples.append({"seed": seed + replica, "weights": weights})
    prior = fit(store, registry, cutoff, method="prior").exact_model()
    stressed = stress_prior(point.specs, 2)
    return {
        "point": RobustModel([point_model]),
        "bootstrap": RobustModel([point_model, *replicas_models]),
        "structural": RobustModel([point_model, prior, stressed]),
        "combined": RobustModel([point_model, *replicas_models, prior, stressed]),
    }, {"replicas": replicas, "block_size": block_size, "stress_multiplier": 2,
        "resamples": resamples, "confidence_level": None}
