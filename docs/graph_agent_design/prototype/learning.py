"""Р3: обучение согласованного распределения по частичным наблюдениям.

Конечная модель малой сети; никаких импортов истины или taris_sim.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import math

from exact_model import ExactModel


LAYERS = ("availability", "loss")
CONTEXTS = ("day", "night")


@dataclass(frozen=True)
class NodeSpec:
    name: str
    group: str                  # публичная/заявленная площадка, не скрытая истина
    joined_at: int


@dataclass(frozen=True)
class Observation:
    time: int                   # начало наблюдаемого окна
    observed_at: int            # когда результат стал известен
    layer: str
    context: str
    values: tuple[tuple[str, bool | None], ...]


class ObservationStore:
    def __init__(self):
        self._rows = []
        self._keys = set()

    def add(self, row: Observation):
        if (isinstance(row.time, bool) or not isinstance(row.time, int) or row.time < 0
                or isinstance(row.observed_at, bool) or not isinstance(row.observed_at, int)
                or row.observed_at <= row.time):
            raise ValueError("Нужны целые времена: 0 <= time < observed_at")
        if row.layer not in LAYERS or row.context not in CONTEXTS:
            raise ValueError("Неизвестный слой или контекст")
        names = [name for name, _ in row.values]
        if len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Некорректные идентификаторы наблюдений")
        if any(value is not None and type(value) is not bool for _, value in row.values):
            raise ValueError("Наблюдение должно быть True, False либо None")
        key = (row.time, row.layer)
        if key in self._keys:
            raise ValueError("Один слой одного окна нельзя дважды считать новым опытом")
        self._keys.add(key)
        self._rows.append(row)

    def known(self, cutoff):
        return tuple(row for row in self._rows if row.observed_at <= cutoff)


def state_space(nodes):
    return tuple(frozenset(node for node, bit in zip(nodes, bits) if bit)
                 for bits in product((False, True), repeat=len(nodes)))


def group_distribution(specs, layer, context, *, individual=None, group_risk=None):
    """Положительная базовая модель: независимые группы, общий шок + свой отказ."""
    if individual is None:
        individual = 0.06 if context == "day" else 0.18
        if layer == "loss":
            individual = 0.005
    if group_risk is None:
        group_risk = 0.03 if context == "day" else 0.07
        if layer == "loss":
            group_risk = 0.008
    nodes = tuple(spec.name for spec in specs)
    groups = {}
    for spec in specs:
        groups.setdefault(spec.group, []).append(spec.name)
    probabilities = []
    states = state_space(nodes)
    for state in states:
        p = 1.0
        for members in groups.values():
            own_pattern = math.prod(individual if node in state else 1 - individual for node in members)
            all_affected = all(node in state for node in members)
            p *= (1 - group_risk) * own_pattern + (group_risk if all_affected else 0)
        probabilities.append(p)
    total = sum(probabilities)
    return states, tuple(p / total for p in probabilities)


def observed_pattern(row, specs):
    joined = {spec.name: spec.joined_at for spec in specs}
    return tuple(sorted((name, value) for name, value in row.values
                        if name in joined and joined[name] <= row.time and value is not None))


def compatible(state, pattern):
    return all((name in state) == value for name, value in pattern)


def learn_joint(states, prior, patterns, strength, iterations=30):
    """EM по частичным маскам с положительными псевдосчётчиками общей модели."""
    if not patterns:
        return prior
    counts = Counter(patterns)
    masks = [(tuple(i for i, state in enumerate(states) if compatible(state, pattern)), count)
             for pattern, count in sorted(counts.items())]
    weights = list(prior)
    denominator = strength + len(patterns)
    for _ in range(iterations):
        expected = [strength * p for p in prior]
        for indices, count in masks:
            mass = sum(weights[i] for i in indices)
            if mass <= 0:
                raise ValueError("Модель не поддерживает наблюдаемое состояние")
            for i in indices:
                expected[i] += count * weights[i] / mass
        updated = [count / denominator for count in expected]
        difference = max(abs(a - b) for a, b in zip(weights, updated))
        weights = updated
        if difference < 1e-10:
            break
    return tuple(weights)


def learn_independent(nodes, states, prior, patterns, strength):
    marginals = {}
    for node in nodes:
        baseline = sum(p for state, p in zip(states, prior) if node in state)
        labels = [value for pattern in patterns for name, value in pattern if name == node]
        marginals[node] = (strength * baseline + sum(labels)) / (strength + len(labels))
    return tuple(math.prod(marginals[node] if node in state else 1 - marginals[node]
                           for node in nodes) for state in states)


def rational_rows(states, probabilities):
    # Согласованное округление всей модели, а не каждого ребра отдельно.
    weights = [max(1, round(p * 10**12)) for p in probabilities]
    total = sum(weights)
    return [{"probability": str(Fraction(weight, total)), "affected": sorted(state)}
            for state, weight in zip(states, weights)]


@dataclass
class LearnedModel:
    nodes: tuple[str, ...]
    specs: tuple[NodeSpec, ...]
    cutoff: int
    method: str
    strength: float
    states: tuple[frozenset[str], ...]
    probabilities: dict
    sample_counts: dict
    label_counts: dict

    def pattern_probability(self, row):
        pattern = observed_pattern(row, self.specs)
        if not pattern:
            return None
        probabilities = self.probabilities[row.layer, row.context]
        return sum(p for state, p in zip(self.states, probabilities) if compatible(state, pattern))

    def exact_model(self, context_weights=None):
        weights = {"day": Fraction(1, 2), "night": Fraction(1, 2)} if context_weights is None else {
            key: Fraction(str(value)) for key, value in context_weights.items()}
        if set(weights) != set(CONTEXTS) or any(v < 0 for v in weights.values()) or sum(weights.values()) != 1:
            raise ValueError("Нужны веса day и night с суммой 1")
        layers = []
        for layer in LAYERS:
            mixed = [sum(float(weights[context]) * self.probabilities[layer, context][i]
                         for context in CONTEXTS) for i in range(len(self.states))]
            layers.append(rational_rows(self.states, mixed))
        return ExactModel(self.nodes, *layers)


def fit(store, registry, cutoff, *, method="joint", strength=10, iterations=30, window_weights=None):
    if type(cutoff) is not int or cutoff < 0:
        raise ValueError("Момент обучения должен быть неотрицательным целым")
    if method not in ("prior", "independent", "unsmoothed", "joint"):
        raise ValueError("Неизвестный метод обучения")
    if not math.isfinite(strength) or strength <= 0:
        raise ValueError("Положительная сила базового опыта обязательна")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("Число итераций должно быть положительным целым")
    registry = tuple(registry)
    if (len({spec.name for spec in registry}) != len(registry)
            or any(not isinstance(spec.name, str) or not spec.name
                   or not isinstance(spec.group, str) or not spec.group
                   or type(spec.joined_at) is not int or spec.joined_at < 0
                   for spec in registry)):
        raise ValueError("Некорректный реестр")
    specs = tuple(sorted((spec for spec in registry if spec.joined_at <= cutoff), key=lambda spec: spec.name))
    if not 1 <= len(specs) <= 10:
        raise ValueError("Учебная конечная модель поддерживает 1–10 известных нод")
    nodes = tuple(spec.name for spec in specs)
    if window_weights is not None:
        window_weights = dict(window_weights)
        if any(type(t) is not int or t < 0 or type(weight) is not int or weight < 0
               for t, weight in window_weights.items()):
            raise ValueError("Веса окон: неотрицательные целые времена и кратности")
    states = state_space(nodes)
    probabilities, counts, label_counts = {}, {}, {}
    past = store.known(cutoff)
    for layer in LAYERS:
        for context in CONTEXTS:
            _, prior = group_distribution(specs, layer, context)
            patterns = [observed_pattern(row, specs) for row in past
                        if row.layer == layer and row.context == context
                        for _ in range(1 if window_weights is None else window_weights.get(row.time, 0))]
            patterns = [pattern for pattern in patterns if pattern]
            counts[layer, context] = len(patterns)
            for node in nodes:
                label_counts[layer, context, node] = sum(any(name == node for name, _ in pattern)
                                                         for pattern in patterns)
            if method in ("joint", "unsmoothed"):
                learned = learn_joint(states, prior, patterns, 0 if method == "unsmoothed" else strength, iterations)
            elif method == "independent":
                learned = learn_independent(nodes, states, prior, patterns, strength)
            else:
                learned = prior
            probabilities[layer, context] = learned
    return LearnedModel(nodes, specs, cutoff, method, 0 if method == "unsmoothed" else strength, states,
                        probabilities, counts, label_counts)


def select_strength(store, registry, *, train_cutoff, validation_cutoff, choices=(2, 10, 40), method="joint"):
    """Подбор только по уже известной исторической проверке, не по будущей истине."""
    if train_cutoff >= validation_cutoff:
        raise ValueError("Обучение должно предшествовать проверке")
    validation = [row for row in store.known(validation_cutoff) if row.time >= train_cutoff]
    scores = []
    for strength in choices:
        model = fit(store, registry, train_cutoff, strength=strength, method=method)
        probabilities = [model.pattern_probability(row) for row in validation]
        probabilities = [p for p in probabilities if p is not None]
        if not probabilities:
            raise ValueError("Нет наблюдений для исторической проверки")
        score = -sum(math.log(max(p, 1e-300)) for p in probabilities) / len(probabilities)
        scores.append({"strength": strength, "negative_log_score": score, "rows": len(probabilities)})
    best = min(scores, key=lambda row: (row["negative_log_score"], row["strength"]))
    return best["strength"], scores
