"""Автономная конечная модель: точные вероятности, без импорта taris_sim."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Iterable


def probability(value) -> Fraction:
    if isinstance(value, bool):
        raise ValueError("Вероятность не может быть логическим значением")
    try:
        result = Fraction(str(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Некорректная вероятность") from exc
    if not 0 <= result <= 1:
        raise ValueError("Вероятность должна находиться между 0 и 1")
    return result


def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name}: требуется положительное целое число")


@dataclass(frozen=True)
class State:
    probability: Fraction
    affected: frozenset[str]


class Distribution:
    """Полное распределение состояний одного слоя, с суммой вероятностей 1."""

    def __init__(self, nodes: Iterable[str], states: Iterable[dict]):
        self.nodes = tuple(nodes)
        if (not self.nodes or any(not isinstance(n, str) or not n for n in self.nodes)
                or len(set(self.nodes)) != len(self.nodes)):
            raise ValueError("Нужны непустые уникальные строковые идентификаторы нод")
        known = set(self.nodes)
        parsed = []
        for row in states:
            affected_list = tuple(row["affected"])
            if any(not isinstance(n, str) for n in affected_list):
                raise ValueError("Идентификаторы нод состояния должны быть строками")
            affected = frozenset(affected_list)
            if len(affected) != len(affected_list) or not affected <= known:
                raise ValueError("Повторяющиеся либо неизвестные ноды в состоянии")
            parsed.append(State(probability(row["probability"]), affected))
        if sum((s.probability for s in parsed), Fraction()) != 1:
            raise ValueError("Сумма вероятностей состояний должна точно равняться 1")
        self.states = tuple(parsed)

    def risk(self, nodes: Iterable[str], k: int) -> Fraction:
        selected = tuple(nodes)
        positive_integer(k, "k")
        if len(set(selected)) != len(selected) or not set(selected) <= set(self.nodes):
            raise ValueError("Размещение содержит повторяющиеся либо неизвестные ноды")
        return sum((s.probability for s in self.states
                    if len(set(selected) - s.affected) < k), Fraction())

    def own(self, node: str) -> Fraction:
        if node not in self.nodes:
            raise ValueError("Неизвестная нода")
        return sum((s.probability for s in self.states if node in s.affected), Fraction())

    def pair(self, left: str, right: str) -> Fraction:
        if left == right or left not in self.nodes or right not in self.nodes:
            raise ValueError("Ребро должно соединять две разные известные ноды")
        return sum((s.probability for s in self.states
                    if left in s.affected and right in s.affected), Fraction())


@dataclass(frozen=True)
class Evaluation:
    nodes: tuple[str, ...]
    availability_risk: Fraction
    loss_risk: Fraction
    graph_score: Fraction
    own_availability_sum: Fraction


@dataclass(frozen=True)
class SearchResult:
    status: str
    checked: int
    feasible_at_min_n: int
    best: Evaluation | None
    min_n: int
    max_n: int


def placement_key(evaluation: Evaluation):
    """Единое правило сравнения допустимых размещений обоих поисковиков."""
    return (evaluation.graph_score, evaluation.own_availability_sum,
            evaluation.loss_risk, evaluation.nodes)


class ExactModel:
    """Доступность и потери — отдельные слои; пути восстановления не моделируются."""

    def __init__(self, nodes, availability_states, loss_states):
        nodes = tuple(nodes)
        self.availability = Distribution(nodes, availability_states)
        self.loss = Distribution(nodes, loss_states)
        self.nodes = self.availability.nodes

    def evaluate(self, nodes: Iterable[str], k: int) -> Evaluation:
        selected = tuple(sorted(nodes))
        availability_risk = self.availability.risk(selected, k)
        loss_risk = self.loss.risk(selected, k)
        return Evaluation(
            selected, availability_risk, loss_risk,
            sum((self.availability.pair(i, j) for i, j in combinations(selected, 2)), Fraction()),
            sum((self.availability.own(i) for i in selected), Fraction()),
        )

    def search_domain(self, *, k: int, availability_limit, loss_limit,
                      min_n: int | None = None, max_n: int | None = None,
                      free_slots: dict[str, int] | None = None):
        """Общие проверки и область поиска для точного и графового алгоритмов."""
        positive_integer(k, "k")
        min_n = k if min_n is None else min_n
        max_n = len(self.nodes) if max_n is None else max_n
        positive_integer(min_n, "min_n")
        positive_integer(max_n, "max_n")
        if min_n < k or max_n < min_n:
            raise ValueError("Требуется k <= min_n <= max_n")
        a_limit, l_limit = probability(availability_limit), probability(loss_limit)
        slots = {node: 1 for node in self.nodes} if free_slots is None else dict(free_slots)
        if set(slots) != set(self.nodes):
            raise ValueError("Свободные места должны быть заданы для всех и только известных нод")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in slots.values()):
            raise ValueError("Свободные места: неотрицательные целые числа")
        candidates = tuple(sorted(node for node in self.nodes if slots[node] > 0))
        return candidates, min_n, max_n, a_limit, l_limit

    def search(self, *, k: int, availability_limit, loss_limit,
               min_n: int | None = None, max_n: int | None = None,
               free_slots: dict[str, int] | None = None) -> SearchResult:
        candidates, min_n, max_n, a_limit, l_limit = self.search_domain(
            k=k, availability_limit=availability_limit, loss_limit=loss_limit,
            min_n=min_n, max_n=max_n, free_slots=free_slots)
        if len(candidates) < min_n:
            return SearchResult("insufficient_capacity", 0, 0, None, min_n, max_n)
        checked = 0
        for n in range(min_n, min(max_n, len(candidates)) + 1):
            feasible = []
            for selected in combinations(candidates, n):
                evaluation = self.evaluate(selected, k)
                checked += 1
                if (evaluation.availability_risk <= a_limit
                        and evaluation.loss_risk <= l_limit):
                    feasible.append(evaluation)
            if feasible:
                # Собственный риск — только разрешение равенства графового критерия.
                best = min(feasible, key=placement_key)
                return SearchResult("optimal_in_scope", checked, len(feasible), best, min_n, max_n)
        return SearchResult("proven_infeasible_in_scope", checked, 0, None, min_n, max_n)
