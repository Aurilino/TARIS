"""Р2: жадное построение и локальные замены с точной проверкой объекта."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

from exact_model import Evaluation, ExactModel, placement_key, positive_integer


@dataclass(frozen=True)
class GraphSearchResult:
    status: str
    checked: int
    feasible_seen_at_returned_n: int
    best: Evaluation | None
    min_n: int
    max_n: int
    starts: int
    accepted_swaps: int
    growth_candidates_scored: int
    graph_pairs_prepared: int


def graph_search(model: ExactModel, *, k: int, availability_limit, loss_limit,
                 min_n=None, max_n=None, free_slots=None,
                 seed_count: int = 3, max_swaps: int = 20) -> GraphSearchResult:
    positive_integer(seed_count, "seed_count")
    if isinstance(max_swaps, bool) or not isinstance(max_swaps, int) or max_swaps < 0:
        raise ValueError("max_swaps: требуется неотрицательное целое число")
    candidates, min_n, max_n, a_limit, l_limit = model.search_domain(
        k=k, availability_limit=availability_limit, loss_limit=loss_limit,
        min_n=min_n, max_n=max_n, free_slots=free_slots)
    cache = {}
    starts = swaps = growth_scores = 0
    pair_count = 0

    def result(status, best=None, feasible=0):
        return GraphSearchResult(status, len(cache), feasible, best, min_n, max_n,
                                 starts, swaps, growth_scores, pair_count)

    if len(candidates) < min_n:
        return result("insufficient_capacity")
    own_a = {i: model.availability.own(i) for i in candidates}
    own_l = {i: model.loss.own(i) for i in candidates}
    pairs = {tuple(sorted((i, j))): model.availability.pair(i, j)
             for i, j in combinations(candidates, 2)}
    pair_count = len(pairs)
    seeds = sorted(candidates, key=lambda i: (own_a[i], own_l[i], i))[:seed_count]

    def evaluate(nodes):
        key = tuple(sorted(nodes))
        if key not in cache:
            cache[key] = model.evaluate(key, k)
        return cache[key]

    def violation(e):
        return (max(Fraction(), e.availability_risk - a_limit)
                + max(Fraction(), e.loss_risk - l_limit))

    def local_key(e):
        # Сначала устраняем нарушения, затем улучшаем тот же критерий, что в эталоне.
        return (violation(e), *placement_key(e))

    for n in range(min_n, min(max_n, len(candidates)) + 1):
        for seed in seeds:
            starts += 1
            selected = [seed]
            while len(selected) < n:
                options = []
                for node in candidates:
                    if node in selected:
                        continue
                    increment = sum((pairs[tuple(sorted((node, i)))] for i in selected), Fraction())
                    options.append((increment, own_a[node], own_l[node], node))
                    growth_scores += 1
                selected.append(min(options)[-1])
            current = evaluate(selected)
            for _ in range(max_swaps):
                neighbors = [current]
                for old in current.nodes:
                    for new in candidates:
                        if new not in current.nodes:
                            trial = tuple(i for i in current.nodes if i != old) + (new,)
                            neighbors.append(evaluate(trial))
                improved = min(neighbors, key=local_key)
                if local_key(improved) >= local_key(current):
                    break
                current = improved
                swaps += 1
        feasible = [e for e in cache.values() if len(e.nodes) == n and violation(e) == 0]
        if feasible:
            return result("feasible_found", min(feasible, key=placement_key), len(feasible))
    # В отличие от полного перебора, эвристика не доказывает невозможность.
    return result("not_found", None)
