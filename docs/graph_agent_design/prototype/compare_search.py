"""Воспроизводимое сравнение графового поиска и точного эталона."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import random
from time import perf_counter

from exact_model import ExactModel, placement_key
from graph_search import graph_search
from run_examples import encode


def trap_case(name="local_minimum", availability_limit="1", max_n=2):
    """Согласованный граф: AB — локальный минимум, CD — глобальный."""
    return {
        "name": name, "nodes": ["A", "B", "C", "D"],
        "availability": [
            {"probability": "0.85", "affected": []},
            {"probability": "0.01", "affected": ["A", "B"]},
            *[{"probability": "0.02", "affected": [i, j]}
              for i, j in (("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"))],
            {"probability": "0.03", "affected": ["C"]},
            {"probability": "0.03", "affected": ["D"]},
        ],
        "loss": [{"probability": 1, "affected": []}],
        "request": {"k": 1, "min_n": 2, "max_n": max_n,
                    "availability_limit": availability_limit, "loss_limit": 0},
    }


def generated_cases(count=12):
    """Учебные конечные распределения; это не калибровка реальной сети."""
    cases = []
    for seed in range(count):
        rng = random.Random(7100 + seed)
        nodes = [f"N{i:02d}" for i in range(6 + seed % 5)]
        layers = []
        for affected_chance, healthy_mass in ((0.35, Fraction(1, 2)), (0.22, Fraction(9, 10))):
            weights = [rng.randint(1, 10) for _ in range(12)]
            total = sum(weights)
            states = [{"probability": str(healthy_mass), "affected": []}]
            for weight in weights:
                states.append({"probability": str((1 - healthy_mass) * Fraction(weight, total)),
                               "affected": [n for n in nodes if rng.random() < affected_chance]})
            layers.append(states)
        cases.append({"name": f"generated_{seed:02d}", "nodes": nodes,
                      "availability": layers[0], "loss": layers[1],
                      "request": {"k": 2, "availability_limit": "0.08", "loss_limit": "0.01"}})
    return cases


def compare_case(case, seed_count):
    model = ExactModel(case["nodes"], case["availability"], case["loss"])
    start = perf_counter()
    exact = model.search(**case["request"])
    exact_seconds = perf_counter() - start
    start = perf_counter()
    heuristic = graph_search(model, **case["request"], seed_count=seed_count)
    graph_seconds = perf_counter() - start
    extra = gap = None
    if exact.best and heuristic.best:
        extra = len(heuristic.best.nodes) - len(exact.best.nodes)
        if extra == 0:
            gap = heuristic.best.graph_score - exact.best.graph_score
        outcome = "same_optimum" if placement_key(heuristic.best) == placement_key(exact.best) else (
            "extra_chunks" if extra else "worse_same_size")
    elif exact.best:
        outcome = "missed_feasible"
    else:
        outcome = "both_without_solution"
    return {"name": case["name"], "seed_count": seed_count, "node_count": len(model.nodes),
            "request": case["request"], "exact": asdict(exact), "heuristic": asdict(heuristic),
            "outcome": outcome, "extra_chunks": extra, "graph_score_gap_same_size": gap,
            "exact_seconds": exact_seconds, "graph_seconds_including_preparation": graph_seconds}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Сравнение поиска TARIS на малых сетях")
    parser.add_argument("--output", type=Path, default=root / "search_comparison.json")
    args = parser.parse_args()
    baseline = json.loads((root / "examples.json").read_text(encoding="utf-8"))["cases"]
    cases = baseline + [trap_case(), trap_case("extra_chunk_trap", "0", 3),
                        trap_case("missed_solution_trap", "0", 2)] + generated_cases()
    rows = [compare_case(case, seeds) for case in cases for seeds in (1, 3)]
    summary = {}
    for seeds in (1, 3):
        subset = [r for r in rows if r["seed_count"] == seeds]
        summary[str(seeds)] = {
            "cases": len(subset),
            "outcomes": {key: sum(r["outcome"] == key for r in subset) for key in (
                "same_optimum", "worse_same_size", "extra_chunks", "missed_feasible", "both_without_solution")},
            "exact_checked": sum(r["exact"]["checked"] for r in subset),
            "heuristic_checked": sum(r["heuristic"]["checked"] for r in subset),
            "exact_seconds": sum(r["exact_seconds"] for r in subset),
            "graph_seconds_including_preparation": sum(r["graph_seconds_including_preparation"] for r in subset),
        }
    report = {"schema": "taris-search-comparison/1", "generated_seed_base": 7100,
              "inputs": cases, "summary": summary, "comparisons": rows}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=encode) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Отчёт: {args.output.resolve()}")


if __name__ == "__main__":
    main()
