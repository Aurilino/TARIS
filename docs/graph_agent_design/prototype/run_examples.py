r"""Запуск из корня: .venv\Scripts\python.exe docs/graph_agent_design/prototype/run_examples.py"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path

from exact_model import ExactModel


def run_case(case):
    model = ExactModel(case["nodes"], case["availability"], case["loss"])
    result = model.search(**case["request"])
    graph = {}
    for name, layer in (("availability", model.availability), ("loss", model.loss)):
        graph[name] = {
            "own": {node: layer.own(node) for node in model.nodes},
            "pairs": [{"nodes": [i, j], "weight": layer.pair(i, j)}
                      for i, j in combinations(model.nodes, 2)],
        }
    return {"name": case["name"], "description": case["description"],
            "request": case["request"], "graph": graph,
            "all_nodes_evaluation": asdict(model.evaluate(model.nodes, case["request"]["k"])),
            "search": asdict(result)}


def encode(value):
    if isinstance(value, Fraction):
        return str(value)
    raise TypeError(type(value).__name__)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Точная проверка малых сетей TARIS")
    parser.add_argument("--input", type=Path, default=root / "examples.json")
    parser.add_argument("--output", type=Path, default=root / "results.json")
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if source.get("schema") != "taris-exact-example/1":
        raise ValueError("Неизвестная схема примеров")
    results = [run_case(case) for case in source["cases"]]
    report = {"schema": "taris-exact-results/1", "probabilities": "exact rational strings",
              "cases": results}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=encode) + "\n",
                           encoding="utf-8")
    for case in results:
        result = case["search"]
        nodes = result["best"]["nodes"] if result["best"] else None
        print(f"{case['name']}: {result['status']}; ноды={nodes}; проверено={result['checked']}")
    print(f"Журнал: {args.output.resolve()}")


if __name__ == "__main__":
    main()
