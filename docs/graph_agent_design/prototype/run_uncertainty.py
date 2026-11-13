"""Сравнение точечных и ансамблевых ограничений на новых зёрнах."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

from graph_search import graph_search
from learning import select_strength
from run_examples import encode
from run_learning import make_experiment, oracle_model
from uncertainty import build_ensembles


REQUEST = {"k": 2, "availability_limit": "0.08", "loss_limit": "0.025"}


def run(seed, replicas):
    registry, store, _, distributions = make_experiment(seed)
    strength, validation = select_strength(store, registry, train_cutoff=100, validation_cutoff=160)
    models, metadata = build_ensembles(store, registry, 160, strength=strength,
                                      seed=10300 + seed * 100, replicas=replicas)
    oracle = oracle_model(registry, distributions)
    results = {}
    for name, model in models.items():
        exact = model.search(**REQUEST)
        graph = graph_search(model, **REQUEST)
        selected = graph.best
        true_risk = oracle.evaluate(selected.nodes, 2) if selected else None
        results[name] = {
            "model_count": len(model.models), "exact": asdict(exact), "graph": asdict(graph),
            "true_risk": asdict(true_risk) if true_risk else None,
            "availability_violation": (true_risk.availability_risk > Fraction(REQUEST["availability_limit"])
                                       if true_risk else None),
            "loss_violation": (true_risk.loss_risk > Fraction(REQUEST["loss_limit"])
                               if true_risk else None),
            "per_member_selected_risks": [asdict(member.evaluate(selected.nodes, 2))
                                           for member in model.models] if selected else [],
        }
    return {"seed": seed, "strength": strength, "validation": validation,
            "ensemble": metadata, "results": results}


def main():
    parser = argparse.ArgumentParser(description="Р3.2: устойчивое размещение")
    parser.add_argument("--start-seed", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("uncertainty_results.json"))
    args = parser.parse_args()
    if args.seeds < 1 or args.replicas < 1 or args.start_seed < 0:
        parser.error("Некорректные параметры зёрен или повторных выборок")
    runs = []
    for seed in range(args.start_seed, args.start_seed + args.seeds):
        runs.append(run(seed, args.replicas))
        print(f"Зерно {seed} завершено", flush=True)
    summary = {}
    for method in ("point", "bootstrap", "structural", "combined"):
        rows = [r["results"][method] for r in runs]
        found = [row for row in rows if row["graph"]["best"]]
        summary[method] = {
            "found": len(found), "total": len(rows),
            "mean_chunks_when_found": sum(len(row["graph"]["best"]["nodes"]) for row in found) / len(found) if found else None,
            "availability_violations": sum(row["availability_violation"] is True for row in found),
            "loss_violations": sum(row["loss_violation"] is True for row in found),
            "mean_true_availability_when_found": sum(float(Fraction(row["true_risk"]["availability_risk"]))
                                                      for row in found) / len(found) if found else None,
        }
    report = {"schema": "taris-uncertainty-results/1", "request": REQUEST,
              "seed_start": args.start_seed, "seeds": args.seeds, "replicas": args.replicas,
              "summary": summary, "runs": runs,
              "limitation": "Конечный ансамбль; статистический уровень покрытия не заявлен"}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=encode) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
