"""Р3: последовательный учебный опыт, проверка риска и поиска раздельно."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from fractions import Fraction
from itertools import combinations
import json
import math
from pathlib import Path
import random

from exact_model import Distribution, ExactModel
from graph_search import graph_search
from learning import (CONTEXTS, LAYERS, NodeSpec, Observation, ObservationStore,
                      fit, rational_rows, select_strength, state_space)
from run_examples import encode


METHODS = ("prior", "independent", "unsmoothed", "joint")


def true_distribution(registry, layer, context):
    """Только среда/оценщик; эта функция не вызывается обучаемой моделью."""
    own_day = dict(zip("ABCDEF", (0.01, 0.09, 0.03, 0.08, 0.07, 0.02)))
    own_night = dict(zip("ABCDEF", (0.35, 0.18, 0.04, 0.11, 0.24, 0.08)))
    own_loss = dict(zip("ABCDEF", (0.004, 0.008, 0.003, 0.006, 0.010, 0.005)))
    own = own_loss if layer == "loss" else own_day if context == "day" else own_night
    shock = 0.012 if layer == "loss" else 0.04 if context == "day" else 0.12
    groups = defaultdict(list)
    for spec in registry:
        groups[spec.group].append(spec.name)
    states = state_space(tuple(spec.name for spec in registry))
    probabilities = []
    for state in states:
        p = 1.0
        for members in groups.values():
            own_pattern = math.prod(own[node] if node in state else 1 - own[node] for node in members)
            p *= (1 - shock) * own_pattern + (shock if all(node in state for node in members) else 0)
        probabilities.append(p)
    total = sum(probabilities)
    return states, tuple(p / total for p in probabilities)


def sample_state(rng, states, probabilities):
    draw = rng.random()
    cumulative = 0.0
    for state, p in zip(states, probabilities):
        cumulative += p
        if draw < cumulative:
            return state
    return states[-1]


def make_experiment(seed, cutoff=160, end=320):
    registry = tuple(NodeSpec(name, f"site_{index // 2}", cutoff if name == "F" else 0)
                     for index, name in enumerate("ABCDEF"))
    # Разные потоки генерации мира и проверок: маска не зависит от исхода.
    world_rng, probe_rng = random.Random(8300 + seed), random.Random(9300 + seed)
    distributions = {(layer, context): true_distribution(registry, layer, context)
                     for layer in LAYERS for context in CONTEXTS}
    store, truth = ObservationStore(), []
    for t in range(end):
        context = "day" if t % 4 < 2 else "night"
        for layer in LAYERS:
            states, probabilities = distributions[layer, context]
            affected = sample_state(world_rng, states, probabilities)
            active = tuple(spec.name for spec in registry if spec.joined_at <= t)
            truth.append({"time": t, "layer": layer, "context": context,
                          "affected": affected.intersection(active)})
            chance = 0.65 if layer == "availability" else 0.55
            values = tuple((node, node in affected if probe_rng.random() < chance else None) for node in active)
            delay = 0 if layer == "availability" else probe_rng.randrange(4)
            store.add(Observation(t, t + 1 + delay, layer, context, values))
    return registry, store, truth, distributions


def score_predictions(learned, truth, distributions, start, end):
    """Истина используется здесь только после обучения, для внешней оценки."""
    scores = {}
    contextual = {}
    for context in CONTEXTS:
        rows = [rational_rows(learned.states, learned.probabilities[layer, context]) for layer in LAYERS]
        contextual[context] = ExactModel(learned.nodes, *rows)
    for layer in LAYERS:
        sums = defaultdict(float)
        counts = defaultdict(int)
        for context in CONTEXTS:
            predicted = getattr(contextual[context], layer)
            true_states, true_probabilities = distributions[layer, context]
            oracle = Distribution(learned.nodes, rational_rows(true_states, true_probabilities))
            targets = [row for row in truth if start <= row["time"] < end
                       and row["layer"] == layer and row["context"] == context]
            features = [("node", (node,), 1) for node in learned.nodes]
            features += [("pair", pair, 1) for pair in combinations(learned.nodes, 2)]
            # risk(..., 1) для пары: потеря обоих; для объекта: менее двух из трёх.
            features += [("object_2_of_3", triple, 2) for triple in combinations(learned.nodes, 3)]
            for category, nodes, k in features:
                p = float(predicted.risk(nodes, k))
                true_p = float(oracle.risk(nodes, k))
                sums[category + "_probability_absolute_error"] += abs(p - true_p)
                counts[category + "_probability_absolute_error"] += 1
                for row in targets:
                    outcome = int(len(set(nodes) - row["affected"]) < k)
                    sums[category + "_brier"] += (p - outcome)**2
                    counts[category + "_brier"] += 1
        scores[layer] = {key: value / counts[key] for key, value in sums.items()}
    return scores


def oracle_model(registry, distributions):
    nodes = tuple(spec.name for spec in registry)
    layers = []
    for layer in LAYERS:
        states = distributions[layer, "day"][0]
        mixed = [(a + b) / 2 for a, b in zip(distributions[layer, "day"][1], distributions[layer, "night"][1])]
        layers.append(rational_rows(states, mixed))
    return ExactModel(nodes, *layers)


def search_diagnostics(learned, oracle):
    prediction = learned.exact_model()
    request = {"k": 2, "availability_limit": "0.08", "loss_limit": "0.025"}
    exact = prediction.search(**request)
    heuristic = graph_search(prediction, **request)
    oracle_search = oracle.search(**request)
    selected = heuristic.best
    true_evaluation = oracle.evaluate(selected.nodes, 2) if selected else None
    exact_true = oracle.evaluate(exact.best.nodes, 2) if exact.best else None
    violation = (bool(true_evaluation.availability_risk > Fraction(request["availability_limit"])
                      or true_evaluation.loss_risk > Fraction(request["loss_limit"]))
                 if true_evaluation else None)
    exact_violation = (bool(exact_true.availability_risk > Fraction(request["availability_limit"])
                            or exact_true.loss_risk > Fraction(request["loss_limit"]))
                       if exact_true else None)
    return {
        "request": request,
        "exact_on_learned": asdict(exact), "graph_on_learned": asdict(heuristic),
        "exact_on_oracle": asdict(oracle_search),
        "graph_selected_true_evaluation": asdict(true_evaluation) if true_evaluation else None,
        "exact_selected_true_evaluation": asdict(exact_true) if exact_true else None,
        "graph_selected_violates_true_limit": violation,
        "exact_selected_violates_true_limit": exact_violation,
        "graph_violates_true_availability": (true_evaluation.availability_risk > Fraction(request["availability_limit"])
                                              if true_evaluation else None),
        "graph_violates_true_loss": (true_evaluation.loss_risk > Fraction(request["loss_limit"])
                                     if true_evaluation else None),
        "search_extra_chunks_vs_learned_exact": (len(selected.nodes) - len(exact.best.nodes)
                                                 if selected and exact.best else None),
        "search_graph_score_gap_same_n": (selected.graph_score - exact.best.graph_score
                                           if selected and exact.best and len(selected.nodes) == len(exact.best.nodes) else None),
    }


def run(seed):
    cutoff, end = 160, 320
    registry, store, truth, distributions = make_experiment(seed, cutoff, end)
    oracle = oracle_model(registry, distributions)
    methods, selections = {}, {}
    for method in METHODS:
        strength = 10
        if method in ("joint", "independent"):
            strength, validation = select_strength(store, registry, train_cutoff=100,
                                                   validation_cutoff=cutoff, method=method)
            selections[method] = {"chosen_strength": strength, "scores": validation}
        learned = fit(store, registry, cutoff, method=method, strength=strength)
        scores = score_predictions(learned, truth, distributions, cutoff, end)
        mixed = learned.exact_model()
        new_node = {layer: float(getattr(mixed, layer).own("F")) for layer in LAYERS}
        graph = {layer: {
            "own": {node: getattr(mixed, layer).own(node) for node in learned.nodes},
            "pairs": [{"nodes": [i, j], "weight": getattr(mixed, layer).pair(i, j)}
                      for i, j in combinations(learned.nodes, 2)],
        } for layer in LAYERS}
        methods[method] = {
            "scores": scores, "search": search_diagnostics(learned, oracle),
            "fitted_strength": learned.strength,
            "learned_graph": graph,
            "cold_node_F_risk": new_node,
            "cold_node_F_labels": sum(count for (layer, context, node), count in learned.label_counts.items() if node == "F"),
            "sample_counts": {f"{layer}/{context}": count for (layer, context), count in learned.sample_counts.items()},
            "observed_positive_labels": {
                layer: sum(value is True for row in store.known(cutoff) if row.layer == layer
                           for name, value in row.values) for layer in LAYERS},
        }
        if method == "joint":
            methods[method]["state_model"] = {
                f"{layer}/{context}": rational_rows(learned.states, learned.probabilities[layer, context])
                for layer in LAYERS for context in CONTEXTS}
    return {"seed": seed, "cutoff": cutoff, "test_end": end,
            "strength_selection": selections, "methods": methods}


def main():
    parser = argparse.ArgumentParser(description="Р3: обучение и проверка на будущем")
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("learning_results.json"))
    args = parser.parse_args()
    if args.seeds < 1:
        parser.error("Нужно положительное число зёрен")
    runs = [run(seed) for seed in range(args.seeds)]
    summary = {}
    for method in METHODS:
        rows = [r["methods"][method] for r in runs]
        summary[method] = {
            "scores": {layer: {metric: sum(row["scores"][layer][metric] for row in rows) / len(rows)
                               for metric in rows[0]["scores"][layer]} for layer in LAYERS},
            "graph_found": sum(row["search"]["graph_on_learned"]["best"] is not None for row in rows),
            "true_limit_violations_graph": sum(row["search"]["graph_selected_violates_true_limit"] is True for row in rows),
            "true_limit_violations_exact_on_learned": sum(row["search"]["exact_selected_violates_true_limit"] is True for row in rows),
            "true_availability_limit_violations_graph": sum(row["search"]["graph_violates_true_availability"] is True for row in rows),
            "true_loss_limit_violations_graph": sum(row["search"]["graph_violates_true_loss"] is True for row in rows),
            "search_extra_chunks_cases": sum((row["search"]["search_extra_chunks_vs_learned_exact"] or 0) > 0 for row in rows),
        }
    report = {"schema": "taris-learning-results/1", "seeds": args.seeds,
              "world_seed_base": 8300, "probe_seed_base": 9300,
              "settings": {"training_cutoff": 100, "decision_cutoff": 160, "test_end": 320,
                           "probe_probability_availability": 0.65, "probe_probability_loss": 0.55,
                           "maximum_audit_delay": 3, "strength_choices": [2, 10, 40],
                           "em_iteration_limit": 30, "em_tolerance": 1e-10,
                           "rationalization_scale": 10**12, "uncertainty_intervals": False},
              "summary": summary, "runs": runs}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=encode) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Отчёт: {args.output.resolve()}")


if __name__ == "__main__":
    main()
