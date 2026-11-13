"""Сквозной автономный опыт: обучение → граф → размещение → восстановление."""
from dataclasses import asdict
import json
from pathlib import Path

from crisis_trials import ForecastRecoveryPolicy, WarningForecaster, WarningOutcome
from graph_search import graph_search
from learning import select_strength
from recovery import ObjectSpec, RecoveryEngine, RecoveryPolicy, StorageNode
from run_examples import encode
from run_learning import make_experiment
from run_recovery import summarize
from uncertainty import build_ensembles


def run(seed=12):
    registry, store, _, _ = make_experiment(seed)
    strength, validation = select_strength(store, registry, train_cutoff=100, validation_cutoff=160)
    models, metadata = build_ensembles(store, registry, 160, strength=strength,
                                      seed=14000 + seed * 100, replicas=8)
    # Прошлая наблюдаемая разметка сигналов, отдельная от будущих сценариев.
    records = [WarningOutcome(t, t + 1, "site_0", t % 5 != 0) for t in range(40)]
    forecaster = WarningForecaster(records, 40)
    shock = [(2, {"kind": "destroy", "node": "A"}), ("2.5", {"kind": "destroy", "node": "B"})]
    warning = [(0, {"kind": "warning", "site": "site_0", "duration": 3})]
    scenarios = {
        "regular_down": [(0, {"kind": "down", "node": "A"}), (4, {"kind": "up", "node": "A"})],
        "predictable": warning + shock,
        "sudden": shock,
        "false_warning": warning,
    }
    results = {}
    for model_name in ("point", "bootstrap"):
        model = models[model_name]
        placement = graph_search(model, k=2, availability_limit="0.08", loss_limit="0.025")
        if placement.best is None:
            results[model_name] = {"placement": asdict(placement), "episodes": None}
            continue
        episodes = {}
        for scenario_name, events in scenarios.items():
            for forecast in (False, True):
                nodes = [StorageNode(spec.name, spec.group) for spec in registry]
                engine = RecoveryEngine(nodes, [ObjectSpec("O", 2, len(placement.best.nodes))],
                    site_bandwidth={spec.group: 2 for spec in registry})
                for index, node in enumerate(placement.best.nodes):
                    engine.add_chunk("O", index, node)
                policy = (ForecastRecoveryPolicy(forecaster, risk_model=model) if forecast
                          else RecoveryPolicy(model))
                engine.run(8, events=events, policy=policy, decision_interval="0.5")
                episodes[f"{scenario_name}/forecast_{int(forecast)}"] = summarize(engine)
        results[model_name] = {"placement": asdict(placement), "episodes": episodes}
    return {"seed": seed, "strength": strength, "validation": validation,
            "ensemble": metadata, "results": results}


def main():
    runs = [run(seed) for seed in (12, 13, 14)]
    output = Path(__file__).with_name("pipeline_results.json")
    report = {"schema": "taris-autonomous-pipeline/1", "runs": runs,
              "limitation": "Обучение на независимых окнах, исполнение в заданных динамических сценариях; не проверка калибровки на этих кризисах."}
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=encode) + "\n", encoding="utf-8")
    for run_result in runs:
        for name, result in run_result["results"].items():
            episodes = result["episodes"] or {}
            print(run_result["seed"], name, "чанки:", result["placement"]["best"]["nodes"] if result["placement"]["best"] else None,
                  "потерянных объектов:", sum(bool(episode["lost"]) for episode in episodes.values()),
                  "эпизодов:", len(episodes))


if __name__ == "__main__":
    main()
