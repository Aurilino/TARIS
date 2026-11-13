"""I4.10: применение зафиксированной модели к отложенным мирам."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
import hashlib
import json
from math import sqrt
from subprojects.taris_variants.restricted_forecast import dataset_rows, predict
from subprojects.taris_variants.diagnose_restricted import observed_state


def scores(rows, field):
    if not rows:
        return None
    errors = [r[field]-r["actual_window_volume"] for r in rows]
    return {"n": len(rows), "mae": sum(abs(v) for v in errors)/len(rows),
            "rmse": sqrt(sum(v*v for v in errors)/len(rows)), "bias": sum(errors)/len(rows)}


def main():
    folder = Path(__file__).parent
    model_path = folder/"state_model_i410/model.json"
    model_bytes = model_path.read_bytes()
    model = json.loads(model_bytes)
    assert model["training_seeds"] == [31, 32, 33] and model["heldout_seeds"] == [37, 38, 39]
    for name, digest in model["source_hashes"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    assert model["training_diagnosis_hash"] == hashlib.sha256((folder/"restricted_diagnosis_i49/results.json").read_bytes()).hexdigest()
    source = folder/"series_i410_validation"
    audit = json.loads((source/"label_audit.json").read_text(encoding="utf-8"))
    rows, hashes = [], {}
    for seed in model["heldout_seeds"]:
        path = source/f"s{seed}.json"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert audit["input_hashes"][path.name] == hashes[path.name]
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        assert case["seed"] == seed
        states = {(profile, variant, snap["time"]): observed_state(snap)
            for profile, variants in case["results"].items() for variant, record in variants.items()
            for snap in record["observations"]}
        current = dataset_rows(case, 7.)
        for row in current:
            row["state"] = states[row["profile"], row["variant"], row["time"]]
        rows.extend(current)
    groups = defaultdict(list)
    for row in rows:
        profile_model = model["models"][row["profile"]]
        coefficient = profile_model["models"][row["state"]]["coefficient"] if profile_model["use_state_model"] else 1.
        row["baseline_prediction"] = predict(row)
        row["calibrated_prediction"] = predict(row, coefficient)
        groups[row["seed"], row["profile"], row["variant"]].append(row)
    summaries = []
    for (seed, profile, variant), group in sorted(groups.items()):
        nonempty = [r for r in group if r["known_work"] > 0]
        summaries.append({"seed": seed, "profile": profile, "variant": variant,
            "eligible_snapshots": len(group), "late_excluded_snapshots": 6, "nonempty_snapshots": len(nonempty),
            "baseline": scores(group, "baseline_prediction"), "calibrated": scores(group, "calibrated_prediction"),
            "nonempty_baseline": scores(nonempty, "baseline_prediction"),
            "nonempty_calibrated": scores(nonempty, "calibrated_prediction")})
    aggregates = []
    for profile in ("ample", "constrained"):
        for variant in ("classic", "graph", "hybrid"):
            group = [r for r in summaries if r["profile"] == profile and r["variant"] == variant]
            aggregate = {"profile": profile, "variant": variant, "worlds": len(group),
                "eligible_snapshots": sum(r["eligible_snapshots"] for r in group),
                "nonempty_snapshots": sum(r["nonempty_snapshots"] for r in group)}
            for mode in ("baseline", "calibrated"):
                aggregate[mode] = {"mae": sum(r[mode]["mae"] for r in group)/len(group),
                    "rmse": sqrt(sum(r[mode]["rmse"]**2 for r in group)/len(group)),
                    "bias": sum(r[mode]["bias"] for r in group)/len(group)}
            aggregates.append(aggregate)
    output = folder/"state_evaluation_i410"
    output.mkdir(exist_ok=True)
    report = {"schema": "taris-state-evaluation/1", "models": model["models"],
        "model_hash": hashlib.sha256(model_bytes).hexdigest(), "validation_input_hashes": hashes,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "independence_unit": "world", "rows": rows, "per_world": summaries, "summary": aggregates,
        "expected_loss_cost": None, "full_actuarial_reserve": None}
    assert model_path.read_bytes() == model_bytes
    (output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# I4.10 — отложенная проверка", "", "Объём передачи текущих repair-заданий за семь суток. Новые миры 37–39; gate и коэффициенты зафиксированы заранее.", "",
        "| Канал | Вариант | Снимков | Непустых | MAE базис | MAE калибр. | RMSE базис | RMSE калибр. | Смещение базис | Смещение калибр. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregates:
        b, c = row["baseline"], row["calibrated"]
        lines.append(f'| {row["profile"]} | {row["variant"]} | {row["eligible_snapshots"]} | {row["nonempty_snapshots"]} | '
            f'{b["mae"]:.6f} | {c["mae"]:.6f} | {b["rmse"]:.6f} | {c["rmse"]:.6f} | {b["bias"]:.6f} | {c["bias"]:.6f} |')
    lines += ["", "Ошибки сначала усреднены внутри мира, затем по трём мирам; RMSE получен из среднего MSE. "
        "В JSON есть отдельные ошибки непустых снимков и каждого мира. Шесть поздних снимков каждого исполнения исключены, "
        "их будущее не объявлено нулём. Цена передачи равна объёму ×0,1. Это не цена всей будущей очереди и не резерв потери данных."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("I4.10: отложенные прогнозы и ошибки сохранены")


if __name__ == "__main__":
    main()
