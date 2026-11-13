"""Объяснительный разбор только обучающих миров I4.9."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import defaultdict
import hashlib
import json


def observed_state(snapshot):
    repairs = [j for j in snapshot["jobs"] if j["kind"] == "repair"]
    if not repairs:
        return "empty_repair_queue"
    if snapshot["parked_job"] is not None and snapshot["full_buffer_objects"]:
        return "parked_full_buffer"
    phase = snapshot["planned_channel_phase"]
    if phase is not None and phase["phase"] in ("read_block", "write_block", "close_proof"):
        return "repair_transfer"
    if phase is not None and phase["phase"] == "proof":
        return "proof_transfer"
    return "waiting_or_setup"


def main():
    folder = Path(__file__).parent
    model_path = folder/"restricted_model_i49/model.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    lookup = {(r["seed"], r["profile"], r["variant"], r["time"]): r for r in model["rows"]}
    groups = defaultdict(list)
    details = []
    for name, digest in model["training_input_hashes"].items():
        path = folder/"series_i48_train"/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                for snapshot in record["observations"]:
                    key = case["seed"], profile, variant, snapshot["time"]
                    if key not in lookup:
                        continue
                    row = lookup[key]
                    state = observed_state(snapshot)
                    detail = {**row, "state": state,
                        "nominal_prediction": min(row["known_work"], row["capacity"])}
                    details.append(detail)
                    if row["known_work"] > 0:
                        groups[case["seed"], profile, state].append(detail)
    summaries = []
    for (seed, profile, state), rows in sorted(groups.items()):
        actual = sum(r["actual_window_volume"] for r in rows)/len(rows)
        nominal = sum(r["nominal_prediction"] for r in rows)/len(rows)
        summaries.append({"seed": seed, "profile": profile, "state": state, "n": len(rows),
            "mean_actual_volume": actual, "mean_nominal_volume": nominal,
            "nominal_bias": nominal-actual,
            "zero_service_rows": sum(r["actual_window_volume"] == 0 for r in rows)})
    output = folder/"restricted_diagnosis_i49"
    output.mkdir(exist_ok=True)
    result = {"schema": "taris-restricted-training-diagnosis/1", "seeds": [31, 32, 33],
        "validation_used": False, "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_hash": hashlib.sha256(model_path.read_bytes()).hexdigest(), "rows": details, "summary": summaries}
    (output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Диагностика I4.9: только обучение", "", "Непустые repair-снимки; варианты внутри мира объединены. Строки зависимы.", "",
        "| Мир | Канал | Наблюдаемое состояние | Строк | Средний фактический объём | Средний базис | Строк без обслуживания |",
        "|---:|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f'| {row["seed"]} | {row["profile"]} | {row["state"]} | {row["n"]} | '
                     f'{row["mean_actual_volume"]:.6f} | {row["mean_nominal_volume"]:.6f} | {row["zero_service_rows"]} |')
    lines += ["", "Состояние определено только из снимка. Низкое будущее обслуживание не доказывает конкретную причину "
              "ожидания и не означает сохранность данных. Это объяснительный разбор, не новая отложенная проверка."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("I4.9: обучающая диагностика сохранена; проверочные миры не использованы")


if __name__ == "__main__":
    main()
