"""I4.9: фиксируем коэффициенты до открытия отложенных миров."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
from subprojects.taris_variants.restricted_forecast import dataset_rows, fit_profile


def main():
    folder = Path(__file__).parent
    source = folder/"series_i48_train"
    audit = json.loads((source/"label_audit.json").read_text(encoding="utf-8"))
    rows, hashes = [], {}
    for seed in (31, 32, 33):
        path = source/f"s{seed}.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert audit["input_hashes"][path.name] == digest
        hashes[path.name] = digest
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        assert case["seed"] == seed and case["common_random_numbers"] and case["world_unchanged"]
        rows.extend(dataset_rows(case))
    models = {profile: fit_profile([r for r in rows if r["profile"] == profile])
              for profile in ("ample", "constrained")}
    sources = [Path(__file__), folder/"restricted_forecast.py", folder/"EXPERIMENT_I49.md",
               folder/"conditional_work.py"]
    report = {"schema": "taris-restricted-model/1", "training_seeds": [31, 32, 33],
        "heldout_seeds": [34, 35, 36], "horizon_days": 7., "transfer_price": .1,
        "models": models, "training_input_hashes": hashes,
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "rows": rows, "expected_loss_cost": None, "full_actuarial_reserve": None}
    output = folder/"restricted_model_i49"
    output.mkdir(exist_ok=True)
    path = output/"model.json"
    value = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if path.exists() and path.read_text(encoding="utf-8") != value:
        raise ValueError("Модель уже зафиксирована; не перезаписывать после проверки")
    path.write_text(value, encoding="utf-8")
    print(json.dumps({"models": models, "training_rows": len(rows),
                      "model_hash": hashlib.sha256(path.read_bytes()).hexdigest()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
