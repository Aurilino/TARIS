"""I4.10: обучение и gate до новых отложенных миров."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
from subprojects.taris_variants.state_forecast import train_gated


def main():
    folder = Path(__file__).parent
    path = folder/"restricted_diagnosis_i49/results.json"
    diagnosis = json.loads(path.read_text(encoding="utf-8"))
    assert diagnosis["seeds"] == [31, 32, 33] and not diagnosis["validation_used"]
    assert diagnosis["source_hash"] == hashlib.sha256((folder/"diagnose_restricted.py").read_bytes()).hexdigest()
    models = {profile: train_gated([r for r in diagnosis["rows"] if r["profile"] == profile])
              for profile in ("ample", "constrained")}
    files = [Path(__file__), folder/"state_forecast.py", folder/"restricted_forecast.py",
             folder/"diagnose_restricted.py", folder/"EXPERIMENT_I410.md"]
    result = {"schema": "taris-state-model/1", "training_seeds": [31, 32, 33], "heldout_seeds": [37, 38, 39],
        "training_diagnosis_hash": hashlib.sha256(path.read_bytes()).hexdigest(), "models": models,
        "source_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "expected_loss_cost": None, "full_actuarial_reserve": None}
    output = folder/"state_model_i410"
    output.mkdir(exist_ok=True)
    target = output/"model.json"
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise ValueError("Зафиксированную модель нельзя менять после проверки")
    target.write_text(content, encoding="utf-8")
    for profile, model in models.items():
        print(profile, "gate=", model["use_state_model"], "CV:", model["cv_baseline_mse"], model["cv_candidate_mse"])
        print({k: (v["coefficient"], v["support_rows"]) for k, v in model["models"].items()})
    print("SHA256", hashlib.sha256(target.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
