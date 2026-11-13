"""I4.10: независимый аудит поддержки, CV gate и отложенных ошибок."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
from math import sqrt
from collections import defaultdict
from subprojects.taris_variants.check_restricted import close, metrics


def verify_models(models, rows):
    for state, model in models.items():
        subset = [r for r in rows if r["state"] == state and r["known_work"] > 0]
        seeds = sorted({r["seed"] for r in subset})
        assert model["support_rows"] == len(subset) and model["support_worlds"] == seeds
        if len(seeds) < 2 or len(subset) < 5:
            assert model["coefficient"] == 1. and model["status"] == "insufficient_support"
            continue
        losses = []
        for tick in range(401):
            world_scores = []
            for seed in seeds:
                group = [r for r in subset if r["seed"] == seed]
                world_scores.append(sum((min(tick/100*r["known_work"], r["capacity"])-r["actual_window_volume"])**2
                                        for r in group)/len(group))
            losses.append(sum(world_scores)/len(world_scores))
        index = min(range(401), key=losses.__getitem__)
        close(model["coefficient"], index/100)
        close(model["training_world_mean_mse"], losses[index])


def independent_state(snapshot):
    if not any(j["kind"] == "repair" for j in snapshot["jobs"]):
        return "empty_repair_queue"
    if snapshot["parked_job"] is not None and len(snapshot["full_buffer_objects"]) > 0:
        return "parked_full_buffer"
    phase = snapshot["planned_channel_phase"]
    name = phase["phase"] if phase else None
    return {"read_block": "repair_transfer", "write_block": "repair_transfer", "close_proof": "repair_transfer",
            "proof": "proof_transfer"}.get(name, "waiting_or_setup")


def main():
    folder = Path(__file__).parent
    model_path = folder/"state_model_i410/model.json"
    path = folder/"state_evaluation_i410/results.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    report = json.loads(path.read_text(encoding="utf-8"))
    assert model["training_seeds"] == [31, 32, 33] and model["heldout_seeds"] == [37, 38, 39]
    assert report["model_hash"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert report["source_hash"] == hashlib.sha256((folder/"report_state.py").read_bytes()).hexdigest()
    for name, digest in model["source_hashes"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    diagnosis_path = folder/"restricted_diagnosis_i49/results.json"
    assert model["training_diagnosis_hash"] == hashlib.sha256(diagnosis_path.read_bytes()).hexdigest()
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    for profile, fitted in model["models"].items():
        rows = [r for r in diagnosis["rows"] if r["profile"] == profile]
        verify_models(fitted["models"], rows)
        b, c = [], []
        for fold in fitted["folds"]:
            heldout = fold["heldout_seed"]
            verify_models(fold["models"], [r for r in rows if r["seed"] != heldout])
            test = [r for r in rows if r["seed"] == heldout]
            mse_b = sum((min(r["known_work"], r["capacity"])-r["actual_window_volume"])**2 for r in test)/len(test)
            mse_c = sum((min(fold["models"][r["state"]]["coefficient"]*r["known_work"], r["capacity"])
                         -r["actual_window_volume"])**2 for r in test)/len(test)
            close(fold["baseline_mse"], mse_b)
            close(fold["candidate_mse"], mse_c)
            b.append(mse_b)
            c.append(mse_c)
        close(fitted["cv_baseline_mse"], sum(b)/3)
        close(fitted["cv_candidate_mse"], sum(c)/3)
        assert fitted["use_state_model"] == (sum(c)/3 < sum(b)/3-1e-12)
    lookup = {(r["seed"], r["profile"], r["variant"], r["time"]): r for r in report["rows"]}
    assert len(lookup) == len(report["rows"]) == 702
    for name, digest in report["validation_input_hashes"].items():
        raw = folder/"series_i410_validation"/name
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == digest
        case = json.loads(raw.read_text(encoding="utf-8"))["case"]
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                for snapshot in record["observations"]:
                    t = snapshot["time"]
                    if t+7 > 45:
                        assert (case["seed"], profile, variant, t) not in lookup
                        continue
                    row = lookup[case["seed"], profile, variant, t]
                    state = independent_state(snapshot)
                    assert state == row["state"]
                    ids = {j["id"] for j in snapshot["jobs"] if j["kind"] == "repair"}
                    actual = 0.
                    for segment in record["segments"]:
                        if segment["job"] not in ids:
                            continue
                        a, b = segment["start"], segment["end"]
                        if b <= t or a >= t+7:
                            continue
                        actual += segment["volume"]*(1-max(0., t-a)/(b-a)-max(0., b-t-7)/(b-a))
                    close(row["actual_window_volume"], actual)
                    stored = next(r for r in record["conditional_work"] if r["time"] == t)["scenarios"][0]
                    close(row["known_work"], stored["known_repair_volume"])
                    close(row["capacity"], snapshot["queue_settings"]["rate_size_per_hour"]*168)
                    fitted = model["models"][profile]
                    coefficient = fitted["models"][state]["coefficient"] if fitted["use_state_model"] else 1.
                    close(row["baseline_prediction"], min(row["known_work"], row["capacity"]))
                    close(row["calibrated_prediction"], min(coefficient*row["known_work"], row["capacity"]))
    groups = defaultdict(list)
    for row in report["rows"]:
        groups[row["seed"], row["profile"], row["variant"]].append(row)
    for stored in report["per_world"]:
        rows = groups[stored["seed"], stored["profile"], stored["variant"]]
        for mode in ("baseline", "calibrated"):
            for prefix, subset in (("", rows), ("nonempty_", [r for r in rows if r["known_work"] > 0])):
                expected = metrics(subset, mode+"_prediction")
                if expected is None:
                    assert stored[prefix+mode] is None
                else:
                    for key, value in expected.items():
                        close(stored[prefix+mode][key], value)
    for row in report["summary"]:
        group = [r for r in report["per_world"] if (r["profile"], r["variant"]) == (row["profile"], row["variant"])]
        assert len(group) == 3
        for mode in ("baseline", "calibrated"):
            close(row[mode]["mae"], sum(r[mode]["mae"] for r in group)/3)
            close(row[mode]["bias"], sum(r[mode]["bias"] for r in group)/3)
            close(row[mode]["rmse"], sqrt(sum(r[mode]["rmse"]**2 for r in group)/3))
    assert report["expected_loss_cost"] is None and report["full_actuarial_reserve"] is None
    verification = {"schema": "taris-state-verification/1", "validation_windows": 702,
        "cv_folds_verified": 6, "profiles_verified": 2, "model_hash": report["model_hash"],
        "results_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "checker_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/"state_evaluation_i410/verification.json").write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print("I4.10: поддержка, 6 CV-fold, gate, 702 окна и ошибки независимо проверены")


if __name__ == "__main__":
    main()
