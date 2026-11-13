"""Независимая арифметика I4.9 без импорта прогнозной модели."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
from math import isclose, sqrt
from collections import defaultdict


def close(a, b):
    assert isclose(a, b, rel_tol=1e-9, abs_tol=1e-10), (a, b)


def metrics(rows, field):
    if not rows:
        return None
    errors = [r[field]-r["actual_window_volume"] for r in rows]
    return {"n": len(rows), "mae": sum(map(abs, errors))/len(rows),
            "rmse": sqrt(sum(x*x for x in errors)/len(rows)), "bias": sum(errors)/len(rows)}


def check():
    folder = Path(__file__).parent
    model_path = folder/"restricted_model_i49/model.json"
    result_path = folder/"restricted_evaluation_i49/results.json"
    model = json.loads(model_path.read_text(encoding="utf-8"))
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["model_hash"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    for name, digest in model["source_hashes"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    assert report["source_hash"] == hashlib.sha256((folder/"report_restricted.py").read_bytes()).hexdigest()
    assert model["training_seeds"] == [31, 32, 33] and model["heldout_seeds"] == [34, 35, 36]
    tracked = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (model_path, result_path)}
    for role, rows, hashes in (("train", model["rows"], model["training_input_hashes"]),
                              ("validation", report["rows"], report["validation_input_hashes"])):
        lookup = {(r["seed"], r["profile"], r["variant"], r["time"]): r for r in rows}
        assert len(lookup) == len(rows) == 702
        checked = 0
        for name, digest in hashes.items():
            path = folder/f"series_i48_{role}"/name
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            tracked[str(path)] = digest
            case = json.loads(path.read_text(encoding="utf-8"))["case"]
            for profile, variants in case["results"].items():
                for variant, record in variants.items():
                    saved_work = {r["time"]: r for r in record["conditional_work"]}
                    for snapshot in record["observations"]:
                        t = snapshot["time"]
                        if t+7 > case["config"]["horizon_days"]:
                            assert (case["seed"], profile, variant, t) not in lookup
                            continue
                        row = lookup[case["seed"], profile, variant, t]
                        known = saved_work[t]["scenarios"][0]
                        assert known["mode"] == "trust_ack"
                        close(row["known_work"], known["known_repair_volume"])
                        ids = {j["id"] for j in snapshot["jobs"] if j["kind"] == "repair"}
                        assert row["current_repair_jobs"] == len(ids)
                        actual = 0.
                        for segment in record["segments"]:
                            if segment["job"] not in ids:
                                continue
                            a, b = segment["start"], segment["end"]
                            # Разбиение фазы на отрезки; интеграл только внутри окна.
                            cuts = sorted({a, b} | {x for x in (t, t+7) if a < x < b})
                            for left, right in zip(cuts, cuts[1:]):
                                if t <= (left+right)/2 < t+7:
                                    actual += segment["volume"]*(right-left)/(b-a)
                        close(row["actual_window_volume"], actual)
                        close(row["capacity"], snapshot["queue_settings"]["rate_size_per_hour"]*168)
                        assert row["horizon_days"] == 7.
                        if role == "validation":
                            close(row["baseline_prediction"], min(row["known_work"], row["capacity"]))
                            coefficient = model["models"][profile]["coefficient"]
                            close(row["calibrated_prediction"], min(coefficient*row["known_work"], row["capacity"]))
                        checked += 1
        assert checked == len(rows)
    for profile, fitted in model["models"].items():
        rows = [r for r in model["rows"] if r["profile"] == profile]
        losses = []
        for tick in range(401):
            world_losses = []
            for seed in model["training_seeds"]:
                world = [r for r in rows if r["seed"] == seed]
                world_losses.append(sum((min(tick/100*r["known_work"], r["capacity"])-r["actual_window_volume"])**2
                                        for r in world)/len(world))
            losses.append(sum(world_losses)/len(world_losses))
        winner = min(range(401), key=losses.__getitem__)
        close(fitted["coefficient"], winner/100)
        close(fitted["training_world_mean_mse"], losses[winner])
    groups = defaultdict(list)
    for row in report["rows"]:
        groups[row["seed"], row["profile"], row["variant"]].append(row)
    for row in report["per_world"]:
        group = groups[row["seed"], row["profile"], row["variant"]]
        assert row["eligible_snapshots"] == len(group) == 39 and row["late_excluded_snapshots"] == 6
        nonempty = [r for r in group if r["known_work"] > 0]
        assert row["nonempty_snapshots"] == len(nonempty)
        for mode in ("baseline", "calibrated"):
            for prefix, subset in (("", group), ("nonempty_", nonempty)):
                expected = metrics(subset, mode+"_prediction")
                stored = row[prefix+mode]
                if expected is None:
                    assert stored is None
                else:
                    for key, value in expected.items():
                        close(stored[key], value)
    for row in report["summary"]:
        group = [r for r in report["per_world"] if (r["profile"], r["variant"]) == (row["profile"], row["variant"])]
        assert row["worlds"] == len(group) == 3
        for mode in ("baseline", "calibrated"):
            close(row[mode]["mae"], sum(r[mode]["mae"] for r in group)/3)
            close(row[mode]["bias"], sum(r[mode]["bias"] for r in group)/3)
            close(row[mode]["rmse"], sqrt(sum(r[mode]["rmse"]**2 for r in group)/3))
    assert report["expected_loss_cost"] is None and report["full_actuarial_reserve"] is None
    assert tracked == {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in tracked}
    verification = {"schema": "taris-restricted-verification/1", "training_rows": 702, "validation_rows": 702,
        "worlds": 6, "grid_fits_verified": 2, "per_world_rows_verified": 18, "summary_rows_verified": 6,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "inputs": tracked,
        "unknowns_preserved": True, "independent_window_integration": True}
    (folder/"restricted_evaluation_i49/verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("I4.9: независимо проверены 1404 окна, 2 подгонки, 18 строк миров и 6 сводок")


if __name__ == "__main__":
    check()
