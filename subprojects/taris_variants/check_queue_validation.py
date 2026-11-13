"""I6.2: независимая арифметика ресурсов, запасов и парных различий."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
from math import isclose
import hashlib
import json
from subprojects.taris_variants.audit_physical import rank_timeline


def close(a, b):
    assert isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def main():
    folder = Path(__file__).parent
    path = folder/"queue_validation_i62/results.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["worlds"] == 3 and report["executions"] == 108 and report["independent_holdout"]
    assert report["audit_hash"] == hashlib.sha256((folder/"series_i62/audit.json").read_bytes()).hexdigest()
    for name, digest in report["source_hashes"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    raw = {}
    for name, digest in report["input_hashes"].items():
        original = folder/"series_i62"/name
        assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
        case = json.loads(original.read_text(encoding="utf-8"))["case"]
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                raw[case["seed"], case["stress_mode"], case["drain_mode"], profile, variant] = record
    assert len(raw) == len(report["rows"]) == 108
    groups, lookup = defaultdict(list), {}
    for row in report["rows"]:
        key = row["seed"], row["stress_mode"], row["mode"], row["profile"], row["variant"]
        assert key not in lookup
        lookup[key] = row
        record = raw[key]
        m = record["metrics"]
        assert row["losses"] == m["irrecoverable_objects"]
        assert row["lost_admitted_volume"] == m["lost_admitted_objects"]
        assert row["initial_incomplete_objects"] == m["initial_incomplete_objects"]
        assert row["unfinished_jobs"] == sum(j["finished"] is None for j in record["jobs"])
        assert row["confirmed_proofs"] == sum(j["status"] == "proof_completed" for j in record["jobs"])
        assert row["restored_map_entries"] == len(record["returns"])
        assert row["waiting_read_jobs_at_end"] == sum(j["finished"] is None and j["status"] == "waiting_read_sources"
                                                     for j in record["jobs"])
        assert row["pressure_fallback_starts"] == sum(s.get("selection_rule") == "pressure_fallback" for s in record["proof_selection_log"])
        copies = sum((f["physical_end"] if f["physical_end"] is not None else 45.)-f["start"] for f in record["physical_fragments"])
        central = sum(r["size"]*((r["end"] if r["end"] is not None else 45.)-r["start"])
                      for r in record["buffers"]+record["decoded_prefixes"])
        receiver = sum(r["size"]*((r["end"] if r["end"] is not None else 45.)-r["start"])
                       for r in record["receiver_prefixes"])
        served = sum(max(0., min(45., s["end"])-s["start"])*24*record["queue_settings"]["rate_size_per_hour"]
                     for s in record["segments"])
        close(row["resources"]["served_volume"], served)
        close(row["resources"]["physical_copy_days"], copies)
        close(row["costs"]["observed_resource_cost"], copies/365+(central+receiver)*2/365+served*.1)
        by_object = defaultdict(list)
        for fragment in record["physical_fragments"]:
            by_object[fragment["obj"]].append(fragment)
        losses = {r["obj"]: r["t"] for r in record["losses"]}
        below, preloss_below = 0., 0.
        for obj, created, *_ in record["initial_writes"]:
            for stop, preloss in ((45., False), (losses.get(obj, 45.), True)):
                intervals = defaultdict(list)
                for f in by_object[obj]:
                    end = min(stop, f["physical_end"] if f["physical_end"] is not None else stop)
                    if f["start"] < end:
                        intervals[f["idx"]].append((f["start"], end))
                value = sum(b-a for a, b, rank in rank_timeline(intervals, created, stop) if rank < 4)
                if preloss:
                    preloss_below += value
                else:
                    below += value
        close(row["stock"]["below_target_object_days"], below)
        close(row["preloss_stock"]["preloss_below_target_days"], preloss_below)
        assert row["costs"]["expected_loss_cost"] is None and row["costs"]["full_actuarial_reserve"] is None
        groups[row["stress_mode"], row["mode"], row["profile"], row["variant"]].append(row)
    assert len(groups) == len(report["summary"]) == 36
    for row in report["summary"]:
        group = groups[row["stress_mode"], row["mode"], row["profile"], row["variant"]]
        assert sorted(r["seed"] for r in group) == [201, 202, 203]
        for stored, source in (("mean_losses", "losses"), ("mean_lost_admitted_volume", "lost_admitted_volume"),
            ("mean_unfinished_jobs", "unfinished_jobs"), ("mean_restored_map_entries", "restored_map_entries")):
            close(row[stored], sum(r[source] for r in group)/3)
        close(row["mean_served_volume"], sum(r["resources"]["served_volume"] for r in group)/3)
        close(row["mean_resource_cost"], sum(r["costs"]["observed_resource_cost"] for r in group)/3)
        close(row["mean_below_target_days"], sum(r["stock"]["below_target_object_days"] for r in group)/3)
        close(row["mean_preloss_below_target_days"], sum(r["preloss_stock"]["preloss_below_target_days"] for r in group)/3)
        assert row["loss_range"] == [min(r["losses"] for r in group), max(r["losses"] for r in group)]
        assert row["total_pressure_fallback_starts"] == sum(r["pressure_fallback_starts"] for r in group)
        assert row["total_waiting_read_jobs_at_end"] == sum(r["waiting_read_jobs_at_end"] for r in group)
    assert len(report["paired_contrasts"]) == 72
    for row in report["paired_contrasts"]:
        prefix = row["seed"], row["stress_mode"]
        current = lookup[*prefix, "pressure", row["profile"], row["variant"]]
        base_mode = row["comparison"].removeprefix("pressure_minus_")
        assert base_mode in ("legacy", "combined")
        previous = lookup[*prefix, base_mode, row["profile"], row["variant"]]
        close(row["loss_difference"], current["losses"]-previous["losses"])
        close(row["unfinished_difference"], current["unfinished_jobs"]-previous["unfinished_jobs"])
        close(row["below_target_difference"], current["stock"]["below_target_object_days"]-previous["stock"]["below_target_object_days"])
        close(row["resource_cost_difference"], current["costs"]["observed_resource_cost"]-previous["costs"]["observed_resource_cost"])
    assert report["expected_loss_cost"] is None and report["full_actuarial_reserve"] is None
    plot = json.loads((folder/"queue_validation_i62/plot_metadata.json").read_text(encoding="utf-8"))
    assert plot["source_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert plot["plotter_hash"] == hashlib.sha256((folder/"plot_queue_validation.py").read_bytes()).hexdigest()
    verification = {"schema": "taris-queue-validation-verification/1", "worlds": 3, "executions": 108,
        "summary_rows": 36, "paired_contrasts": 72, "unknowns_preserved": True,
        "plot_inputs_verified": True,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results_hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder/"queue_validation_i62/verification.json").write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print("I6.2: 108строк, ресурсы, запасы,36средних и72парных различия независимо проверены")


if __name__ == "__main__":
    main()
