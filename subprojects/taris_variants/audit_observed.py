"""I4.8: независимый аудит времён и posthoc-меток, не вызывает label_snapshots."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import argparse
import hashlib
import json
from math import isclose
from subprojects.taris_variants.run_observed import provenance


def close(a, b):
    if not isclose(a, b, rel_tol=1e-9, abs_tol=1e-10):
        raise AssertionError((a, b))


def audit_labels(record, end):
    snapshots, labels = record["observations"], record["posthoc_labels"]
    assert [row["time"] for row in snapshots] == list(map(float, range(int(end))))
    lookup = {(r["time"], r["job"]): r for r in labels}
    assert len(lookup) == len(labels) == sum(len(s["jobs"]) for s in snapshots)
    for snapshot in snapshots:
        t = snapshot["time"]
        assert snapshot["schema"] == "taris-queue-observation/2"
        assert set(snapshot) == {"schema", "time", "jobs", "waiting", "active_job", "parked_job",
            "full_buffer_objects", "planned_channel_phase", "block_acknowledgements", "queue_settings",
            "transport_settings", "object_sizes", "k", "local_memory", "closing_acknowledgements", "memory_notifications"}
        assert all(row["time"] < t for key in ("block_acknowledgements", "closing_acknowledgements", "memory_notifications")
                   for row in snapshot[key])
        assert all(row["confirmed_at"] < t for row in snapshot["local_memory"])
        for job in snapshot["jobs"]:
            assert set(job) <= {"id", "kind", "submitted", "started", "obj", "node", "indices", "nodes"}
            final = record["jobs"][job["id"]]
            assert job["submitted"] <= t
            finish = final["finished"]
            assert finish is None or finish >= t
            observed_end = min(finish, end) if finish is not None else end
            censored = finish is None or finish >= end
            row = lookup[t, job["id"]]
            assert row["censored"] == censored
            assert row["kind"] == job["kind"]
            # Полная фаза минус часть до t минус часть после наблюдаемого конца.
            volume = 0.
            for segment in record["segments"]:
                if segment["job"] != job["id"]:
                    continue
                a, b = segment["start"], segment["end"]
                if b <= t or a >= observed_end:
                    continue
                unit = segment["volume"]/(b-a)
                volume += segment["volume"]-unit*max(0., t-a)-unit*max(0., b-observed_end)
            close(row["observed_future_volume"], volume)
            close(row["observed_followup_days"], observed_end-t)
            failures = 0
            for result in record["block_results"]:
                segment = record["segments"][result["segment"]]
                if (segment["job"] == job["id"] and not result["ok"] and segment["end"] >= t
                        and (segment["end"] < end if censored else segment["end"] <= finish)):
                    failures += 1
            assert row["future_failed_blocks_observed"] == failures
            if censored:
                assert row["full_future_volume"] is None and row["time_to_completion_days"] is None
                assert row["final_status"] is None
            else:
                close(row["full_future_volume"], volume)
                close(row["time_to_completion_days"], finish-t)
                assert row["final_status"] == final["status"]
    censored = [r for r in labels if r["censored"]]
    return {"snapshots": len(snapshots), "job_snapshot_rows": len(labels),
        "distinct_observed_jobs": len({r["job"] for r in labels}),
        "censored_rows": len(censored), "censored_positive_work_rows": sum(r["observed_future_volume"] > 0 for r in censored),
        "observed_repair_rows": sum(r["kind"] == "repair" for r in labels)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).with_name("series_i48_pilot"))
    args = parser.parse_args()
    before, rows = provenance(), []
    paths = sorted(args.directory.glob("s[0-9]*.json"))
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    for path in paths:
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["provenance"] == before
        case = saved["case"]
        assert case["world_unchanged"] and case["common_random_numbers"]
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                if case["seed"] == 30:
                    assert record["baseline_identical"]
                assert record["physical_transport_audit"]["objects"] == record["metrics"]["objects"]
                rows.append({"seed": case["seed"], "profile": profile, "variant": variant,
                    **audit_labels(record, case["config"]["horizon_days"])})
    assert rows and before == provenance()
    assert hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = {"schema": "taris-observed-label-audit/1", "input_hashes": hashes,
        "audit_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows, "temporal_whitelist_verified": True, "labels_independently_verified": True,
        "unit_of_independence": "world_not_job_snapshot"}
    (args.directory/"label_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
