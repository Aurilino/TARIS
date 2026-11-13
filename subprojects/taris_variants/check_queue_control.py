"""I6: независимая арифметика отчёта и запаса разных номеров."""
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
    assert isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def main():
    folder = Path(__file__).parent
    source = folder/"queue_evaluation_i6/results.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    assert report["worlds"] == 1 and report["executions"] == 24 and not report["independent_holdout"]
    for name, digest in report["source_hashes"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    raw = {}
    for name, digest in report["input_hashes"].items():
        path = folder/"series_i6_replay"/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                raw[case["queue_control_mode"], profile, variant] = record
    assert len(raw) == len(report["rows"]) == 24
    for row in report["rows"]:
        record = raw[row["mode"], row["profile"], row["variant"]]
        jobs, m = record["jobs"], record["metrics"]
        assert row["losses"] == m["irrecoverable_objects"] and row["unfinished_jobs"] == sum(j["finished"] is None for j in jobs)
        assert row["confirmed_proofs"] == sum(j["status"] == "proof_completed" for j in jobs)
        assert row["restored_map_entries"] == len(record["returns"])
        close(row["parked_owner_days"], row["parked_read_wait_days"]+m["parked_buffer_days"])
        copies = sum((f["physical_end"] if f["physical_end"] is not None else 45.)-f["start"]
                     for f in record["physical_fragments"])
        close(row["resources"]["physical_copy_days"], copies)
        central = sum(r["size"]*((r["end"] if r["end"] is not None else 45.)-r["start"])
                      for r in record["buffers"]+record["decoded_prefixes"])
        receiver = sum(r["size"]*((r["end"] if r["end"] is not None else 45.)-r["start"])
                       for r in record["receiver_prefixes"])
        served = sum(max(0., min(45., s["end"])-s["start"])*24*record["queue_settings"]["rate_size_per_hour"]
                     for s in record["segments"])
        close(row["resources"]["served_volume"], served)
        close(row["costs"]["observed_resource_cost"], copies/365+(central+receiver)*2/365+served*.1)
        by_object = defaultdict(lambda: defaultdict(list))
        for f in record["physical_fragments"]:
            stop = f["physical_end"] if f["physical_end"] is not None else 45.
            if f["start"] < stop:
                by_object[f["obj"]][f["idx"]].append((f["start"], stop))
        below = 0.
        for obj, created, *_ in record["initial_writes"]:
            below += sum(b-a for a, b, rank in rank_timeline(by_object[obj], created, 45.) if rank < 4)
        close(row["stock"]["below_target_object_days"], below)
        assert row["costs"]["expected_loss_cost"] is None and row["costs"]["full_actuarial_reserve"] is None
    result = {"schema": "taris-queue-control-verification/1", "executions": 24,
        "physical_rank_stock_verified": True, "resource_units_verified": True,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results_hash": hashlib.sha256(source.read_bytes()).hexdigest()}
    (folder/"queue_evaluation_i6/verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print("I6: 24 строки, разные номера, экспозиции и стоимость независимо проверены")


if __name__ == "__main__":
    main()
