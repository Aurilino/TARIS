"""Проверка сохранённой стоимости отдельно от функции observed_charges."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
import hashlib
import json
import math


def main():
    folder = Path(__file__).resolve().parent
    directory = folder/"resource_cost_evaluation_i45b"
    path = directory/"results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["scope"] == "observed_resources_only" and data["legacy_metrics_verified"]
    assert data["unknown_expected_loss_cost"] and data["unknown_future_liability"] and data["discount_rate"] == 0.
    for name, digest in {**data["input_hashes"], **data["source_hashes"]}.items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    reserve_path = folder/"object_evaluation_i45a"/"results.json"
    assert hashlib.sha256(reserve_path.read_bytes()).hexdigest() == data["reserve_report_hash"]
    resources = {(r["cohort"], r["seed"], r["profile"], r["variant"]): r for r in data["resources"]}
    assert len(resources) == 72 and len(data["charges"]) == 432 and len(data["summary"]) == 144
    k, size = data["coding_units"]["k"], data["coding_units"]["object_size"]
    grouped, seen = defaultdict(list), set()
    for row in data["charges"]:
        key = row["cohort"], row["seed"], row["profile"], row["variant"]
        r, prices = resources[key], row["prices"]
        cell = (*key, prices["central_ratio"], prices["receiver_ratio"])
        assert cell not in seen
        seen.add(cell)
        v, tr = prices["fragment_day"], prices["transfer_unit"]
        expected = (v*r["physical_copy_days"] + tr*r["served_volume"] + v*k/size*(
            prices["central_ratio"]*(r["full_buffer_size_days"]+r["decoded_prefix_size_days"])
            +prices["receiver_ratio"]*r["receiver_prefix_size_days"]))
        assert math.isclose(expected, row["observed_resource_cost"], rel_tol=1e-12, abs_tol=1e-10)
        assert math.isclose(sum(row[x] for x in ("disk", "full_buffer", "decoded_prefix", "receiver_prefix", "channel")), expected,
                            rel_tol=1e-12, abs_tol=1e-10)
        assert row["expected_loss_cost"] is None and row["future_liability_cost"] is None and row["full_actuarial_reserve"] is None
        grouped[(row["cohort"],row["profile"],row["variant"],prices["central_ratio"],prices["receiver_ratio"])].append(row)
    for row in data["summary"]:
        key = row["cohort"],row["profile"],row["variant"],row["central_ratio"],row["receiver_ratio"]
        cases = grouped[key]
        assert len(cases) == 3
        for field in ("observed_resource_cost", "objects", "unfinished_jobs", "open_buffers", "below_target_object_days",
                      "target_shortfall_fragment_days", "bare_code_object_days", "irrecoverable_objects"):
            assert math.isclose(row[field], sum(r[field] for r in cases), rel_tol=1e-12, abs_tol=1e-10)
    plot = json.loads((directory/"plot_metadata.json").read_text(encoding="utf-8"))
    assert plot["source_report_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert plot["plot_source_hash"] == hashlib.sha256((folder/"plot_resource_costs.py").read_bytes()).hexdigest()
    report = {"runs_checked": 72, "price_calculations_checked": 432, "summaries_checked": 144,
        "original_inputs_unchanged": True, "units_and_arithmetic_verified": True,
        "unknown_loss_and_future_liability_preserved": True, "plot_input_verified": True,
        "verifier_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (directory/"verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
