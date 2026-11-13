"""I5.1/2: независимая арифметика ресурсов, средних и условных границ."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import defaultdict
from math import isclose
import hashlib
import json


def close(a, b):
    assert isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), (a, b)


def main():
    folder = Path(__file__).parent
    path = folder/"stress_sensitivity_i51/results.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    for name, digest in report["source_hashes"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    assert report["audit_hash"] == hashlib.sha256((folder/"series_i51/audit.json").read_bytes()).hexdigest()
    raw_rows = {}
    for name, digest in report["input_hashes"].items():
        raw_path = folder/"series_i51"/name
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == digest
        case = json.loads(raw_path.read_text(encoding="utf-8"))["case"]
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                raw_rows[case["seed"], case["stress_mode"], profile, variant] = record
    assert len(report["rows"]) == len(raw_rows) == 150
    groups = defaultdict(list)
    for row in report["rows"]:
        record = raw_rows[row["seed"], row["mode"], row["profile"], row["variant"]]
        m, usage = record["metrics"], row["resources"]
        assert row["losses"] == m["irrecoverable_objects"]
        assert row["lost_admitted_volume"] == m["lost_admitted_objects"]
        assert row["objects"] == m["objects"] and row["unfinished_jobs"] == m["unfinished_jobs"]
        close(row["unavailable_object_days"], m["unavailable_object_days"])
        copy_days = sum(max(0., min(45., f["physical_end"] if f["physical_end"] is not None else 45.)-f["start"])
                        for f in record["physical_fragments"])
        close(usage["physical_copy_days"], copy_days)
        size_days = {}
        for name, key in (("buffers", "full_buffer_size_days"), ("decoded_prefixes", "decoded_prefix_size_days"),
                          ("receiver_prefixes", "receiver_prefix_size_days")):
            value = sum(r["size"]*max(0., min(45., r["end"] if r["end"] is not None else 45.)-r["start"])
                        for r in record[name])
            close(usage[key], value)
            size_days[key] = value
        close(usage["central_size_days"], size_days["full_buffer_size_days"]+size_days["decoded_prefix_size_days"])
        served = sum(max(0., min(45., s["end"])-s["start"])*24.*record["queue_settings"]["rate_size_per_hour"]
                     for s in record["segments"])
        close(usage["served_volume"], served)
        for ratio in (1., 10.):
            expected = usage["physical_copy_days"]/365 + usage["central_size_days"]*2/365*ratio \
                + usage["receiver_prefix_size_days"]*2/365 + usage["served_volume"]*.1
            close(row["costs"][str(ratio)]["observed_resource_cost"], expected)
            assert row["costs"][str(ratio)]["expected_loss_cost"] is None
        if row["mode"] != "control":
            assert row["losses_at_first_wave"] == sum(r["t"] == 15. for r in record["losses"])
            second = record["stress_schedule"][1]["time"]
            assert row["losses_at_second_wave"] == sum(r["t"] == second for r in record["losses"])
            assert set(row["first_wave_exposure"]["immediately_vulnerable_objects"]) == {
                r["obj"] for r in record["losses"] if r["t"] == 15.}
        groups[row["mode"], row["profile"], row["variant"]].append(row)
    assert len(groups) == len(report["summary"]) == 30
    first_wave = {}
    for row in report["rows"]:
        if row["mode"] == "control":
            continue
        key = row["seed"], row["profile"], row["variant"]
        if key in first_wave:
            assert first_wave[key] == row["losses_at_first_wave"], "Будущая вторая волна повлияла на первый исход"
        first_wave[key] = row["losses_at_first_wave"]
    summaries = {}
    for row in report["summary"]:
        key = row["mode"], row["profile"], row["variant"]
        group = groups[key]
        assert sorted(r["seed"] for r in group) == [101, 102, 103, 104, 105]
        assert row["worlds"] == 5
        close(row["mean_losses"], sum(r["losses"] for r in group)/5)
        close(row["mean_lost_admitted_volume"], sum(r["lost_admitted_volume"] for r in group)/5)
        close(row["mean_losses_at_first_wave"], sum(r["losses_at_first_wave"] for r in group)/5)
        close(row["mean_losses_at_second_wave"], sum(r["losses_at_second_wave"] for r in group)/5)
        close(row["mean_served_volume"], sum(r["resources"]["served_volume"] for r in group)/5)
        close(row["mean_preloss_bare_code_days"], sum(r["preloss_stock"]["preloss_bare_code_days"] for r in group)/5)
        close(row["mean_unfinished_jobs"], sum(r["unfinished_jobs"] for r in group)/5)
        assert row["min_losses"] == min(r["losses"] for r in group)
        assert row["max_losses"] == max(r["losses"] for r in group)
        for ratio in (1., 10.):
            close(row["mean_resource_cost"][str(ratio)], sum(r["costs"][str(ratio)]["observed_resource_cost"] for r in group)/5)
        summaries[key] = row
    assert report["supplied_weights"] is None and report["supplied_loss_price"] is None
    assert report["full_actuarial_reserve"] is None and not report["real_world_scenario_probabilities_estimated"]
    assert len(report["conditional_mixtures"]) == 12
    for mixture in report["conditional_mixtures"]:
        assert mixture["conditional_total_cost"] is None and mixture["expected_lost_volume"] is None
        assert mixture["expected_resource_cost"] is None and mixture["full_actuarial_reserve"] is None
    assert len(report["conditional_branch_crossings"]) == 15
    for crossing in report["conditional_branch_crossings"]:
        a = summaries[crossing["mode"], "ample", crossing["variant"]]
        b = summaries[crossing["mode"], "constrained", crossing["variant"]]
        difference = a["mean_lost_admitted_volume"]-b["mean_lost_admitted_volume"]
        cost_diff = a["mean_resource_cost"]["1.0"]-b["mean_resource_cost"]["1.0"]
        if difference == 0:
            assert crossing["loss_price"] is None
        else:
            close(crossing["loss_price"], -cost_diff/difference)
            price = crossing["loss_price"]
            close(a["mean_resource_cost"]["1.0"]+price*a["mean_lost_admitted_volume"],
                  b["mean_resource_cost"]["1.0"]+price*b["mean_lost_admitted_volume"])
    verification = {"schema": "taris-stress-sensitivity-verification/1", "executions": 150,
        "worlds": 5, "summary_rows": 30, "conditional_crossings": 15, "unknowns_preserved": True,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results_hash": hashlib.sha256(path.read_bytes()).hexdigest()}
    (folder/"stress_sensitivity_i51/verification.json").write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    print("I5.1/2: 150 строк, 30 средних, 15 условных границ и неизвестные компоненты проверены")


if __name__ == "__main__":
    main()
