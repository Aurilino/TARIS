"""Проверка связности сохранённых решений, аудита и исходников I3.2."""
import hashlib
import json
from pathlib import Path


def main():
    folder = Path(__file__).resolve().parent
    root = folder.parents[1]
    output = folder / "series_i32"
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    summary, audit = read(output / "summary.json"), read(output / "audit.json")
    assert summary["completed_cases"] == 12 and not summary["failures"]
    assert summary["integrity_passed"] and summary["paired_profiles"]
    assert audit["main_source_unchanged"] and audit["worlds_verified"]
    assert audit["worlds"] == summary["distinct_worlds"] == 3
    provenance = summary["provenance"]
    for name, digest in provenance["main"].items():
        assert hashlib.sha256((root / "taris_sim" / name).read_bytes()).hexdigest() == digest
    for name, digest in provenance["integration"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((folder / "audit_sizing.py").read_bytes()).hexdigest() == audit["audit_source_hash"]
    indexed = {(r["seed"], r["profile"], r["variant"], r["obj"]): r for r in audit["rows"]}
    assert len(indexed) == len(audit["rows"]) == audit["placements"] == 1620
    checked, guarded = 0, 0
    for path in sorted(output.glob("s*_*.json")):
        saved = read(path)
        assert saved["status"] == "completed" and saved["provenance"] == provenance
        case = saved["case"]
        assert case["common_random_numbers"] and case["world_unchanged"]
        for variant, result in case["variants"].items():
            assert result["objects_requested"] == result["objects_admitted"] == 45
            assert result["initial_rejected_objects"] == result["lost_admitted_objects"] == 0
            for d in result["placement_decisions"]:
                key = case["seed"], case["profile"], variant, d["obj"]
                row = indexed[key]
                selected = d["selected"]
                assert len(set(d["nodes"])) == d["n"] == selected["n"]
                assert selected["nodes"] == d["nodes"]
                assert row["n"] == d["n"] and row["estimated_passed"] == d["requirements_passed"]
                assert row["estimated_window_loss"] == selected["estimated_window_loss"]
                assert 0 <= row["true_window_loss_lower"] <= row["true_window_loss_upper"] <= 1
                assert row["true_exceeded"] == (row["true_window_loss_lower"] > d["window_loss_limit"])
                if d["requirements_passed"]:
                    assert d["scored"] and selected["estimated_window_loss"] <= d["window_loss_limit"]
                    assert selected["estimated_unavailability"] <= d["availability_limit"]
                if case["profile"] == "guard":
                    assert selected["object_confidence_level"] is None
                    assert selected["estimated_unavailability"] >= selected["point_unavailability"]
                    point = selected["point_window_loss"]
                    if point is not None:
                        assert selected["estimated_window_loss"] >= point
                    guarded += 1
                checked += 1
    assert checked == 1620 and guarded == 405
    report = {"placements_checked": checked, "guarded_checked": guarded,
              "source_hashes_verified": True, "audit_linked": True,
              "guard_never_lowers_point_estimates": True}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
