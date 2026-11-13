"""I6.1: повторный аудит двух режимов служебного fallback."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
from taris_sim.config import Config
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from subprojects.taris_variants.run_queue_drain import provenance
from subprojects.taris_variants.run_compare import table_hash
from subprojects.taris_variants.audit_queue_drain_stress import audit_stress_record
from subprojects.taris_variants.audit_cancellation import audit_cancellations
from subprojects.taris_variants.audit_observed import audit_labels


def main():
    directory = Path(__file__).with_name("series_i61_replay")
    before, rows, cases, hashes = provenance(), [], {}, {}
    defaults = Config()
    for mode in ("original", "pressure"):
        path = directory/f"s103_close_{mode}.json"
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["provenance"] == before
        case = saved["case"]
        assert case["seed"] == 103 and case["drain_mode"] == mode and case["queue_control_mode"] == "combined" and case["stress_mode"] == "close"
        assert case["control_settings"] == {"reserve_proof_slot": True, "bypass_read_wait": True}
        cfg = Config(**{key: tuple(value) if isinstance(getattr(defaults, key), tuple) else value
                        for key, value in case["config"].items()})
        world, chain = World(cfg), None
        chain, data = Chain(world), DataStream(cfg, len(world.nodes))
        assert {key: table_hash(getattr(world, key)) for key in case["world_hashes"]} == case["world_hashes"]
        inputs = {"observed": table_hash(chain.observed), "audits": table_hash(chain.audits),
            "true_down": table_hash(chain.true_down), "objects": table_hash(data.objects),
            "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()}
        assert inputs == case["input_hashes"] and case["world_unchanged"] and case["common_random_numbers"]
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                assert record["control_settings"] == case["control_settings"]
                assert record["drain_irrelevant_proof"] == (mode == "pressure")
                audit_cancellations(record)
                physical = audit_stress_record(record, cfg, world, chain, data)
                assert physical == record["physical_transport_audit"]
                if mode == "original":
                    old_path = directory.parent/"series_i6_replay/s103_close_combined.json"
                    assert record["baseline_identical"]
                    assert record["baseline_input_hash"] == hashlib.sha256(old_path.read_bytes()).hexdigest()
                    previous = json.loads(old_path.read_text(encoding="utf-8"))["case"]["results"][profile][variant]
                    for key in ("jobs", "segments", "block_results", "scheduler_events", "buffers", "physical_fragments",
                                "queue_events", "repair_outcomes", "declarations", "returns", "observations"):
                        assert record[key] == previous[key], key
                rows.append({"mode": mode, "profile": profile, "variant": variant, **physical,
                             "labels": audit_labels(record, cfg.horizon_days)})
        cases[mode] = case
    assert len(rows) == 12
    for case in cases.values():
        for key in ("config", "settings", "world_hashes", "input_hashes", "public_targets", "stress_schedule"):
            assert case[key] == cases["original"][key]
    assert before == provenance()
    assert hashes == {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in hashes}
    report = {"schema": "taris-queue-drain-audit/1", "worlds": 1, "executions": 12,
        "input_hashes": hashes, "common_exogenous_inputs": True, "baseline_history_identical": True,
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows}
    (directory/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("I6.1: 12 исполнений, платный fallback, физика, транспорт и контроль повторно проверены")


if __name__ == "__main__":
    main()
