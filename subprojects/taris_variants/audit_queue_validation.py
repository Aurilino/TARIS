"""I6.2: независимые миры, неизменные правила и повторный аудит108исполнений."""
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
from subprojects.taris_variants.run_queue_validation import provenance
from subprojects.taris_variants.run_compare import table_hash
from subprojects.taris_variants.disk_stress import public_wave_targets
from subprojects.taris_variants.audit_queue_drain_stress import audit_stress_record
from subprojects.taris_variants.audit_cancellation import audit_cancellations
from subprojects.taris_variants.audit_observed import audit_labels


def main():
    folder = Path(__file__).parent
    directory = folder/"series_i62"
    before = provenance()
    frozen_path = folder/"series_i61_replay/s103_close_pressure.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))["provenance"]
    for section, sources in frozen.items():
        for name, digest in sources.items():
            assert before[section][name] == digest, ("Изменён зафиксированный источник", section, name)
    rows, cases, hashes, cache = [], {}, {}, {}
    defaults = Config()
    for seed in (201, 202, 203):
        for stress in ("close", "far"):
            for mode in ("legacy", "combined", "pressure"):
                path = directory/f"s{seed}_{stress}_{mode}.json"
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                saved = json.loads(path.read_text(encoding="utf-8"))
                assert saved["provenance"] == before
                case = saved["case"]
                assert (case["seed"], case["stress_mode"], case["drain_mode"]) == (seed, stress, mode)
                controls = {"reserve_proof_slot": mode != "legacy", "bypass_read_wait": mode != "legacy"}
                assert case["control_settings"] == controls
                if seed not in cache:
                    cfg = Config(**{key: tuple(value) if isinstance(getattr(defaults, key), tuple) else value
                                    for key, value in case["config"].items()})
                    world = World(cfg)
                    cache[seed] = cfg, world, Chain(world), DataStream(cfg, len(world.nodes))
                cfg, world, chain, data = cache[seed]
                assert {key: table_hash(getattr(world, key)) for key in case["world_hashes"]} == case["world_hashes"]
                assert json.dumps(public_wave_targets(world.nodes), sort_keys=True) == json.dumps(case["public_targets"], sort_keys=True)
                inputs = {"observed": table_hash(chain.observed), "audits": table_hash(chain.audits),
                    "true_down": table_hash(chain.true_down), "objects": table_hash(data.objects),
                    "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()}
                assert inputs == case["input_hashes"] and case["world_unchanged"] and case["common_random_numbers"]
                groups = case["public_targets"]["groups"]
                expected = [{"time": 15., "nodes": sorted(groups[0]+groups[1])},
                            {"time": 15.5 if stress == "close" else 22., "nodes": groups[2]}]
                assert case["stress_schedule"] == expected
                for profile, variants in case["results"].items():
                    for variant, record in variants.items():
                        assert record["control_settings"] == controls
                        assert record["drain_irrelevant_proof"] == (mode == "pressure")
                        assert record["stress_schedule"] == expected
                        audit_cancellations(record)
                        physical = audit_stress_record(record, cfg, world, chain, data)
                        assert physical == record["physical_transport_audit"]
                        if (seed, stress, mode) == (201, "close", "legacy"):
                            assert record["baseline_identical"]
                        rows.append({"seed": seed, "stress_mode": stress, "mode": mode,
                            "profile": profile, "variant": variant, **physical,
                            "labels": audit_labels(record, cfg.horizon_days)})
                cases[seed, stress, mode] = case
    assert len(rows) == 108 and len(cases) == 18
    pairs = 0
    for seed in (201, 202, 203):
        baseline = cases[seed, "close", "legacy"]
        for stress in ("close", "far"):
            for mode in ("legacy", "combined", "pressure"):
                for key in ("config", "settings", "world_hashes", "input_hashes", "public_targets"):
                    assert cases[seed, stress, mode][key] == baseline[key]
        for mode in ("legacy", "combined", "pressure"):
            a, b = cases[seed, "close", mode], cases[seed, "far", mode]
            for profile, variants in a["results"].items():
                for variant, record in variants.items():
                    other = b["results"][profile][variant]
                    prefix = lambda r: [s for s in r["observations"] if s["time"] <= 15.5]
                    assert prefix(record) == prefix(other)
                    pairs += 1
    assert pairs == 54 and before == provenance()
    assert hashes == {name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in hashes}
    result = {"schema": "taris-queue-validation-audit/1", "worlds": 3, "executions": 108,
        "input_hashes": hashes, "frozen_source_subset_verified": True, "common_past_pairs": pairs,
        "frozen_reference_hash": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows}
    (directory/"audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("I6.2: 108исполнений, замороженные правила, физика, транспорт и54пары прошлого проверены")


if __name__ == "__main__":
    main()
