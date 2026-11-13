"""I5.1: повторный аудит мира, расписаний и общего прошлого пяти ветвей."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
from taris_sim.config import Config
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from subprojects.taris_variants.run_stress_sensitivity import provenance
from subprojects.taris_variants.run_compare import table_hash
from subprojects.taris_variants.disk_stress import public_wave_targets
from subprojects.taris_variants.audit_disk_stress import audit_stress_record
from subprojects.taris_variants.audit_cancellation import audit_cancellations
from subprojects.taris_variants.audit_observed import audit_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).with_name("series_i51"))
    args = parser.parse_args()
    before, rows, cases = provenance(), [], {}
    paths = sorted(args.directory.glob("s[0-9]*_*.json"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    defaults = Config()
    for path in paths:
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["provenance"] == before
        case = saved["case"]
        cfg = Config(**{key: tuple(value) if isinstance(getattr(defaults, key), tuple) else value
                        for key, value in case["config"].items()})
        world = World(cfg)
        assert {key: table_hash(getattr(world, key)) for key in case["world_hashes"]} == case["world_hashes"]
        assert json.dumps(public_wave_targets(world.nodes), sort_keys=True) == json.dumps(case["public_targets"], sort_keys=True)
        chain, data = Chain(world), DataStream(cfg, len(world.nodes))
        inputs = {"observed": table_hash(chain.observed), "audits": table_hash(chain.audits),
            "true_down": table_hash(chain.true_down), "objects": table_hash(data.objects),
            "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()}
        assert inputs == case["input_hashes"]
        assert case["world_unchanged"] and case["common_random_numbers"]
        groups = case["public_targets"]["groups"]
        mode = case["stress_mode"]
        expected = [] if mode == "control" else [
            {"time": 15., "nodes": sorted(groups[0]+groups[1])},
            {"time": 15.+{"close": .5, "after_decision": 1.5, "medium": 3., "far": 7.}[mode], "nodes": groups[2]}]
        assert expected == case["stress_schedule"]
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                assert record["stress_schedule"] == expected
                if mode == "control" and case["seed"] == 101:
                    assert record["baseline_identical"]
                audit_cancellations(record)
                physical = audit_stress_record(record, cfg, world, chain, data)
                assert physical == record["physical_transport_audit"]
                rows.append({"seed": case["seed"], "mode": mode, "profile": profile, "variant": variant,
                             **physical, "observations": audit_labels(record, cfg.horizon_days)})
        cases[case["seed"], mode] = case
    assert cases
    seeds = sorted({key[0] for key in cases})
    assert seeds == [101, 102, 103, 104, 105]
    modes = ("control", "close", "after_decision", "medium", "far")
    assert len(cases) == 25 and len(rows) == 150
    pairs_checked = 0
    for seed in seeds:
        baseline = cases[seed, "control"]
        for mode in modes:
            current = cases[seed, mode]
            for key in ("world_hashes", "input_hashes", "public_targets", "config", "settings"):
                assert current[key] == baseline[key]
        for left in range(len(modes)):
            for right in range(left+1, len(modes)):
                a, b = cases[seed, modes[left]], cases[seed, modes[right]]
                cutoff = 15. if "control" in (modes[left], modes[right]) else min(
                    a["stress_schedule"][1]["time"], b["stress_schedule"][1]["time"])
                for profile, variants in a["results"].items():
                    for variant, previous in variants.items():
                        current = b["results"][profile][variant]
                        prefix = lambda record: [r for r in record["observations"] if r["time"] <= cutoff]
                        assert prefix(previous) == prefix(current), "Будущая разница волн изменила прошлое"
                        writes = lambda record: [r for r in record["initial_writes"] if r[1] <= cutoff]
                        assert writes(previous) == writes(current)
                        pairs_checked += 1
    assert before == provenance()
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = {"schema": "taris-stress-sensitivity-audit/1", "worlds": len({key[0] for key in cases}),
        "executions": len(rows), "common_history_pairs": pairs_checked, "inputs": hashes, "common_past_verified": True,
        "interventions_verified": True, "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows}
    (args.directory/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f'I5.1: {len(rows)} исполнений прошли повторный физический, транспортный и парный аудит')


if __name__ == "__main__":
    main()
