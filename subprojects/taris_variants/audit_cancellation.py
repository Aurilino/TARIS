"""I4.4a: прежний независимый аудит транспорта плюс наблюдаемые основания отмены."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections import Counter
import hashlib
import json
from taris_sim.config import Config
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from subprojects.taris_variants.audit_resumable import audit_record
from subprojects.taris_variants.run_compare import table_hash
from subprojects.taris_variants.run_cancellation import provenance


def audit_cancellations(record):
    rows, metrics = record["obsolete_cancellations"], record["metrics"]
    assert len(rows) == metrics["obsolete_targets_cancelled"]
    assert sum(j["status"] == "obsolete_before_read" for j in record["jobs"]) == metrics["obsolete_jobs_cancelled"]
    expected = Counter((r["time"], r["obj"], r["idx"], r["node"]) for r in rows)
    actual = Counter((t, obj, idx, node) for t, obj, idx, node, status in record["repair_outcomes"] if status == "cancelled_obsolete")
    assert expected == actual
    for row in rows:
        t, obj, idx = row["time"], row["obj"], row["idx"]
        assert row["reason"] == "not_in_observed_missing" and idx not in row["observed_missing"]
        assert not any(b["obj"] == obj and b["start"] <= t and (b["end"] is None or b["end"] > t) for b in record["buffers"])
        confirmed_return = any(rt <= t and ro == obj and ri == idx for rt, ro, ri, node in record["returns"])
        confirmed_write = any(rt <= t and ro == obj and ri == idx and status == "written"
                              for rt, ro, ri, node, status in record["repair_outcomes"])
        assert confirmed_return or confirmed_write, "Отмена без наблюдаемого получения номера"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).with_name("series_i44a"))
    args = parser.parse_args()
    before, defaults, rows = provenance(), Config(), []
    for path in sorted(args.directory.glob("s[0-9]*.json")):
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["provenance"] == before
        case, config = saved["case"], saved["case"]["config"]
        cfg = Config(**{key: tuple(value) if isinstance(getattr(defaults, key), tuple) else value for key, value in config.items()})
        world = World(cfg)
        assert {key: table_hash(getattr(world, key)) for key in case["world_hashes"]} == case["world_hashes"]
        chain, data = Chain(world), DataStream(cfg, len(world.nodes))
        inputs = {"observed": table_hash(chain.observed), "audits": table_hash(chain.audits),
            "true_down": table_hash(chain.true_down), "objects": table_hash(data.objects),
            "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()}
        assert inputs == case["input_hashes"]
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                audited = audit_record(record, cfg, world, chain, data)
                audit_cancellations(record)
                rows.append({"seed": case["seed"], "profile": profile, "variant": variant, **audited,
                    "obsolete_targets_cancelled": len(record["obsolete_cancellations"])})
    assert rows and before == provenance()
    report = {"schema": "taris-cancel-audit/1", "source_hashes_verified": True, "worlds_and_inputs_verified": True,
        "objects_checked": sum(r["objects"] for r in rows), "rows": rows,
        "audit_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("audit_cancellation.py", "audit_resumable.py", "audit_physical.py")}}
    (args.directory/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Проверено исполнений: {len(rows)}, объектов: {report['objects_checked']}; отмен: "
          f"{sum(r['obsolete_targets_cancelled'] for r in rows)}")


if __name__ == "__main__":
    main()
