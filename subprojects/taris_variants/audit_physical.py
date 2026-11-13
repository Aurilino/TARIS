"""Независимый пересчёт физических исходов по журналам копий и миру I4.1."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections import defaultdict
import hashlib
import json
import math
import numpy as np
from taris_sim.config import Config
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from subprojects.taris_variants.run_compare import table_hash
from subprojects.taris_variants.run_physical import provenance


def union(intervals):
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1] = merged[-1][0], max(b, merged[-1][1])
        else:
            merged.append((a, b))
    return merged


def rank_timeline(intervals, created, end):
    changes = defaultdict(int)
    changes[created] = 0
    changes[end] = 0
    for values in intervals.values():
        for a, b in union(values):
            changes[a] += 1
            changes[b] -= 1
    points, rank, result = sorted(changes), 0, []
    for pos, t in enumerate(points[:-1]):
        rank += changes[t]
        result.append((t, points[pos+1], rank))
    return result


def audit_result(record, cfg, chain, data):
    end, k = float(cfg.horizon_days), cfg.k
    fragments = record["physical_fragments"]
    down = {int(i): list(zip(g.start, g.end)) for i, g in chain.true_down.groupby("node")}
    by_object = defaultdict(list)
    storage = 0.
    for f in fragments:
        by_object[f["obj"]].append(f)
        stop = min(end, f["physical_end"] if f["physical_end"] is not None else end)
        storage += max(0., stop - f["start"])
    unavailable, losses = 0., {}
    for obj, row in data.objects.iterrows():
        created = float(row.created_epoch)
        physical, readable = defaultdict(list), defaultdict(list)
        for f in by_object[obj]:
            a, b = max(created, f["start"]), min(end, f["physical_end"] if f["physical_end"] is not None else end)
            if a >= b:
                continue
            physical[f["idx"]].append((a, b))
            cuts = sorted({a, b} | {float(t) for s, e in down.get(f["node"], []) for t in (s, e) if a < t < b})
            for left, right in zip(cuts, cuts[1:]):
                mid = (left + right)/2
                if not any(s <= mid < e for s, e in down.get(f["node"], [])):
                    readable[f["idx"]].append((left, right))
        bad = [a for a, b, rank in rank_timeline(physical, created, end) if b > a and rank < k]
        if bad:
            losses[int(obj)] = min(bad)
        unavailable += sum(b-a for a, b, rank in rank_timeline(readable, created, end) if rank < k)
    metrics = record["physical_metrics"]
    reported = {int(r["obj"]): float(r["t"]) for r in record["losses"]}
    assert losses.keys() == reported.keys(), (losses, reported)
    assert all(math.isclose(t, reported[obj], abs_tol=1e-10) for obj, t in losses.items())
    assert math.isclose(storage, metrics["physical_fragment_days"], abs_tol=1e-8)
    assert math.isclose(unavailable, metrics["unavailable_object_days"], abs_tol=1e-8)
    assert metrics["physical_copies"] == sum(f["physical_end"] is None for f in fragments)
    assert metrics["allocated_slots"] == sum(f["allocated"] for f in fragments) + metrics["pending_writes"]
    incomplete = sum(len(r[3]) < k for r in record["initial_writes"])
    assert incomplete == metrics["initial_incomplete_objects"] == record["initial_incomplete"]
    assert len(losses) - incomplete == metrics["lost_admitted_objects"]
    return {"objects_checked": len(data.objects), "copies_checked": len(fragments),
            "losses_recomputed": len(losses), "unavailable_object_days": unavailable,
            "physical_fragment_days": storage}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).with_name("series_i41_final"))
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
        fingerprint = {"observed": table_hash(chain.observed), "audits": table_hash(chain.audits),
            "objects": table_hash(data.objects), "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()}
        assert fingerprint == case["input_hashes"]
        for variant, record in case["results"]["physical"].items():
            rows.append({"seed": case["seed"], "variant": variant, **audit_result(record, cfg, chain, data)})
    assert rows, "Нет случаев для аудита"
    assert provenance() == before
    report = {"schema": "taris-physical-audit/1", "source_hashes_verified": True,
        "worlds_and_inputs_verified": True, "audit_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows, "objects_checked": sum(r["objects_checked"] for r in rows),
        "copies_checked": sum(r["copies_checked"] for r in rows)}
    (args.directory / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
