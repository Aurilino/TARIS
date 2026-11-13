"""Независимый аудит дисков, полных буферов и общего канала I4.2."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import numpy as np
from taris_sim.config import Config
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from subprojects.taris_variants.audit_physical import union, rank_timeline
from subprojects.taris_variants.run_compare import table_hash
from subprojects.taris_variants.run_queued import provenance


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def audit_record(record, cfg, world, chain, data):
    end, k = float(cfg.horizon_days), cfg.k
    fragments, jobs, segments, buffers = (record[key] for key in ("physical_fragments", "jobs", "segments", "buffers"))
    settings, metrics = record["queue_settings"], record["metrics"]
    down = {int(node): list(zip(g.start, g.end)) for node, g in chain.true_down.groupby("node")}

    def continuous(node, a, b):
        return (world.nodes.alive_to.iloc[node] > b and not any(
            (s < b and e > a) or s <= b < e for s, e in down.get(node, [])))

    by_job, volumes = defaultdict(list), Counter()
    previous = 0.
    for segment in segments:
        a, b, volume = (segment[key] for key in ("start", "end", "volume"))
        assert b > a and a >= previous-1e-10
        previous = b
        close(volume, (b-a)*settings["rate_size_per_hour"]*24.)
        job = jobs[segment["job"]]
        assert job["started"] <= a
        if job["finished"] is not None:
            assert b <= job["finished"]+1e-10
        phase = segment["phase"]
        if phase == "proof":
            expected = settings["proof_fraction"]*sum(data.objects.loc[fragments[f]["obj"], "size"]/k for f in job["copies"])
            if job["status"] == "proof_completed":
                assert continuous(job["node"], a, b)
        else:
            size = float(data.objects.loc[job["obj"], "size"])
            expected = size if phase == "read" else len(job["indices"])*size/k
        close(volume, expected)
        volumes[phase] += max(0., min(end, b)-a)*settings["rate_size_per_hour"]*24.
        by_job[job["id"]].append(segment)
    assert sum(volumes.values()) <= end*24.*settings["rate_size_per_hour"]+1e-8
    for phase in set(volumes) | set(metrics["serviced_volume"]):
        close(volumes[phase], metrics["serviced_volume"].get(phase, 0.))

    outstanding, started = set(), []
    for t, kind, key, depth in record["queue_events"]:
        if kind == "enqueue":
            assert key not in outstanding
            outstanding.add(key)
        elif kind == "finish":
            outstanding.remove(key)
        elif kind == "start":
            assert key in outstanding
            started.append(key)
        else:
            raise AssertionError(kind)
        assert len(outstanding) == depth <= settings["max_jobs"]
    assert started == list(range(len(started))), "Нарушен FIFO"
    assert len(outstanding) == metrics["unfinished_jobs"]
    assert {j["id"] for j in jobs if j["finished"] is None} == outstanding
    close(sum(j["started"]-j["submitted"] for j in jobs if j["started"] is not None), metrics["started_wait_days_sum"])
    close(sum(end-j["submitted"] for j in jobs if j["started"] is None), metrics["pending_wait_days_sum"])

    ordered_buffers = sorted(buffers, key=lambda r: r["start"])
    for previous_buffer, following in zip(ordered_buffers, ordered_buffers[1:]):
        assert previous_buffer["end"] is not None and previous_buffer["end"] <= following["start"]+1e-10
    for buffer in buffers:
        assert buffer["size"] <= settings["buffer_capacity_size"]
        job = jobs[buffer["job"]]
        read = next(s for s in by_job[job["id"]] if s["phase"] == "read")
        write = next(s for s in by_job[job["id"]] if s["phase"] == "write")
        close(buffer["start"], read["end"])
        close(write["start"], read["end"])
        assert len({fragments[f]["idx"] for f in job["sources"]}) == k
        for f in job["sources"]:
            copy = fragments[f]
            assert copy["start"] <= read["start"]
            assert copy["physical_end"] is None or copy["physical_end"] >= read["end"]-1e-10
            assert continuous(copy["node"], read["start"], read["end"])
        if buffer["end"] is None:
            assert write["end"] >= end and job["finished"] is None
        else:
            assert buffer["start"] <= buffer["end"] <= write["end"]+1e-10
            if buffer["reason"] == "write_attempt_finished":
                close(buffer["end"], write["end"])
    for t, obj, idx, node, status in record["repair_outcomes"]:
        if status == "written":
            matches = [j for j in jobs if j["kind"] == "repair" and j["obj"] == obj
                and any(i == idx and n == node for i, n in zip(j["indices"], j["nodes"]))
                and j["finished"] is not None and math.isclose(j["finished"], t, abs_tol=1e-10)]
            assert matches
            job = matches[0]
            write = next(s for s in by_job[job["id"]] if s["phase"] == "write")
            assert continuous(node, write["start"], t)
            assert any(b["job"] == job["id"] and b["start"] <= t and b["end"] >= t-1e-10 for b in buffers)
            assert any(f["obj"] == obj and f["idx"] == idx and f["node"] == node
                       and math.isclose(f["start"], t, abs_tol=1e-10) for f in fragments)

    copies_by_object, buffers_by_object = defaultdict(list), defaultdict(list)
    storage = unavailable = disk_deficit = buffer_only = buffer_days = 0.
    for copy in fragments:
        copies_by_object[copy["obj"]].append(copy)
        stop = min(end, copy["physical_end"] if copy["physical_end"] is not None else end)
        storage += max(0., stop-copy["start"])
    for buffer in buffers:
        stop = min(end, buffer["end"] if buffer["end"] is not None else end)
        buffers_by_object[buffer["obj"]].append((buffer["start"], stop))
        buffer_days += max(0., stop-buffer["start"])
    losses = {}
    for obj, row in data.objects.iterrows():
        created = float(row.created_epoch)
        physical, readable = defaultdict(list), defaultdict(list)
        for f in copies_by_object[obj]:
            a, b = f["start"], min(end, f["physical_end"] if f["physical_end"] is not None else end)
            if a >= b:
                continue
            physical[f["idx"]].append((a, b))
            cuts = sorted({a, b} | {float(t) for s, e in down.get(f["node"], []) for t in (s, e) if a < t < b})
            for left, right in zip(cuts, cuts[1:]):
                mid = (left+right)/2
                if not any(s <= mid < e for s, e in down.get(f["node"], [])):
                    readable[f["idx"]].append((left, right))
        for a, b, rank in rank_timeline(physical, created, end):
            if rank >= k:
                continue
            disk_deficit += b-a
            cursor = a
            for left, right in union(buffers_by_object[obj]):
                x, y = max(a, left), min(b, right)
                if x >= y:
                    continue
                buffer_only += y-x
                if x > cursor+1e-10:
                    losses.setdefault(int(obj), cursor)
                cursor = max(cursor, y)
            if cursor < b-1e-10:
                losses.setdefault(int(obj), cursor)
        unavailable += sum(b-a for a, b, rank in rank_timeline(readable, created, end) if rank < k)
    reported = {int(row["obj"]): float(row["t"]) for row in record["losses"]}
    assert losses.keys() == reported.keys(), (losses, reported)
    for obj, t in losses.items():
        close(t, reported[obj])
    for value, key in ((storage, "physical_fragment_days"), (unavailable, "unavailable_object_days"),
                       (disk_deficit, "disk_deficit_object_days"), (buffer_only, "buffer_only_object_days"),
                       (buffer_days, "buffer_object_days")):
        close(value, metrics[key])
    assert len(losses) == metrics["irrecoverable_objects"]
    reserved = np.zeros(len(world.nodes), int)
    for job in jobs:
        if job["kind"] == "repair" and job["finished"] is None:
            for node in job["nodes"]:
                reserved[node] += 1
    allocated = np.bincount([f["node"] for f in fragments if f["allocated"]], minlength=len(world.nodes))
    assert np.array_equal(reserved, record["reserved_by_node"])
    assert np.array_equal(allocated+reserved, record["allocated_slots_by_node"])
    scale = world.nodes.cap_scale.to_numpy() if "cap_scale" in world.nodes else np.ones(len(world.nodes))
    capacity = np.where(scale == 1., data.capacity, np.maximum(1, np.round(data.capacity*scale)).astype(int))
    assert np.all(allocated+reserved <= capacity)
    return {"objects": len(data.objects), "copies": len(fragments), "buffers": len(buffers),
        "jobs": len(jobs), "irrecoverable_objects": len(losses), "serviced_volume": dict(volumes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).with_name("series_i42"))
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
                rows.append({"seed": case["seed"], "profile": profile, "variant": variant,
                             **audit_record(record, cfg, world, chain, data)})
    assert rows and provenance() == before
    report = {"schema": "taris-queue-audit/1", "source_hashes_verified": True,
        "worlds_and_inputs_verified": True, "audit_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audit_helper_hash": hashlib.sha256(Path(__file__).with_name("audit_physical.py").read_bytes()).hexdigest(),
        "objects_checked": sum(r["objects"] for r in rows), "rows": rows}
    (args.directory/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
