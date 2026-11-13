"""I4.4b: изолированный аудит транспорта и явно разрешённых обходов FIFO."""
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
from subprojects.taris_variants.run_shared_channel import provenance
from subprojects.taris_variants.audit_cancellation import audit_cancellations


def audit_schedule(record, end):
    jobs, pauses, events = record["jobs"], record["parked_intervals"], record["scheduler_events"]
    waiting, bypassed = [], set()
    bypass = {(r["job"], r["time"]): r for r in events if r["kind"] == "proof_bypass"}
    assert len(bypass) == sum(r["kind"] == "proof_bypass" for r in events)
    for previous, following in zip(pauses, pauses[1:]):
        assert previous["end"] is not None and previous["end"] <= following["start"]
    for t, kind, key, depth in record["queue_events"]:
        if kind == "enqueue":
            waiting.append(key)
        elif kind == "start":
            assert key in waiting
            pause = next((p for p in pauses if p["start"] <= t < (p["end"] if p["end"] is not None else end)), None)
            if pause is not None:
                event = bypass[(key, t)]
                assert event["owner"] == pause["job"] and jobs[key]["kind"] == "proof"
                assert t < pause["wake"]
                assert key == next(j for j in waiting if jobs[j]["kind"] == "proof")
                bypassed.add((key, t))
            else:
                assert key == waiting[0], "Неразрешённый обход очереди"
            waiting.remove(key)
    assert bypassed == set(bypass)
    for p in pauses:
        obj = jobs[p["job"]]["obj"]
        stop = p["end"] if p["end"] is not None else end
        assert any(b["obj"] == obj and b["start"] <= p["start"] and (b["end"] is None or b["end"] > p["start"]) for b in record["buffers"])
        selected = [e for e in bypass.values() if e["owner"] == p["job"] and p["start"] <= e["time"] < stop]
        assert len(selected) <= 1
        if p["end"] is not None:
            assert p["end"] >= p["wake"]
        for s in record["segments"]:
            if s["start"] < stop and s["end"] > p["start"]:
                assert jobs[s["job"]]["kind"] == "proof", "Ремонт занял канал при приостановленном владельце"
        if selected:
            proof = jobs[selected[0]["job"]]
            if proof["finished"] is not None and p["end"] is not None:
                close(p["end"], max(p["wake"], proof["finished"]))
        elif p["end"] is not None:
            close(p["end"], p["wake"])
    m = record["metrics"]
    assert m["proof_bypass_starts"] == len(bypass)
    assert m["parked_jobs"] == sum(p["end"] is None for p in pauses) <= 1
    close(sum(max(0., min(end, p["end"] if p["end"] is not None else end)-p["start"]) for p in pauses), m["parked_buffer_days"])
    close(sum(max(0., p["end"]-p["wake"]) for p in pauses if p["end"] is not None), m["resume_delay_days"])


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def audit_record(record, cfg, world, chain, data):
    end, k = float(cfg.horizon_days), cfg.k
    copies, jobs, segments, buffers = (record[key] for key in ("physical_fragments", "jobs", "segments", "buffers"))
    metrics, settings = record["metrics"], record["queue_settings"]
    stripes = record["transport_settings"]["stripes"]
    down = {int(node): list(zip(g.start, g.end)) for node, g in chain.true_down.groupby("node")}
    destroy = defaultdict(list)
    for row in world.outages[(world.outages.cls == "total") & world.outages.destroyed.astype(bool)].itertuples():
        destroy[int(row.node)].append(float(row.start))
    for node, t in enumerate(world.nodes.leave_t):
        if math.isfinite(t):
            destroy[node].append(float(t))

    def continuous(node, a, b):
        return (world.nodes.alive_to.iloc[node] > b and not any(
            (s < b and e > a) or s <= b < e for s, e in down.get(node, [])))

    def valid_sources(ids, a, b):
        return (len(ids) == k and len({copies[f]["idx"] for f in ids}) == k and all(
            copies[f]["start"] <= a and (copies[f]["physical_end"] is None or copies[f]["physical_end"] >= b-1e-10)
            and continuous(copies[f]["node"], a, b) for f in ids))

    results = {r["segment"]: r["ok"] for r in record["block_results"]}
    assert len(results) == len(record["block_results"])
    checks = {(r["job"], r["time"]): r for r in record["close_checks"]}
    read_progress, write_progress = Counter(), Counter()
    proven_buffers, proven_copies = set(), set()
    proven_prefixes = {"read_block": set(), "write_block": set()}
    volumes, previous = Counter(), 0.
    for number, s in enumerate(segments):
        a, b, phase = s["start"], s["end"], s["phase"]
        assert b > a and a >= previous-1e-10
        previous = b
        close(s["volume"], (b-a)*settings["rate_size_per_hour"]*24.)
        volumes[phase] += max(0., min(end, b)-a)*settings["rate_size_per_hour"]*24.
        job = jobs[s["job"]]
        assert job["started"] <= a
        if job["finished"] is not None:
            assert b <= job["finished"]+1e-10
        if phase == "proof":
            close(s["volume"], settings["proof_fraction"]*sum(data.objects.loc[copies[f]["obj"], "size"]/k for f in job["copies"]))
            continue
        obj, size = job["obj"], float(data.objects.loc[job["obj"], "size"])
        if phase == "close_proof":
            close(s["volume"], settings["proof_fraction"]*size)
            if b < end:
                check = checks[(job["id"], b)]
                if check["ok"]:
                    assert valid_sources(s["sources"], a, b)
                    assert any(r["job"] == job["id"] and r["end"] == b and r["reason"] == "confirmed_disk_code" for r in buffers)
            continue
        close(s["volume"], size/stripes if phase == "read_block" else size/k/stripes)
        assert 0 <= s["stripe"] < stripes
        if phase == "read_block":
            progress_key = job["id"], s["memory_generation"]
            assert s["stripe"] == read_progress[progress_key]
        elif phase == "write_block":
            progress_key = job["id"], s["idx"], s["node"], s["disk_generation"]
            if write_progress[progress_key] == stripes:
                write_progress[progress_key] = 0  # новая замена того же номера после признанной потери
            assert s["stripe"] == write_progress[progress_key]
        else:
            raise AssertionError(phase)
        if b >= end:
            assert number not in results
            continue
        assert number in results
        if not results[number]:
            continue
        if phase == "read_block":
            assert valid_sources(s["sources"], a, b)
            read_progress[progress_key] += 1
            matches = [r for r in record["decoded_prefixes"] if r["job"] == job["id"] and r["start"] == b and r["stripe"] == s["stripe"]]
            assert len(matches) == 1
            proven_prefixes[phase].add((job["id"], b, s["stripe"]))
            if read_progress[progress_key] == stripes:
                assert any(r["job"] == job["id"] and r["start"] == b for r in buffers)
                proven_buffers.add((job["id"], b))
        else:
            assert continuous(s["node"], a, b)
            assert not any(a < t <= b for t in destroy[s["node"]])
            assert any(r["job"] == job["id"] and r["start"] <= a and (r["end"] is None or r["end"] >= b-1e-10) for r in buffers)
            write_progress[progress_key] += 1
            matches = [r for r in record["receiver_prefixes"] if r["job"] == job["id"] and r["start"] == b
                       and r["stripe"] == s["stripe"] and r["node"] == s["node"] and r["idx"] == s["idx"]]
            assert len(matches) == 1
            proven_prefixes[phase].add((job["id"], b, s["stripe"], s["idx"], s["node"]))
            if write_progress[progress_key] == stripes:
                assert any(f["obj"] == obj and f["idx"] == s["idx"] and f["node"] == s["node"] and f["start"] == b for f in copies)
                proven_copies.add((obj, s["idx"], s["node"], b))
    for phase in set(volumes) | set(metrics["serviced_volume"]):
        close(volumes[phase], metrics["serviced_volume"].get(phase, 0.))
    assert sum(volumes.values()) <= end*24.*settings["rate_size_per_hour"]+1e-8
    for buffer in buffers:
        assert (buffer["job"], buffer["start"]) in proven_buffers
        assert buffer["size"] <= settings["buffer_capacity_size"]
        if buffer["end"] is not None and buffer["reason"] != "buffer_failure":
            assert buffer["reason"] == "confirmed_disk_code"
            assert checks[(buffer["job"], buffer["end"])]["ok"]
    for r in record["decoded_prefixes"]:
        assert (r["job"], r["start"], r["stripe"]) in proven_prefixes["read_block"]
        close(r["size"], float(data.objects.loc[r["obj"], "size"])/stripes)
    for r in record["receiver_prefixes"]:
        assert (r["job"], r["start"], r["stripe"], r["idx"], r["node"]) in proven_prefixes["write_block"]
        close(r["size"], float(data.objects.loc[r["obj"], "size"])/k/stripes)
    if "initial_writes" in record:
        for obj, t, requested, written in record["initial_writes"]:
            seen = set()
            for idx, node in enumerate(requested):
                if node in written and node not in seen:
                    proven_copies.add((obj, idx, node, t))
                    seen.add(node)
        assert all((f["obj"], f["idx"], f["node"], f["start"]) in proven_copies for f in copies)
    assert len(results) == metrics["block_attempts"]
    assert sum(not ok for ok in results.values()) == metrics["failed_blocks"]

    memory_changes = defaultdict(float)
    for rows, key in ((record["decoded_prefixes"], "decoded_prefix_size_days"),
                      (record["receiver_prefixes"], "receiver_prefix_size_days")):
        cost = 0.
        for r in rows:
            stop = min(end, r["end"] if r["end"] is not None else end)
            assert r["start"] <= stop
            cost += r["size"]*(stop-r["start"])
        close(cost, metrics[key])
    for r in record["decoded_prefixes"] + buffers:
        stop = min(end, r["end"] if r["end"] is not None else end)
        if r["start"] < stop:
            memory_changes[r["start"]] += r["size"]
            memory_changes[stop] -= r["size"]
    used = peak = 0.
    for t in sorted(memory_changes):
        used += memory_changes[t]
        assert -1e-8 <= used <= settings["buffer_capacity_size"]+1e-8
        peak = max(peak, used)
    close(peak, metrics["peak_central_size"])
    outstanding, starts = set(), []
    for t, kind, key, depth in record["queue_events"]:
        if kind == "enqueue":
            assert key not in outstanding
            outstanding.add(key)
        elif kind == "finish":
            outstanding.remove(key)
        elif kind == "start":
            starts.append(key)
        else:
            raise AssertionError(kind)
        assert len(outstanding) == depth <= settings["max_jobs"]
    audit_schedule(record, end)
    assert len(outstanding) == metrics["unfinished_jobs"]
    close(sum(j["started"]-j["submitted"] for j in jobs if j["started"] is not None), metrics["started_wait_days_sum"])
    close(sum(end-j["submitted"] for j in jobs if j["started"] is None), metrics["pending_wait_days_sum"])
    reserved = np.zeros(len(world.nodes), int)
    for job in jobs:
        if job["kind"] == "repair" and job["finished"] is None:
            for node in job["nodes"][job.get("cursor", 0):]:
                reserved[node] += 1
    allocated = np.bincount([f["node"] for f in copies if f["allocated"]], minlength=len(world.nodes))
    assert np.array_equal(reserved, record["reserved_by_node"])
    assert np.array_equal(allocated+reserved, record["allocated_slots_by_node"])
    scale = world.nodes.cap_scale.to_numpy() if "cap_scale" in world.nodes else np.ones(len(world.nodes))
    capacity = np.where(scale == 1., data.capacity, np.maximum(1, np.round(data.capacity*scale)).astype(int))
    assert np.all(allocated+reserved <= capacity)

    by_object, by_buffer = defaultdict(list), defaultdict(list)
    storage = deficit = buffer_only = unavailable = buffer_days = 0.
    for copy in copies:
        by_object[copy["obj"]].append(copy)
        stop = min(end, copy["physical_end"] if copy["physical_end"] is not None else end)
        storage += max(0., stop-copy["start"])
    for buffer in buffers:
        stop = min(end, buffer["end"] if buffer["end"] is not None else end)
        by_buffer[buffer["obj"]].append((buffer["start"], stop))
        buffer_days += max(0., stop-buffer["start"])
    losses = {}
    for obj, row in data.objects.iterrows():
        physical, readable = defaultdict(list), defaultdict(list)
        for f in by_object[obj]:
            a, b = f["start"], min(end, f["physical_end"] if f["physical_end"] is not None else end)
            if a >= b:
                continue
            physical[f["idx"]].append((a, b))
            cuts = sorted({a, b} | {float(t) for s, e in down.get(f["node"], []) for t in (s, e) if a < t < b})
            for x, y in zip(cuts, cuts[1:]):
                if not any(s <= (x+y)/2 < e for s, e in down.get(f["node"], [])):
                    readable[f["idx"]].append((x, y))
        for a, b, rank in rank_timeline(physical, float(row.created_epoch), end):
            if rank >= k:
                continue
            deficit += b-a
            cursor = a
            for left, right in union(by_buffer[obj]):
                x, y = max(a, left), min(b, right)
                if x >= y:
                    continue
                buffer_only += y-x
                if x > cursor+1e-10:
                    losses.setdefault(int(obj), cursor)
                cursor = max(cursor, y)
            if cursor < b-1e-10:
                losses.setdefault(int(obj), cursor)
        unavailable += sum(b-a for a, b, rank in rank_timeline(readable, float(row.created_epoch), end) if rank < k)
    reported = {int(r["obj"]): r["t"] for r in record["losses"]}
    assert losses.keys() == reported.keys(), (losses, reported)
    for obj, t in losses.items():
        close(t, reported[obj])
    for value, key in ((storage, "physical_fragment_days"), (deficit, "disk_deficit_object_days"),
                       (buffer_only, "buffer_only_object_days"), (unavailable, "unavailable_object_days"),
                       (buffer_days, "buffer_object_days")):
        close(value, metrics[key])
    assert len(losses) == metrics["irrecoverable_objects"]
    return {"objects": len(data.objects), "copies": len(copies), "buffers": len(buffers), "jobs": len(jobs),
            "block_attempts": len(results), "irrecoverable_objects": len(losses), "serviced_volume": dict(volumes)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).with_name("series_i44b"))
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
                audit_cancellations(record)
                rows.append({"seed": case["seed"], "profile": profile, "variant": variant,
                             **audit_record(record, cfg, world, chain, data)})
    assert rows and before == provenance()
    report = {"schema": "taris-shared-channel-audit/1", "source_hashes_verified": True, "worlds_and_inputs_verified": True,
        "audit_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "audit_helper_hash": hashlib.sha256(Path(__file__).with_name("audit_physical.py").read_bytes()).hexdigest(),
        "cancellation_helper_hash": hashlib.sha256(Path(__file__).with_name("audit_cancellation.py").read_bytes()).hexdigest(),
        "objects_checked": sum(r["objects"] for r in rows), "rows": rows}
    (args.directory/"audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
