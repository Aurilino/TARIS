"""I6: прежний физический аудит плюс явно проверяемые read-wait паузы и резерв proof."""
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
from subprojects.taris_variants.audit_cancellation import audit_cancellations


def audit_schedule(record, end, world, chain):
    settings = record["control_settings"]
    witnesses = {(r["time"], r["job"]): r for r in record["read_wait_pauses"]}
    assert len(witnesses) == len(record["read_wait_pauses"])
    read_pause_keys = {(p["start"], p["job"]) for p in record["parked_intervals"] if p.get("pause_type") == "read_wait"}
    assert read_pause_keys == set(witnesses)
    down = {int(node): list(zip(g.start, g.end)) for node, g in chain.true_down.groupby("node")}
    repair_open = set()
    for t, kind, key, depth in record["queue_events"]:
        if record["jobs"][key]["kind"] != "repair":
            continue
        if kind == "enqueue":
            if settings["reserve_proof_slot"]:
                assert len(repair_open) < record["queue_settings"]["max_jobs"]-1
            repair_open.add(key)
        elif kind == "finish":
            repair_open.remove(key)
    jobs, pauses, events = record["jobs"], record["parked_intervals"], record["scheduler_events"]
    waiting, bypassed = [], set()
    bypass = {(r["job"], r["time"]): r for r in events if r["kind"] == "proof_bypass"}
    selections = {(r["selected"], r["time"]): r for r in record["proof_selection_log"] if r["selected"] is not None}
    assert len(selections) == sum(r["selected"] is not None for r in record["proof_selection_log"])
    declarations = {(t, obj, idx, node) for t, obj, idx, node, live in record["declarations"]}
    for row in record["proof_selection_log"]:
        assert row["obj"] == jobs[row["owner"]]["obj"]
        eligible = []
        for choice in row["candidates"]:
            assert jobs[choice["job"]]["kind"] == "proof" and jobs[choice["job"]]["node"] == choice["node"]
            for holder in choice["known_holders"]:
                assert holder["declared_at"] <= row["time"]
                assert (holder["declared_at"], row["obj"], holder["idx"], choice["node"]) in declarations
            expected = sorted(set(row["observed_missing"]) & {h["idx"] for h in choice["known_holders"]})
            assert expected == choice["relevant_indices"]
            if expected:
                eligible.append(choice["job"])
        assert row["selected"] == (eligible[0] if eligible else None)
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
                selection = selections[(key, t)]
                assert selection["owner"] == pause["job"]
                assert [r["job"] for r in selection["candidates"]] == [j for j in waiting if jobs[j]["kind"] == "proof"]
                bypassed.add((key, t))
            else:
                assert key == waiting[0], "Неразрешённый обход очереди"
            waiting.remove(key)
    assert bypassed == set(bypass)
    assert set(selections) == set(bypass)
    for p in pauses:
        obj = jobs[p["job"]]["obj"]
        stop = p["end"] if p["end"] is not None else end
        full = any(b["obj"] == obj and b["start"] <= p["start"] and (b["end"] is None or b["end"] > p["start"]) for b in record["buffers"])
        if p.get("pause_type") == "read_wait":
            assert settings["bypass_read_wait"] and not full
            witness = witnesses[p["start"], p["job"]]
            assert witness["obj"] == obj and 0 <= witness["read_progress"] < record["transport_settings"]["stripes"]
            holders = witness["known_holders"]
            assert len({h["copy"] for h in holders}) == len(holders)
            assert sorted({h["idx"] for h in holders}) == witness["known_indices"]
            readable = set()
            for holder in holders:
                f = record["physical_fragments"][holder["copy"]]
                assert f["obj"] == obj and f["idx"] == holder["idx"] and f["node"] == holder["node"]
                assert f["start"] <= p["start"]
                declarations = [t for t, o, i, n, live in record["declarations"]
                                if (o, i, n) == (obj, f["idx"], f["node"]) and f["start"] < t <= p["start"]]
                if declarations:
                    assert any((o, i, n) == (obj, f["idx"], f["node"]) and max(declarations) <= t <= p["start"]
                               for t, o, i, n in record["returns"])
                node, now = f["node"], p["start"]
                if ((f["physical_end"] is None or f["physical_end"] > now)
                        and world.nodes.alive_to.iloc[node] > now
                        and not any(a <= now < b for a, b in down.get(node, []))):
                    readable.add(f["idx"])
            assert len(readable) < world.cfg.k, "Пауза чтения без фактического препятствия"
        else:
            assert full
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
    assert m["targeted_proof_starts"] == len(selections)
    assert m["proof_opportunities_without_relevant"] == sum(r["selected"] is None for r in record["proof_selection_log"])
    assert m["parked_jobs"] == sum(p["end"] is None for p in pauses) <= 1
    close(sum(max(0., min(end, p["end"] if p["end"] is not None else end)-p["start"]) for p in pauses), m["parked_owner_days"])
    close(sum(max(0., p["end"]-p["wake"]) for p in pauses if p["end"] is not None), m["resume_delay_days"])
    read_days = sum(max(0., min(end, p["end"] if p["end"] is not None else end)-p["start"])
                    for p in pauses if p.get("pause_type") == "read_wait")
    close(read_days, m["parked_read_wait_days"])
    close(m["parked_owner_days"]-read_days, m["parked_buffer_days"])
    assert m["read_wait_pause_count"] == len(witnesses)


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
    audit_schedule(record, end, world, chain)
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

