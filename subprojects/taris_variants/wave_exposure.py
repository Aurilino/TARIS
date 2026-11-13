"""Внешний физический аудит уязвимости непосредственно перед волной."""
from collections import defaultdict


def wave_exposure(record, nodes, time, k):
    targets = set(nodes)
    before, after = defaultdict(set), defaultdict(set)
    for fragment in record["physical_fragments"]:
        stop = fragment["physical_end"]
        if fragment["start"] <= time and (stop is None or stop >= time):
            obj = fragment["obj"]
            before[obj].add(fragment["idx"])
            if fragment["node"] not in targets:
                after[obj].add(fragment["idx"])
    full = {r["obj"] for r in record["buffers"]
            if r["start"] <= time and (r["end"] is None or r["end"] >= time)}
    already_lost = {r["obj"] for r in record["losses"] if r["t"] < time}
    rows = []
    for obj, created, requested, written in record["initial_writes"]:
        if created > time or obj in already_lost or len(set(written)) < k:
            continue
        before_rank, after_rank = len(before[obj]), len(after[obj])
        buffered = obj in full
        vulnerable = (before_rank >= k or buffered) and after_rank < k and not buffered
        rows.append({"obj": obj, "rank_before": before_rank, "rank_after_removing_targets": after_rank,
            "full_buffer_before": buffered, "immediately_vulnerable": vulnerable})
    return {"time": time, "targets": sorted(targets), "objects_at_risk": len(rows),
        "immediately_vulnerable_objects": [r["obj"] for r in rows if r["immediately_vulnerable"]], "rows": rows}
