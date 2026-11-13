"""I6: проекция точечных разрушений и отдельный аудит очереди."""
from copy import copy
from math import isfinite
import pandas as pd
from .audit_queue_control import audit_record


def audit_stress_record(record, cfg, world, chain, data):
    events = [(float(wave["time"]), int(node)) for wave in record["stress_schedule"] for node in wave["nodes"]]
    assert len(events) == len(set(events))
    assert all(0 <= t < cfg.horizon_days and 0 <= node < len(world.nodes) for t, node in events)
    notices = record["disk_notifications"]
    assert [(r["time"], r["node"]) for r in notices] == sorted(events)
    copies = record["physical_fragments"]
    for notice in notices:
        assert notice["kind"] == "confirmed_disk_failure"
        t, node = notice["time"], notice["node"]
        assert len(notice["declared_copies"]) == len(set(notice["declared_copies"]))
        for f in notice["declared_copies"]:
            row = copies[f]
            assert row["node"] == node and row["start"] <= t
            assert row["physical_end"] is not None and row["physical_end"] <= t
            assert not row["allocated"]
        for f, row in enumerate(copies):
            if row["node"] != node or row["start"] > t:
                continue
            stop = row["physical_end"]
            assert stop is not None and stop <= t, "Копия пережила разрушение диска"
            if stop == t:
                # Ранее освобождённая копия может иметь тот же момент смерти;
                # её уведомление не обязано повторяться.
                assert isfinite(stop)
    # Только аудиторская проекция: нулевая длительность, физическая смерть в t.
    # Мир исполнения и chain.true_down остаются прежними.
    projected = copy(world)
    extra = pd.DataFrame([{"node": node, "start": t, "end": t, "cls": "total", "destroyed": True}
                          for t, node in events])
    projected.outages = pd.concat([world.outages, extra], ignore_index=True) if events else world.outages
    result = audit_record(record, cfg, projected, chain, data)
    return {**result, "disk_error_notices": len(notices), "point_destructions_verified": len(events)}
