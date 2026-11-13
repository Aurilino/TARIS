"""Р4: восстановление, повторное появление нод и общий канал."""
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path

from recovery import ObjectSpec, RecoveryEngine, RecoveryPolicy, StorageNode
from run_examples import encode


def base_engine():
    engine = RecoveryEngine([StorageNode(name, name) for name in "ABCDE"], [ObjectSpec("O", 2, 3)])
    for index, node in enumerate("ABC"):
        engine.add_chunk("O", index, node)
    return engine


def shared_engine(site_rate):
    nodes = [StorageNode("A", "source_A"), StorageNode("B", "source_B"),
             StorageNode("C", "destination", capacity=1), StorageNode("D", "destination", capacity=1)]
    engine = RecoveryEngine(nodes, [ObjectSpec("O1", 2, 3), ObjectSpec("O2", 2, 3)],
                            site_bandwidth={"destination": site_rate})
    for obj in ("O1", "O2"):
        engine.add_chunk(obj, 0, "A")
        engine.add_chunk(obj, 1, "B")
    return engine


def summarize(engine):
    service_down = {obj: Fraction() for obj in engine.objects}
    storage_down = dict(service_down)
    for left, right in zip(engine.history, engine.history[1:]):
        duration = right["time"] - left["time"]
        for obj, state in left["objects"].items():
            if state["readable_unique"] < engine.objects[obj].k:
                storage_down[obj] += duration
                if not state["buffered"]:
                    service_down[obj] += duration
    return {"end": engine.time, "traffic_chunk_units": engine.traffic,
            "lost": engine.lost, "final_physical": sorted(engine.physical),
            "final_catalog": sorted(engine.catalog), "buffered": sorted(engine.buffers),
            "storage_unavailability_hours": storage_down, "service_unavailability_hours": service_down,
            "maximum_allocated_and_reserved": dict(engine.maximum_reserved_used),
            "jobs": [asdict(job) for job in engine.jobs], "log": engine.log, "history": engine.history}


def run_cases():
    cases = {}
    engine = base_engine().run(6, events=[(0, {"kind": "down", "node": "C"}),
                                         (4, {"kind": "up", "node": "C"})], policy=RecoveryPolicy())
    cases["temporary_down_and_return"] = summarize(engine)
    engine = base_engine().run(6, events=[(0, {"kind": "silent_loss", "node": "C", "obj": "O", "index": 2}),
                                         (1, {"kind": "audit", "node": "C"})], policy=RecoveryPolicy())
    cases["silent_loss_until_audit"] = summarize(engine)
    cases["shared_bandwidth_1"] = summarize(shared_engine(1).run(6, policy=RecoveryPolicy()))
    cases["shared_bandwidth_2"] = summarize(shared_engine(2).run(6, policy=RecoveryPolicy()))
    events = [(0, {"kind": "down", "node": "C"}),
              ("0.5", {"kind": "silent_loss", "node": "A", "obj": "O", "index": 0}),
              (4, {"kind": "up", "node": "C"})]
    cases["failed_source_and_retry"] = summarize(base_engine().run(8, events=events, policy=RecoveryPolicy()))
    events = [(0, {"kind": "down", "node": "C"}),
              ("1.5", {"kind": "destroy", "node": "A"}),
              ("1.5", {"kind": "destroy", "node": "B"})]
    cases["buffer_preserves_reconstructed_object"] = summarize(base_engine().run(8, events=events, policy=RecoveryPolicy()))
    cases["buffer_failure_after_source_loss"] = summarize(base_engine().run(8,
        events=events + [("1.5", {"kind": "buffer_loss"})], policy=RecoveryPolicy()))
    return cases


def main():
    cases = run_cases()
    output = Path(__file__).with_name("recovery_results.json")
    output.write_text(json.dumps({"schema": "taris-recovery-results/1", "time_unit": "hour",
                                  "chunk_size": 1, "cases": cases}, ensure_ascii=False, indent=2, default=encode) + "\n",
                       encoding="utf-8")
    for name, case in cases.items():
        completed = [job["finished"] for job in case["jobs"] if job["status"] == "done"]
        print(name, "завершения:", completed, "потери:", case["lost"], "трафик:", case["traffic_chunk_units"])


if __name__ == "__main__":
    main()
