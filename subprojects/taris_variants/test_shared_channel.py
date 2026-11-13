import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from dataclasses import asdict
import unittest
import heapq
import numpy as np
import pandas as pd
from copy import deepcopy
from subprojects.taris_variants.test_physical import fixture
from subprojects.taris_variants.shared_channel import SharedChannelLedger
from subprojects.taris_variants.queued import QueueSettings
from subprojects.taris_variants.resumable import ResumableLedger, TransportSettings
from subprojects.taris_variants.audit_shared_channel import audit_record, audit_schedule


def prepared(kind=SharedChannelLedger, proof_fraction=.01, return_node=True):
    L = fixture(lambda w, c, d: kind(w, c, d, QueueSettings(rate_size_per_hour=1/24, proof_fraction=proof_fraction),
                                   TransportSettings(stripes=4)))
    L.cfg = replace(L.cfg, repair_hours=0.)
    L.w.cfg = L.cfg
    L._add_fragment(0, 1, 4, 0.)
    L.down[4] = L.obs_iv[4] = np.array([.5]), np.array([3.])
    L._declare(3, .5, "простой > T_rep")
    L.down[0] = L.obs_iv[0] = np.array([.5]), np.array([7.])
    L._declare(0, .5, "простой > T_rep")
    L._repair(1., [(0, [0], [3])])
    L._push(2.1, "destroy_node", 1)
    L._push(2.1, "destroy_node", 2)
    L._process(3.)
    if return_node:
        L._return_node(4, 3.)
    return L


class SharedChannelTests(unittest.TestCase):
    def test_independent_audit_of_sharing_and_parked_memory_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                L = prepared(proof_fraction=1.)
                if fail:
                    L._push(3.6, "destroy_node", 4)
                    L._push(3.7, "queue_buffer_fail", 0)
                L._process(8.)
                L.ch.true_down = pd.DataFrame([
                    {"node": node, "start": a, "end": b} for node, (starts, ends) in L.down.items()
                    for a, b in zip(starts, ends)])
                record = {"queue_settings": asdict(L.queue_settings), "transport_settings": asdict(L.transport),
                    "metrics": L.queue_report(8.), "jobs": L.jobs, "segments": L.segments,
                    "buffers": L.buffer_intervals, "block_results": L.block_results, "close_checks": L.close_checks,
                    "decoded_prefixes": L.decoded_prefixes, "receiver_prefixes": L.receiver_prefixes,
                    "queue_events": L.queue_events, "losses": [{"obj": obj, "t": t} for obj, t in L.log_loss],
                    "reserved_by_node": L.reserved.tolist(), "allocated_slots_by_node": L.used.tolist(),
                    "scheduler_events": L.scheduler_events, "parked_intervals": L.parked_intervals,
                    "physical_fragments": [{"obj": L.f_obj[f], "idx": L.f_idx[f], "node": L.f_node[f],
                        "start": L.f_from[f], "physical_end": L.f_dead[f] if np.isfinite(L.f_dead[f]) else None,
                        "allocated": L.allocated[f]} for f in range(len(L.f_obj))]}
                audited = audit_record(record, L.cfg, L.w, L.ch, L.data)
                self.assertEqual(audited["irrecoverable_objects"], int(fail))
                bad = deepcopy(record)
                event = next(e for e in bad["scheduler_events"] if e["kind"] == "proof_bypass")
                bad["scheduler_events"].append(deepcopy(event))
                with self.assertRaises(AssertionError):
                    audit_schedule(bad, 8.)

    def test_idle_pause_without_proof_does_not_discard_buffer(self):
        L = prepared(return_node=False)
        L._process(5.)
        self.assertIn(0, L.buffers)
        self.assertFalse(any(r["kind"] == "proof_bypass" for r in L.scheduler_events))
        self.assertTrue(all(p["end"] >= p["wake"] for p in L.parked_intervals if p["end"] is not None))

    def test_world_event_at_deadline_resumes_owner_before_new_proof(self):
        L = prepared(proof_fraction=1., return_node=False)
        L._process(3.51)
        pause = L.parked_intervals[-1]
        wake = pause["wake"]
        heapq.heappush(L.heap, (wake, -1, "observed_return", 4))
        L._process(wake+.02)
        self.assertEqual(pause["end"], wake)
        first = next(r for r in L.scheduler_events if r["kind"] == "proof_bypass")
        self.assertGreater(first["time"], wake)

    def test_receiver_destruction_clears_parked_prefix_immediately(self):
        L = fixture(lambda w, c, d: SharedChannelLedger(w, c, d,
            QueueSettings(rate_size_per_hour=1/24, proof_fraction=1.), TransportSettings(stripes=4)))
        L.cfg = replace(L.cfg, repair_hours=0.)
        L.w.cfg = L.cfg
        L._add_fragment(0, 3, 4, 0.)
        L.down[4] = L.obs_iv[4] = np.array([.5]), np.array([2.26])
        L._declare(3, .5, "простой > T_rep")
        L.down[0] = L.obs_iv[0] = np.array([.5]), np.array([7.])
        L.down[3] = np.array([2.15]), np.array([4.])
        L._declare(0, .5, "простой > T_rep")
        L._repair(1., [(0, [0], [3])])
        L._process(2.26)
        L._return_node(4, 2.26)
        L._push(2.3, "destroy_node", 3)
        L._process(2.31)
        self.assertIsNotNone(L.parked_job)
        self.assertEqual(L.jobs[L.active_job]["kind"], "proof")
        self.assertEqual(L.jobs[L.parked_job]["write_progress"], 0)
        self.assertTrue(L.receiver_prefixes)
        self.assertEqual(L.receiver_prefixes[0]["end"], 2.3)
        self.assertEqual(L.disk_generation[3], 1)

    def test_proof_unlocks_buffer_without_discarding_information(self):
        old, new = prepared(ResumableLedger), prepared()
        old._process(5.)
        new._process(5.)
        self.assertIn(0, old.buffers)
        self.assertFalse(new.buffers)
        self.assertFalse(new.log_loss)
        self.assertTrue(any(r["kind"] == "proof_bypass" for r in new.scheduler_events))
        self.assertEqual(new.buffer_intervals[0]["reason"], "confirmed_disk_code")

    def test_wakeup_waits_for_active_proof(self):
        L = prepared(proof_fraction=1.)
        L._process(6.)
        self.assertFalse(L.buffers)
        self.assertGreater(L.queue_report(6.)["resume_delay_days"], 0.)
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(L.segments, L.segments[1:])))

    def test_parked_buffer_failure_is_applied_while_proof_is_active(self):
        L = prepared(proof_fraction=1.)
        L._process(3.55)
        self.assertIsNotNone(L.parked_job)
        self.assertEqual(L.jobs[L.active_job]["kind"], "proof")
        L._push(3.6, "destroy_node", 4)
        L._push(3.7, "queue_buffer_fail", 0)
        L._process(6.)
        self.assertEqual(L.log_loss, [(0, 3.7)])
        self.assertFalse(L.buffers)
        self.assertEqual(L.jobs[0]["read_progress"], 0)

    def test_new_targets_go_to_parked_owner(self):
        L = prepared(proof_fraction=1.)
        L._process(3.6)
        self.assertIsNotNone(L.parked_job)
        self.assertEqual(L.jobs[L.active_job]["kind"], "proof")
        owner = L.parked_job
        L._repair(3.6, [(0, [1, 2], [4, 5])])
        self.assertEqual(sum(j["kind"] == "repair" for j in L.jobs), 1)
        self.assertEqual(L.jobs[owner]["indices"], [0, 1, 2])
        self.assertEqual(L._outstanding(), 2)
        self.assertEqual(L.reserved[5], 1)
        self.assertEqual(L.jobs[L.active_job]["kind"], "proof")

    def test_at_most_one_proof_per_pause(self):
        L = prepared()
        L._process(5.)
        pauses = []
        count = 0
        for event in L.scheduler_events:
            if event["kind"] == "park":
                count = 0
            elif event["kind"] == "proof_bypass":
                count += 1
                self.assertLessEqual(count, 1)
            elif event["kind"] == "resume":
                pauses.append(count)
        self.assertIn(1, pauses)


if __name__ == "__main__":
    unittest.main()
