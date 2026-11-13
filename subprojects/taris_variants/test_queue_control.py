import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from dataclasses import asdict
from copy import deepcopy
from subprojects.taris_variants.queue_control import QueueControlLedger, QueueControlSettings
from subprojects.taris_variants.queued import QueueSettings
from subprojects.taris_variants.resumable import TransportSettings
from subprojects.taris_variants.test_physical import fixture
from subprojects.taris_variants.disk_stress import DiskStressLedger
from subprojects.taris_variants.test_shared_channel import prepared
from subprojects.taris_variants.audit_queue_stress import audit_stress_record


def audit_fixture(ledger):
    ledger._process(8.)
    ledger.ch.true_down = pd.DataFrame([{"node": node, "start": a, "end": b}
        for node, (starts, ends) in ledger.down.items() for a, b in zip(starts, ends)], columns=["node", "start", "end"])
    record = {"metrics": ledger.queue_report(8.), "control_settings": asdict(ledger.control),
        "queue_settings": asdict(ledger.queue_settings), "transport_settings": asdict(ledger.transport),
        "jobs": ledger.jobs, "segments": ledger.segments, "buffers": ledger.buffer_intervals,
        "decoded_prefixes": ledger.decoded_prefixes, "receiver_prefixes": ledger.receiver_prefixes,
        "block_results": ledger.block_results, "close_checks": ledger.close_checks,
        "queue_events": ledger.queue_events, "parked_intervals": ledger.parked_intervals,
        "scheduler_events": ledger.scheduler_events, "proof_selection_log": ledger.proof_selection_log,
        "declarations": ledger.declarations, "returns": ledger.returns,
        "read_wait_pauses": ledger.read_wait_pauses, "repair_admissions": ledger.repair_admissions,
        "losses": [{"obj": obj, "t": t} for obj, t in ledger.log_loss],
        "reserved_by_node": ledger.reserved.tolist(), "allocated_slots_by_node": ledger.used.tolist(),
        "stress_schedule": [], "disk_notifications": [],
        "physical_fragments": [{"obj": ledger.f_obj[f], "idx": ledger.f_idx[f], "node": ledger.f_node[f],
            "start": ledger.f_from[f], "physical_end": ledger.f_dead[f] if np.isfinite(ledger.f_dead[f]) else None,
            "allocated": ledger.allocated[f]} for f in range(len(ledger.f_obj))]}
    return record


def blocked(reserve=False, bypass=False, max_jobs=2, stripes=4):
    ledger = fixture(lambda w, c, d: QueueControlLedger(w, c, d,
        QueueSettings(rate_size_per_hour=1/24, max_jobs=max_jobs), TransportSettings(stripes=stripes),
        control=QueueControlSettings(reserve, bypass)))
    ledger.cfg = replace(ledger.cfg, repair_hours=0.)
    ledger.w.cfg = ledger.cfg
    ledger.down[0] = ledger.obs_iv[0] = np.array([.5]), np.array([8.])
    ledger.down[1] = ledger.obs_iv[1] = np.array([.5]), np.array([1.])
    ledger._declare(0, .5, "простой > T_rep")
    ledger._declare(1, .5, "простой > T_rep")
    ledger._world_events()
    ledger._repair(.6, [(0, [0], [3]), (0, [1], [4])])
    return ledger


def partial_wait(fault=False, interrupted_proof=False):
    ledger = fixture(lambda w, c, d: QueueControlLedger(w, c, d,
        QueueSettings(rate_size_per_hour=1/24, proof_fraction=1.), TransportSettings(stripes=4),
        control=QueueControlSettings(True, True)))
    ledger.cfg = replace(ledger.cfg, repair_hours=0.)
    ledger.w.cfg = ledger.cfg
    ledger.down[0] = ledger.obs_iv[0] = np.array([.5]), np.array([8.])
    starts, ends = ([.9, 1.6], [1.5, 1.8]) if interrupted_proof else ([.9], [1.5])
    ledger.down[1] = ledger.obs_iv[1] = np.array(starts), np.array(ends)
    ledger._declare(0, .5, "простой > T_rep")
    ledger._world_events()
    ledger._push(.9, "declare_node", 1)
    if fault:
        ledger._push(1.7, "queue_buffer_fail", 0)
    ledger._repair(.6, [(0, [0], [3])])
    return ledger


class QueueControlTests(unittest.TestCase):
    def test_independent_audit_accepts_partial_pause_but_rejects_fabricated_buffer(self):
        ledger = partial_wait(fault=True)
        record = audit_fixture(ledger)
        result = audit_stress_record(record, ledger.cfg, ledger.w, ledger.ch, ledger.data)
        self.assertEqual(result["irrecoverable_objects"], 0)
        corrupted = deepcopy(record)
        p = next(p for p in corrupted["parked_intervals"] if p.get("pause_type") == "read_wait")
        del p["pause_type"]
        with self.assertRaises(AssertionError):
            audit_stress_record(corrupted, ledger.cfg, ledger.w, ledger.ch, ledger.data)

    def test_disabled_rules_preserve_complete_transport_history(self):
        original, control = prepared(DiskStressLedger), prepared(QueueControlLedger)
        original._process(8.)
        control._process(8.)
        for key in ("jobs", "segments", "block_results", "scheduler_events", "buffer_intervals",
                    "decoded_prefixes", "receiver_prefixes", "proof_selection_log", "log_loss"):
            self.assertEqual(getattr(original, key), getattr(control, key), key)

    def test_prefix_survives_wait_and_fault_resets_it_during_proof(self):
        for fault in (False, True):
            with self.subTest(fault=fault):
                ledger = partial_wait(fault=fault)
                ledger._process(1.55)
                self.assertIsNotNone(ledger.parked_job)
                self.assertEqual(ledger.jobs[ledger.active_job]["kind"], "proof")
                self.assertEqual(sum(r["size"] for r in ledger.decoded_prefixes if r["end"] is None), .25)
                ledger._process(1.71)
                if fault:
                    self.assertTrue(all(r["end"] is not None for r in ledger.decoded_prefixes))
                ledger._process(6.)
                reads = [s for s in ledger.segments if s["phase"] == "read_block" and s["start"] > 2.]
                self.assertEqual(reads[0]["stripe"], 0 if fault else 1)
                self.assertFalse(ledger.log_loss)
                self.assertTrue(all(j["finished"] is not None for j in ledger.jobs))
                proof = next(j for j in ledger.jobs if j["kind"] == "proof")
                resumes = [e for e in ledger.scheduler_events if e["kind"] == "resume" and e["time"] >= proof["started"]]
                self.assertGreaterEqual(resumes[0]["time"], proof["finished"])

    def test_failed_proof_is_paid_and_retried_without_restoring_map(self):
        ledger = partial_wait(interrupted_proof=True)
        ledger._process(2.1)
        self.assertFalse(ledger.returns)
        self.assertTrue(any(j["status"] == "proof_failed" for j in ledger.jobs))
        ledger._repair(2.2, [])
        ledger._process(7.)
        self.assertTrue(ledger.returns)
        self.assertTrue(all(j["finished"] is not None for j in ledger.jobs))
        self.assertGreaterEqual(sum(s["volume"] for s in ledger.segments if s["phase"] == "proof"), 1.)

    def test_extra_target_for_full_buffer_owner_is_not_new_job(self):
        ledger = blocked(True, True)
        ledger._process(2.2)
        self.assertIn(0, ledger.buffers)
        ledger._declare(2, 2.2, "test")
        ledger._repair(2.2, [(0, [2], [5])])
        self.assertEqual(sum(j["kind"] == "repair" for j in ledger.jobs), 1)
        self.assertIn(2, ledger.jobs[0]["indices"])
        self.assertTrue(ledger.repair_admissions[-1]["permitted_for_processing"])
        self.assertEqual(ledger.repair_admissions[-1]["full_buffer_owner"], 0)

    def test_two_obstacles_need_both_changes_when_queue_is_full(self):
        outcomes = {}
        for reserve in (False, True):
            for bypass in (False, True):
                ledger = blocked(reserve, bypass)
                ledger._process(4.)
                outcomes[reserve, bypass] = all(j["finished"] is not None for j in ledger.jobs)
                self.assertFalse(ledger.log_loss)
                self.assertLessEqual(ledger.queue_report(4.)["max_outstanding_jobs"], 2)
                if reserve:
                    self.assertTrue(any(not r["permitted_for_processing"] for r in ledger.repair_admissions))
                if reserve and bypass:
                    self.assertTrue(ledger.returns)
                    self.assertTrue(any(s["phase"] == "proof" and s["volume"] > 0 for s in ledger.segments))
                    self.assertTrue(any(o[-1] == "written" for o in ledger.operation_outcomes))
        self.assertEqual(outcomes, {(False, False): False, (False, True): False,
                                   (True, False): False, (True, True): True})

    def test_bypass_alone_works_when_proof_already_fits(self):
        ledger = blocked(False, True, max_jobs=8)
        ledger._process(4.)
        self.assertTrue(all(j["finished"] is not None for j in ledger.jobs))
        self.assertTrue(ledger.read_wait_pauses)
        self.assertTrue(any(e["kind"] == "proof_bypass" for e in ledger.scheduler_events))

    def test_reserve_does_not_lie_about_outstanding_jobs(self):
        ledger = blocked(True, False)
        self.assertEqual(ledger._outstanding(), len(ledger.waiting)+int(ledger.active_job is not None))
        self.assertEqual(sum(j["kind"] == "repair" for j in ledger.jobs), 1)

    def test_reserve_needs_two_slots(self):
        with self.assertRaises(ValueError):
            fixture(lambda w, c, d: QueueControlLedger(w, c, d, QueueSettings(max_jobs=1),
                                                     control=QueueControlSettings(True, False)))


if __name__ == "__main__":
    unittest.main()
