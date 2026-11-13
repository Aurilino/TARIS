import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from dataclasses import asdict
import unittest
import numpy as np
import pandas as pd
from copy import deepcopy
from subprojects.taris_variants.targeted_proof import TargetedProofLedger, relevant_indices
from subprojects.taris_variants.test_shared_channel import prepared
from subprojects.taris_variants.test_physical import fixture
from subprojects.taris_variants.queued import QueueSettings
from subprojects.taris_variants.resumable import TransportSettings
from subprojects.taris_variants.audit_targeted_proof import audit_record, audit_schedule


class TargetedProofTests(unittest.TestCase):
    def test_independent_audit_requires_past_declared_holder(self):
        L = prepared(TargetedProofLedger)
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
            "proof_selection_log": L.proof_selection_log, "declarations": L.declarations,
            "physical_fragments": [{"obj": L.f_obj[f], "idx": L.f_idx[f], "node": L.f_node[f],
                "start": L.f_from[f], "physical_end": L.f_dead[f] if np.isfinite(L.f_dead[f]) else None,
                "allocated": L.allocated[f]} for f in range(len(L.f_obj))]}
        self.assertEqual(audit_record(record, L.cfg, L.w, L.ch, L.data)["irrecoverable_objects"], 0)
        bad = deepcopy(record)
        row = next(r for r in bad["proof_selection_log"] if r["selected"] is not None)
        row["candidates"][0]["known_holders"][0]["declared_at"] = row["time"]+1.
        with self.assertRaises(AssertionError):
            audit_schedule(bad, 8.)

    def test_oldest_relevant_proof_is_selected_after_irrelevant_head(self):
        L = prepared(TargetedProofLedger, return_node=False)
        L._process(3.01)
        self.assertIsNotNone(L.parked_job)
        L._enqueue({"kind": "proof", "node": 5}, 3.01)
        first = len(L.jobs)-1
        L._enqueue({"kind": "proof", "node": 4}, 3.01)
        relevant = len(L.jobs)-1
        L._pump(3.01)
        self.assertEqual(L.active_job, relevant)
        self.assertEqual(list(L.waiting), [first])
        self.assertEqual(L.proof_selection_log[-1]["selected"], relevant)

    def test_relevant_proof_unlocks_owner(self):
        L = prepared(TargetedProofLedger)
        L._process(5.)
        self.assertFalse(L.buffers)
        self.assertFalse(L.log_loss)
        selections = [r for r in L.proof_selection_log if r["selected"] is not None]
        self.assertTrue(selections)
        self.assertEqual(selections[0]["candidates"][0]["relevant_indices"], [1])

    def test_hidden_integrity_does_not_change_relevance(self):
        L = prepared(TargetedProofLedger, return_node=False)
        L._process(3.01)
        owner = L.parked_job
        self.assertIsNotNone(owner)
        # Кандидат создаётся без запуска, чтобы сравнить один публичный снимок.
        L.verification_pending.add(4)
        L._enqueue({"kind": "proof", "node": 4}, 3.01)
        before = L._candidates(owner)
        old = L.f_exist[3]
        old_alive = L.alive_to
        L.f_exist[3] = not old
        L.alive_to = old_alive.copy()
        L.alive_to[4] = 0.
        self.assertEqual(before, L._candidates(owner))
        L.f_exist[3] = old
        L.alive_to = old_alive

    def test_irrelevant_proof_waits_for_normal_fifo_service(self):
        L = fixture(lambda w, c, d: TargetedProofLedger(w, c, d,
            QueueSettings(rate_size_per_hour=1/24), TransportSettings(stripes=4)))
        L.cfg = replace(L.cfg, repair_hours=0.)
        L.w.cfg = L.cfg
        L._add_fragment(0, 1, 4, 0.)
        L.down[4] = L.obs_iv[4] = np.array([.5]), np.array([2.26])
        L._declare(3, .5, "простой > T_rep")
        L.down[0] = L.obs_iv[0] = np.array([.5]), np.array([7.])
        L.down[3] = np.array([2.15]), np.array([2.6])
        L._declare(0, .5, "простой > T_rep")
        L._repair(1., [(0, [0], [3])])
        L._process(2.26)
        L._return_node(4, 2.26)
        L._process(4.)
        self.assertTrue(any(r["selected"] is None for r in L.proof_selection_log))
        self.assertFalse(any(r["kind"] == "proof_bypass" for r in L.scheduler_events))
        self.assertTrue(any(j["kind"] == "proof" and j["status"] == "proof_completed" for j in L.jobs))

    def test_pure_rule_deduplicates_and_ignores_unknown_parameters(self):
        known = [{"idx": 1, "truth": False}, {"idx": 1, "future_leave": 0}, {"idx": 3}]
        self.assertEqual(relevant_indices({1, 2}, known), [1])


if __name__ == "__main__":
    unittest.main()
