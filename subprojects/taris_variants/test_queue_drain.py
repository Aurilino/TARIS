import unittest
from dataclasses import replace
from copy import deepcopy
import numpy as np
from subprojects.taris_variants.queue_drain import QueueDrainLedger
from subprojects.taris_variants.queue_control import QueueControlSettings
from subprojects.taris_variants.queued import QueueSettings
from subprojects.taris_variants.resumable import TransportSettings
from subprojects.taris_variants.test_physical import fixture
from subprojects.taris_variants.test_queue_control import audit_fixture
from subprojects.taris_variants.test_shared_channel import prepared
from subprojects.taris_variants.audit_queue_drain_stress import audit_stress_record


def pressure_case(enabled=True, max_jobs=2):
    ledger = fixture(lambda w, c, d: QueueDrainLedger(w, c, d,
        QueueSettings(rate_size_per_hour=1/24, max_jobs=max_jobs), TransportSettings(stripes=4),
        control=QueueControlSettings(True, True), drain_irrelevant_proof=enabled))
    ledger.cfg = replace(ledger.cfg, repair_hours=0.)
    ledger.w.cfg = ledger.cfg
    ledger._add_fragment(0, 2, 4, 0.)
    for node, until in ((0, 8.), (1, 1.), (4, .8)):
        ledger.down[node] = ledger.obs_iv[node] = np.array([.5]), np.array([until])
    for f in (0, 1, 3):
        ledger._declare(f, .5, "простой > T_rep")
    ledger._world_events()
    ledger._repair(.6, [(0, [0], [3])])
    return ledger


class QueueDrainTests(unittest.TestCase):
    def test_auditor_rejects_fallback_with_fabricated_free_slot(self):
        ledger = pressure_case(True)
        record = audit_fixture(ledger)
        record["drain_irrelevant_proof"] = True
        result = audit_stress_record(record, ledger.cfg, ledger.w, ledger.ch, ledger.data)
        self.assertEqual(result["irrecoverable_objects"], 0)
        bad = deepcopy(record)
        selected = next(r for r in bad["proof_selection_log"] if r.get("selection_rule") == "pressure_fallback")
        selected["outstanding"] = 1
        with self.assertRaises(AssertionError):
            audit_stress_record(bad, ledger.cfg, ledger.w, ledger.ch, ledger.data)

    def test_hidden_integrity_does_not_change_fallback_choice(self):
        ledger = pressure_case(True)
        ledger._process(.8)
        ledger._enqueue({"kind": "proof", "node": 4}, .8)
        alternative = deepcopy(ledger)
        alternative.f_exist[3] = False
        alternative.alive_to = alternative.alive_to.copy()
        alternative.alive_to[4] = 0.
        ledger._pump(.8)
        alternative._pump(.8)
        self.assertEqual(ledger.proof_selection_log, alternative.proof_selection_log)
        self.assertEqual(ledger.active_job, alternative.active_job)
        self.assertEqual(ledger.proof_selection_log[-1]["selection_rule"], "pressure_fallback")

    def test_irrelevant_proof_releases_reserved_slot_for_needed_proof(self):
        for enabled in (False, True):
            ledger = pressure_case(enabled)
            ledger._process(4.)
            self.assertEqual(all(j["finished"] is not None for j in ledger.jobs), enabled)
            self.assertFalse(ledger.log_loss)
            if enabled:
                selections = [r for r in ledger.proof_selection_log if r.get("selection_rule") == "pressure_fallback"]
                self.assertEqual(len(selections), 1)
                self.assertEqual(selections[0]["outstanding"], 2)
                self.assertTrue(all(not c["relevant_indices"] for c in selections[0]["candidates"]))
                self.assertTrue(any(r[1:] == (0, 1, 1) for r in ledger.returns))
                self.assertTrue(any(s["phase"] == "proof" and s["volume"] > 0 for s in ledger.segments))

    def test_free_slot_does_not_allow_fallback(self):
        ledger = pressure_case(True, max_jobs=3)
        ledger._process(.9)
        self.assertFalse(any(r.get("selection_rule") == "pressure_fallback" for r in ledger.proof_selection_log))
        self.assertIsNone(ledger.active_job)
        self.assertEqual(ledger._outstanding(), 2)
        ledger._process(4.)
        self.assertFalse(ledger.log_loss)

    def test_relevant_candidate_keeps_priority(self):
        ledger = pressure_case(True, max_jobs=3)
        ledger._process(1.01)
        selected = [r for r in ledger.proof_selection_log if r["selected"] is not None]
        self.assertTrue(selected)
        self.assertNotIn("selection_rule", selected[0])
        self.assertTrue(any(c["relevant_indices"] for c in selected[0]["candidates"]))

    def test_full_buffer_pause_never_uses_pressure_fallback(self):
        ledger = prepared(lambda w, c, d, q, tr: QueueDrainLedger(w, c, d, replace(q, max_jobs=2), tr,
            control=QueueControlSettings(True, True), drain_irrelevant_proof=True), return_node=False)
        ledger._process(3.01)
        self.assertIsNotNone(ledger.parked_job)
        ledger._enqueue({"kind": "proof", "node": 5}, 3.01)
        self.assertEqual(ledger._outstanding(), ledger.queue_settings.max_jobs)
        ledger._pump(3.01)
        self.assertIsNone(ledger.active_job)
        self.assertFalse(any(r.get("selection_rule") == "pressure_fallback" for r in ledger.proof_selection_log))
        # Даже ошибочный ярлык паузы не должен скрыть реальный полный буфер.
        ledger.parked_intervals[-1]["pause_type"] = "read_wait"
        ledger._pump(3.01)
        self.assertIsNone(ledger.active_job)
        self.assertFalse(any(r.get("selection_rule") == "pressure_fallback" for r in ledger.proof_selection_log))


if __name__ == "__main__":
    unittest.main()
