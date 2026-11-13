import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import json
import unittest
import numpy as np
from subprojects.taris_variants.observable_memory import ObservableMemoryLedger
from subprojects.taris_variants.targeted_proof import TargetedProofLedger
from subprojects.taris_variants.test_shared_channel import prepared
from subprojects.taris_variants.test_physical import fixture


class ObservableMemoryTests(unittest.TestCase):
    def prefix_ledger(self):
        ledger = fixture(ObservableMemoryLedger)
        ledger.down[0] = ledger.obs_iv[0] = np.array([0.]), np.array([8.])
        ledger._declare(0, 0., "test")
        ledger._repair(0., [(0, [0], [3])])
        ledger._process(.09)
        return ledger

    def test_partial_prefix_converts_atomically(self):
        ledger = self.prefix_ledger()
        partial = ledger.queue_observation(.09)["local_memory"]
        self.assertTrue(partial)
        self.assertTrue(all(r["kind"] == "decoded_block" for r in partial))
        self.assertGreater(sum(r["size"] for r in partial), 0.)
        self.assertLess(sum(r["size"] for r in partial), 1.)
        ledger._process(.105)
        full = ledger.queue_observation(.105)["local_memory"]
        self.assertEqual([r["kind"] for r in full], ["full"])
        self.assertEqual(sum(r["size"] for r in full), 1.)

    def test_partial_prefix_lost_on_observed_local_fault(self):
        ledger = self.prefix_ledger()
        ledger._push(.09001, "queue_buffer_fail", 0)
        ledger._process(.09002)
        snapshot = ledger.queue_observation(.09002)
        self.assertEqual(snapshot["local_memory"], [])
        self.assertEqual(snapshot["memory_notifications"][0]["time"], .09001)

    def test_prefix_conversion_has_no_double_allocation(self):
        ledger = prepared(ObservableMemoryLedger, return_node=False)
        # prepared уже дошёл до полного буфера; префиксы должны быть закрыты.
        snapshot = ledger.queue_observation(3.)
        self.assertEqual(sum(r["size"] for r in snapshot["local_memory"]), 1.)
        self.assertEqual([r["kind"] for r in snapshot["local_memory"]], ["full"])
        self.assertEqual(snapshot["k"], 2)
        self.assertEqual(snapshot["object_sizes"], [{"obj": 0, "size": 1.}])
        json.dumps(snapshot, allow_nan=False)

    def test_memory_failure_while_proof_owns_channel(self):
        ledger = prepared(ObservableMemoryLedger, proof_fraction=1.)
        ledger._push(3.7, "queue_buffer_fail", 0)
        ledger._process(3.8)
        snapshot = ledger.queue_observation(3.8)
        self.assertEqual(snapshot["local_memory"], [])
        self.assertEqual(snapshot["memory_notifications"],
                         [{"obj": 0, "time": 3.7, "kind": "local_memory_failure"}])
        self.assertIsNotNone(ledger.parked_job)
        self.assertNotEqual(ledger.active_job, ledger.parked_job)

    def test_closing_responses_hide_source_ids_and_future(self):
        ledger = prepared(ObservableMemoryLedger)
        ledger._process(5.)
        snapshot = ledger.queue_observation(5.)
        self.assertTrue(snapshot["closing_acknowledgements"])
        self.assertTrue(any(r["ok"] for r in snapshot["closing_acknowledgements"]))
        self.assertTrue(all(set(r) == {"job", "time", "ok"}
                            for r in snapshot["closing_acknowledgements"]))
        snapshot["closing_acknowledgements"].clear()
        self.assertTrue(ledger.queue_observation(5.)["closing_acknowledgements"])

    def test_hidden_receiver_state_does_not_change_local_memory(self):
        ledger = prepared(ObservableMemoryLedger)
        before = ledger.queue_observation(3.)
        ledger.f_exist = [False] * len(ledger.f_exist)
        ledger.disk_generation[:] = 999
        for row in ledger.receiver_prefixes:
            row["end"] = 0.
        self.assertEqual(before, ledger.queue_observation(3.))

    def test_fault_notifications_do_not_change_executor(self):
        baseline = prepared(TargetedProofLedger, proof_fraction=1.)
        observed = prepared(ObservableMemoryLedger, proof_fraction=1.)
        for ledger in (baseline, observed):
            ledger._push(3.1, "queue_buffer_fail", 0)
            ledger._process(8.)
        observed.queue_observation(8.)
        for key in ("jobs", "segments", "block_results", "buffer_intervals", "decoded_prefixes",
                    "close_checks", "scheduler_events", "log_loss"):
            self.assertEqual(getattr(baseline, key), getattr(observed, key), key)


if __name__ == "__main__":
    unittest.main()
