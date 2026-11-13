import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace
from copy import deepcopy
import json
import unittest
import numpy as np
from subprojects.taris_variants.observable_queue import ObservableQueueLedger, ObservingPolicy
from subprojects.taris_variants.targeted_proof import TargetedProofLedger
from subprojects.taris_variants.test_shared_channel import prepared
from subprojects.taris_variants.test_physical import fixture


class ObservationTests(unittest.TestCase):
    def test_current_only_and_no_midprocessing(self):
        ledger = prepared(ObservableQueueLedger)
        ledger.queue_observation(3.)
        for t in (0., 2., 4., float("nan")):
            with self.assertRaises(ValueError):
                ledger.queue_observation(t)
        ledger.observation_ready = False
        with self.assertRaises(ValueError):
            ledger.queue_observation(3.)

    def test_hidden_truth_and_progress_do_not_change_snapshot(self):
        ledger = prepared(ObservableQueueLedger)
        before = ledger.queue_observation(3.)
        ledger.f_exist = [False] * len(ledger.f_exist)
        ledger.f_dead = [0.] * len(ledger.f_dead)
        ledger.alive_to = np.zeros_like(ledger.alive_to)
        ledger.disk_generation[:] = 123
        ledger.memory_generation[:] = 456
        ledger.obj_lost_t[:] = 0.
        ledger.down = {}
        for job in ledger.jobs:
            job["write_progress"] = 999
        self.assertEqual(before, ledger.queue_observation(3.))

    def test_copy_and_explicit_plan_not_future_outcome(self):
        ledger = prepared(ObservableQueueLedger)
        observation = ledger.queue_observation(3.)
        json.dumps(observation, allow_nan=False)
        self.assertTrue(all(r["time"] <= 3. for r in observation["block_acknowledgements"]))
        for row in observation["jobs"]:
            self.assertNotIn("write_progress", row)
            self.assertNotIn("finished", row)
            self.assertNotIn("status", row)
        observation["jobs"].clear()
        self.assertTrue(ledger.queue_observation(3.)["jobs"])
        ledger._process(3.005)
        plan = ledger.queue_observation(3.005)["planned_channel_phase"]
        self.assertIsNotNone(plan)
        self.assertGreater(plan["end"], 3.005)
        self.assertNotIn("disk_generation", plan)

    def test_observation_leaves_transport_outcomes_unchanged(self):
        baseline = prepared(TargetedProofLedger)
        observed = prepared(ObservableQueueLedger)
        for t in (3., 3.02, 3.5, 4., 5., 8.):
            observed._process(t)
            observed.queue_observation(t)
        baseline._process(8.)
        for key in ("jobs", "segments", "block_results", "scheduler_events", "buffer_intervals",
                    "proof_selection_log", "operation_outcomes", "log_loss"):
            self.assertEqual(getattr(baseline, key), getattr(observed, key), key)

    def test_policy_wrapper_observes_before_decision_and_preserves_results(self):
        results = []
        for observe in (False, True):
            ledger = fixture(ObservableQueueLedger if observe else TargetedProofLedger)
            ledger.data.new_objects = lambda epoch: np.array([0]) if epoch == 0 else np.array([], int)
            ledger.ch.reset = lambda: None
            ledger.ch.step = lambda: {"seed": "00" * 32}
            policy = SimpleNamespace(n=3, name="control", alarm=lambda t, v: [],
                on_epoch=lambda t, v: [], migrate=lambda t, v: [], release=lambda t, v: [],
                place=lambda obj, t, v: [0, 1, 2])
            if observe:
                policy = ObservingPolicy(policy, ledger)
            result = ledger.run(policy)
            results.append((deepcopy(ledger.initial_writes), result.daily.to_dict(), result.policy_name))
            if observe:
                self.assertEqual([r["time"] for r in policy.snapshots], list(map(float, range(8))))
                self.assertEqual(policy.snapshots[0]["jobs"], [])
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
