import unittest
from subprojects.taris_variants.state_forecast import fit_states, train_gated, state_prediction
from subprojects.taris_variants.diagnose_restricted import observed_state


def row(seed, state="repair_transfer", actual=.5):
    return {"seed": seed, "state": state, "known_work": 1., "capacity": 10., "actual_window_volume": actual}


class StateTests(unittest.TestCase):
    def test_many_rows_one_world_cannot_supply_credibility(self):
        models = fit_states([row(1) for _ in range(100)])
        self.assertEqual(models["repair_transfer"]["status"], "insufficient_support")
        self.assertEqual(state_prediction(row(1), models), 1.)

    def test_cross_validation_excludes_each_heldout_world(self):
        rows = [row(seed) for seed in (1, 2, 3) for _ in range(4)]
        model = train_gated(rows)
        self.assertTrue(model["use_state_model"])
        self.assertEqual(model["models"]["repair_transfer"]["coefficient"], .5)
        for fold in model["folds"]:
            self.assertNotIn(fold["heldout_seed"], fold["models"]["repair_transfer"]["support_worlds"])

    def test_no_support_has_no_optimistic_gate(self):
        model = train_gated([row(seed) for seed in (1, 2, 3)])
        self.assertFalse(model["use_state_model"])
        self.assertEqual(model["cv_baseline_mse"], model["cv_candidate_mse"])

    def test_parked_owner_has_priority_over_bypass_proof(self):
        snapshot = {"jobs": [{"kind": "repair"}], "parked_job": 0, "full_buffer_objects": [2],
                    "planned_channel_phase": {"phase": "proof"}}
        self.assertEqual(observed_state(snapshot), "parked_full_buffer")
        snapshot["full_buffer_objects"] = []
        self.assertEqual(observed_state(snapshot), "proof_transfer")


if __name__ == "__main__":
    unittest.main()
