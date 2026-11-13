import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from copy import deepcopy
from types import SimpleNamespace
import unittest
import pandas as pd

from subprojects.taris_variants.run_compare import exposure_metrics
from subprojects.taris_variants.run_series import summarize, ablations, fingerprint


class ExposureTests(unittest.TestCase):
    def result(self):
        objects = pd.DataFrame({"created_epoch": [0, 1, 3], "lost_t": [2, float("inf"), 3]})
        return SimpleNamespace(objects=objects, unavailability=lambda n_objects: pd.DataFrame(
            {"obj": [0, 1], "unavail_share": [0.25, 0]}))

    def test_time_after_loss_is_not_dropped(self):
        metrics = exposure_metrics(self.result(), 4)
        self.assertEqual(metrics["object_days"], 8)
        self.assertEqual(metrics["lost_object_days"], 3)
        self.assertEqual(metrics["unavailable_object_days"], 3.5)
        self.assertEqual(metrics["unavailability_share"], 3.5 / 8)
        self.assertEqual(metrics["lost_objects"], 2)

    def test_post_warmup_cohort_and_empty_cohort(self):
        self.assertEqual(exposure_metrics(self.result(), 4, 2)["unavailability_share"], 1)
        self.assertIsNone(exposure_metrics(self.result(), 4, 4)["unavailability_share"])

    def test_missing_pre_loss_estimate_is_an_error(self):
        result = self.result()
        result.unavailability = lambda n_objects: pd.DataFrame({"obj": [0], "unavail_share": [0.25]})
        with self.assertRaises(ValueError):
            exposure_metrics(result, 4)


def case(exposure=1, unavailable=0.5, mode="fixed"):
    value = {"objects": 1, "lost_objects": 0, "repair_traffic": 2,
             "fragment_days": 4, "runtime_warnings": {}, "seconds": 1,
             "full_exposure": {"object_days": exposure, "unavailable_object_days": unavailable,
                               "unavailability_share": unavailable / exposure},
             "post_warmup_cohort": {"object_days": exposure, "unavailable_object_days": unavailable},
             "initial_placements": [{"obj": 0, "nodes": [0, 1, 2]}]}
    return {"scenario": "S0", "seed": 1, "settings": {"n": 3, "repair_mode": mode},
            "world_hashes": {"world": "same"}, "input_hashes": {"input": "same"},
            "variants": {key: deepcopy(value) for key in ("classic", "graph", "hybrid")}}


class SummaryTests(unittest.TestCase):
    def test_aggregation_is_exposure_weighted(self):
        result = summarize([case(), case(9, 0)])[0]
        self.assertEqual(result["variants"]["classic"]["unavailability_share"], 0.05)
        self.assertEqual(result["distinct_worlds"], 1)

    def test_graph_is_a_control_for_repair_mode(self):
        a, b = case(), case(mode="thiele")
        b["variants"]["hybrid"]["repair_traffic"] = 3
        result = ablations([a, b])[0]
        self.assertTrue(result["variants"]["graph"]["control_identical"])
        self.assertEqual(result["variants"]["hybrid"]["repair_traffic_thiele_minus_fixed"], 1)

    def test_absent_actuarial_model_is_not_zero_violations(self):
        row = case()
        row["variants"]["classic"]["placement_risk"] = {
            "checked": 10, "violations": 3, "unscored": 2}
        result = summarize([row])[0]["variants"]
        self.assertEqual(result["classic"]["actuarial_violations"], 3)
        self.assertEqual(result["classic"]["actuarial_unscored"], 2)
        self.assertIsNone(result["graph"]["actuarial_violations"])

    def test_different_worlds_cannot_be_paired_as_mode_ablation(self):
        a, b = case(), case(mode="thiele")
        b["world_hashes"]["world"] = "different"
        self.assertNotEqual(fingerprint(a), fingerprint(b))
        with self.assertRaises(AssertionError):
            ablations([a, b])


if __name__ == "__main__":
    unittest.main(verbosity=2)
