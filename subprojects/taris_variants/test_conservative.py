import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd

from subprojects.taris_variants.conservative import GuardSettings, GuardRates, AvailabilityGuard, gamma_rate_upper
from subprojects.taris_variants.risk_graph import ObservedRiskGraph
from subprojects.taris_variants.run_guarded import run_case


class FakeExperience:
    N = 2
    t = 5
    reg = SimpleNamespace(dc=np.asarray([0, 1]), provider=np.asarray([0, 1]), region=np.asarray([0, 1]))
    join = np.asarray([0, 0])
    silent = []
    node_days_stored = np.asarray([4.0, 4.0])

    def intervals(self, t):
        return pd.DataFrame()

    def table(self, t, intervals):
        return pd.DataFrame({"node": [0, 1], "e": [5.0, 5.0]})

    def events(self, t, intervals):
        return pd.DataFrame(columns=["node", "total"])

    def shock_episodes(self, t, intervals):
        return pd.DataFrame(columns=["ctype"]), None


class ConservativeTests(unittest.TestCase):
    def rates(self):
        point = SimpleNamespace(fragment_loss=lambda nodes, t: np.full(len(nodes), 0.001),
                                shock_clusters=lambda nodes, t: [])
        return GuardRates(FakeExperience(), point, GuardSettings())

    def test_zero_events_do_not_produce_zero_risk(self):
        result = gamma_rate_upper([0, 0], [0, 100], 0.003, 30, 0.95)
        self.assertGreater(result[0], result[1])
        self.assertGreater(result[1], 0)

    def test_observed_loss_increases_parameter_scenario(self):
        a = gamma_rate_upper([0], [10], 0.003, 30, 0.95)
        b = gamma_rate_upper([1], [10], 0.003, 30, 0.95)
        self.assertGreater(b[0], a[0])

    def test_guard_never_lowers_individual_point_rate(self):
        rates = self.rates()
        self.assertTrue((rates.fragment_loss([0, 1], 5) >= 0.001).all())
        self.assertTrue(all(rate > 0 for _, _, rate, _ in rates.shock_clusters([0, 1], 5)))

    def test_future_audit_is_not_counted(self):
        rates = self.rates()
        rates.exp.silent = [(10, 0)]
        rates.refresh(5)
        self.assertEqual(rates.diagnostics["silent_events"], 0)

    def test_future_exposure_cannot_be_used_for_past_evaluation(self):
        rates = self.rates()
        with self.assertRaises(ValueError):
            rates.refresh(4)
        with self.assertRaises(ValueError):
            rates.refresh(6)

    def test_block_envelope_does_not_lower_availability_risk(self):
        registry = SimpleNamespace(N=2, dc=np.asarray([0, 1]))
        graph = ObservedRiskGraph(registry, samples=8)
        view = SimpleNamespace(step=None, registered=lambda: np.asarray([0, 1]))
        graph.update(0, view)
        guard = AvailabilityGuard(graph, GuardSettings(bootstrap_replicas=4))
        risk = guard.upper([0, 1], 1)
        self.assertGreaterEqual(risk, graph.object_unavailability([0, 1], 1))
        np.testing.assert_allclose(guard.weights.sum(axis=1), 1)

    def test_invalid_parameters(self):
        for settings in (GuardSettings(quantile=1), GuardSettings(prior_node_days=0),
                         GuardSettings(bootstrap_replicas=0), GuardSettings(block_hours=0)):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                settings.validate()
        with self.assertRaises(ValueError):
            gamma_rate_upper([float("nan")], [10], 0.003, 30, 0.95)

    def test_guarded_agents_execute_and_do_not_lower_point_estimates(self):
        case = run_case(42, "guard", days=16, nodes=16)
        self.assertTrue(case["world_unchanged"])
        self.assertTrue(case["common_random_numbers"])
        evaluated = 0
        for variant in case["variants"].values():
            self.assertEqual(variant["objects_admitted"], 16)
            for decision in variant["placement_decisions"]:
                selected = decision["selected"]
                self.assertGreaterEqual(selected["estimated_unavailability"], selected["point_unavailability"])
                self.assertIsNone(selected["object_confidence_level"])
                if selected["point_window_loss"] is not None:
                    evaluated += 1
                    self.assertGreaterEqual(selected["estimated_window_loss"], selected["point_window_loss"])
                if decision["requirements_passed"]:
                    self.assertLessEqual(selected["estimated_window_loss"], decision["window_loss_limit"])
        self.assertGreater(evaluated, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
