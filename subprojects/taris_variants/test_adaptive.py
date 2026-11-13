import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from subprojects.taris_variants.adaptive import AdaptiveAgent, SizingRequirements, object_size
from subprojects.taris_variants.run_adaptive import run_case


def fixture(requirements=None, fitted=True):
    agent = AdaptiveAgent.__new__(AdaptiveAgent)
    agent.requirements = requirements or SizingRequirements()
    agent.variant, agent.k = "graph", 2
    agent.model = SimpleNamespace(fitted=fitted)
    agent.world = SimpleNamespace(cfg=SimpleNamespace(T_rep=1, repair_hours=2))
    agent.rates = object()
    agent.graph = SimpleNamespace(object_unavailability=lambda nodes, k: 0.01,
                                  score=lambda nodes: float(len(nodes)))
    agent._observed_hours = np.full(10, 100)
    agent.cache, agent.placement_log, agent.status_counts = {}, [], Counter()
    agent.placement_risk = {"checked": 0, "feasible": 0, "violations": 0, "unscored": 0}
    agent._propose = lambda candidates, n: list(candidates[:n])
    return agent


def loss_estimate(_rates, nodes, *_args):
    return (0.0001 if len(nodes) == 3 else 0.00001), 0.0, float(len(nodes))


class SizingTests(unittest.TestCase):
    def test_selects_first_feasible_size_and_does_not_claim_proof(self):
        agent = fixture()
        with patch("subprojects.taris_variants.adaptive.placement_risk_metrics", side_effect=loss_estimate):
            decision = agent._choose(0, 2, SimpleNamespace(candidates=lambda: np.arange(8)))
        self.assertEqual(decision["n"], 4)
        self.assertEqual(decision["status"], "estimated_feasible")
        self.assertTrue(decision["requirements_passed"])
        self.assertFalse(decision["minimum_proven"])

    def test_cold_start_uses_maximum_and_is_unscored(self):
        agent = fixture(fitted=False)
        decision = agent._choose(0, 0, SimpleNamespace(candidates=lambda: np.arange(8)))
        self.assertEqual(decision["n"], 6)
        self.assertFalse(decision["scored"])
        self.assertFalse(decision["requirements_passed"])
        self.assertEqual(decision["status"], "unscored_cold_start")

    def test_new_node_cannot_be_treated_as_observed(self):
        agent = fixture()
        agent._observed_hours[0] = 0
        with patch("subprojects.taris_variants.adaptive.placement_risk_metrics", side_effect=loss_estimate):
            decision = agent._choose(0, 5, SimpleNamespace(candidates=lambda: np.arange(8)))
        self.assertEqual(decision["status"], "unscored_cold_start")
        self.assertEqual(decision["n"], 6)

    def test_failed_cap_is_not_success(self):
        agent = fixture(SizingRequirements(min_n=3, max_n=3))
        with patch("subprojects.taris_variants.adaptive.placement_risk_metrics", side_effect=loss_estimate):
            decision = agent._choose(0, 5, SimpleNamespace(candidates=lambda: np.arange(8)))
        self.assertEqual(decision["status"], "no_feasible_found_best_effort")
        self.assertFalse(decision["requirements_passed"])
        self.assertTrue(decision["scored"])

    def test_rejection_distinguished_from_unscored(self):
        agent = fixture(SizingRequirements(min_n=3, max_n=3, fallback="reject"))
        with patch("subprojects.taris_variants.adaptive.placement_risk_metrics", side_effect=loss_estimate):
            decision = agent._choose(0, 5, SimpleNamespace(candidates=lambda: np.arange(8)))
        self.assertFalse(decision["nodes"])
        self.assertTrue(decision["scored"])
        self.assertTrue(decision["status"].startswith("rejected_"))

    def test_capacity_and_search_failure_differ(self):
        agent = fixture()
        short = agent._choose(0, 5, SimpleNamespace(candidates=lambda: np.arange(2)))
        self.assertEqual(short["status"], "insufficient_capacity")
        agent._propose = lambda candidates, n: []
        blocked = agent._choose(0, 5, SimpleNamespace(candidates=lambda: np.arange(8)))
        self.assertEqual(blocked["status"], "no_placement_found")

    def test_target_query_does_not_repeat_search(self):
        agent = fixture()
        with patch.object(agent, "_choose", wraps=agent._choose) as choose, patch(
                "subprojects.taris_variants.adaptive.placement_risk_metrics", side_effect=loss_estimate):
            view = SimpleNamespace(candidates=lambda: np.arange(8))
            self.assertEqual(agent.target_fragments(0, 2, view), 4)
            self.assertEqual(agent.target_fragments(0, 2, view), 4)
            self.assertEqual(len(agent.place(0, 2, view)), 4)
            self.assertEqual(choose.call_count, 1)
        self.assertEqual(agent.placement_risk["feasible"], 1)

    def test_threshold_and_memo_are_isolated_by_object_size(self):
        policy = SimpleNamespace(n=6, R_fixed=3, memo={"initial": 1})
        original = policy.memo
        with object_size(policy, 3, 2):
            self.assertEqual(policy.n, 3)
            self.assertEqual(policy.R_fixed, 0)
            policy.memo["risk"] = "n3"
        with object_size(policy, 4, 2):
            self.assertNotIn("risk", policy.memo)
            policy.memo["risk"] = "n4"
        with object_size(policy, 3, 2):
            self.assertEqual(policy.memo["risk"], "n3")
        self.assertIs(policy.memo, original)
        self.assertEqual((policy.n, policy.R_fixed), (6, 3))

    def test_invalid_requirements(self):
        for requirements in (SizingRequirements(min_n=2), SizingRequirements(max_n=2),
                             SizingRequirements(trials=0), SizingRequirements(window_loss_limit=0)):
            with self.subTest(requirements=requirements), self.assertRaises(ValueError):
                requirements.validate(2)


class AdaptiveIntegrationTests(unittest.TestCase):
    def test_three_agents_execute_selected_sizes(self):
        case = run_case(41, "adaptive", days=8, nodes=16, scenario_name="S0")
        self.assertTrue(case["world_unchanged"])
        self.assertTrue(case["common_random_numbers"])
        for row in case["variants"].values():
            self.assertEqual(row["objects_requested"], 8)
            self.assertEqual(row["objects_admitted"], 8)
            self.assertTrue(all(3 <= n <= 6 for n in row["final_sizes"]))
            self.assertEqual(sum(row["status_counts"].values()), 8)
            self.assertTrue(all(not d["requirements_passed"] for d in row["placement_decisions"]
                                if d["status"] != "estimated_feasible"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
