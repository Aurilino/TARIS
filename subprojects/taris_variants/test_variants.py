import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from unittest.mock import patch

from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.policy import ReservePolicy
from taris_sim.risk import EstimatedRates
from subprojects.taris_variants.agents import VariantSettings, make_agent
from subprojects.taris_variants.risk_graph import ObservedRiskGraph
from subprojects.taris_variants.run_compare import run_case, source_hashes


class PublicView:
    def __init__(self, t1=1, joins=None):
        self.step = {"t0": 0, "t1": t1,
            "outages_ended": pd.DataFrame([(0, 0, 0.5)], columns=["node", "start", "end"]),
            "outages_ongoing": pd.DataFrame([], columns=["node", "start"])}
        self.joins = np.asarray([0, 0, 1] if joins is None else joins, float)

    def join_times(self):
        return self.joins.copy()

    def registered(self):
        return np.flatnonzero(self.joins <= 1)

    @property
    def L(self):
        raise AssertionError("Граф не должен получать ledger")


def graph():
    registry = SimpleNamespace(N=3, dc=np.asarray([0, 0, 1]))
    return ObservedRiskGraph(registry, samples=16)


class GraphTests(unittest.TestCase):
    def test_no_future_and_no_ledger_access(self):
        model = graph()
        with self.assertRaises(ValueError):
            model.update(0, PublicView())
        model.update(1, PublicView())
        self.assertEqual(len(model.rows), 24)
        self.assertGreater(model.own[2], 0)
        self.assertEqual(model.fitted_at, 1)

    def test_same_slice_is_not_counted_twice(self):
        model = graph()
        model.update(1, PublicView())
        first = model.pairs.copy()
        model.update(1, PublicView())
        self.assertEqual(len(model.rows), 24)
        np.testing.assert_array_equal(first, model.pairs)

    def test_pair_and_object_probabilities_are_coherent(self):
        model = graph()
        model.update(1, PublicView())
        self.assertAlmostEqual(model.weights.sum(), 1)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertLessEqual(model.pairs[i, j], min(model.own[i], model.own[j]) + 1e-12)
                self.assertAlmostEqual(model.pairs[i, j], model.object_unavailability([i, j], 1))

    def test_unregistered_nodes_have_no_public_scores(self):
        model = graph()
        model.update(1, PublicView(joins=[0, 0, np.inf]))
        self.assertTrue(np.isnan(model.own[2]))
        self.assertTrue(np.isnan(model.pairs[2]).all())

    def test_unknown_future_population_does_not_change_known_risks(self):
        small = graph()
        large = ObservedRiskGraph(SimpleNamespace(N=4, dc=np.asarray([0, 0, 999, 888])), samples=16)
        small.update(1, PublicView(joins=[0, 0, np.inf]))
        large.update(1, PublicView(joins=[0, 0, np.inf, np.inf]))
        np.testing.assert_allclose(small.own[:2], large.own[:2], atol=1e-15)
        np.testing.assert_allclose(small.pairs[:2, :2], large.pairs[:2, :2], atol=1e-15)

    def test_history_window_is_bounded(self):
        model = graph()
        model.window_days = 1
        model.update(1, PublicView())
        view = PublicView()
        view.step = None
        model.update(3, view)
        self.assertEqual(len(model.rows), 0)

    def test_positive_cold_risk_does_not_depend_on_sample_luck(self):
        model = graph()
        model.samples = 1
        view = PublicView()
        view.step = None
        model.update(0, view)
        self.assertGreater(model.own[0], 0)
        self.assertGreater(model.own[1], 0)


class RefinementTests(unittest.TestCase):
    def fixture(self):
        from subprojects.taris_variants.agents import GraphRefinement
        policy = GraphRefinement()
        policy.settings = VariantSettings(n=3, candidate_limit=4, swap_rounds=1, availability_limit=1)
        policy.k = 2
        policy.dc = policy.prov = policy.reg = policy.own = policy.subnet_domain = np.arange(4)
        class FakeGraph:
            fitted_at = 1
            own = np.asarray([0.2, 0.2, 0.2, 0.1])
            def score(self, nodes):
                return 0.1 if 3 in nodes else 0.5
            def object_unavailability(self, nodes, k):
                return 0.01
        policy.graph = FakeGraph()
        policy.model = SimpleNamespace(fitted=True)
        policy.world = SimpleNamespace(cfg=SimpleNamespace(T_rep=1, repair_hours=2))
        policy.rates = object()
        policy.graph_stats = {"calls": 0, "accepted_swaps": 0, "actuarial_rejections": 0,
                              "availability_rejections": 0, "unavailability_exceeded": 0, "cold_calls": 0}
        policy.decision_log = []
        return policy

    def test_hybrid_rejects_graph_gain_when_actuarial_risk_increases(self):
        policy = self.fixture()
        def loss(_rates, nodes, *_args):
            return (0.2 if 3 in nodes else 0.1), 0
        with patch("subprojects.taris_variants.agents.placement_loss_probability", side_effect=loss):
            selected = policy._refine([0, 1, 2], range(4), (), 1, actuarial=True)
        self.assertEqual(selected, [0, 1, 2])
        self.assertGreater(policy.graph_stats["actuarial_rejections"], 0)

    def test_repair_never_moves_existing_holders(self):
        policy = self.fixture()
        selected = policy._refine([1, 2], range(4), {0}, 1)
        self.assertNotIn(0, selected)
        self.assertIn(3, selected)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = World(scenario("S0", seed=31, horizon_days=8, train_days=4,
            n_initial_nodes=16, node_join_rate=0.1, n_dc=4, n_providers=3, n_regions=2,
            blocks_per_epoch=1, k=2))

    def test_factory_preserves_classic_and_does_not_use_true_rates(self):
        settings = VariantSettings(n=4, repair_mode="fixed")
        classic = make_agent("classic", self.world, settings)
        graph_policy = make_agent("graph", self.world, settings)
        hybrid = make_agent("hybrid", self.world, settings)
        self.assertIs(type(classic), ReservePolicy)
        self.assertIsInstance(classic.rates, EstimatedRates)
        self.assertIsInstance(hybrid.rates, EstimatedRates)
        self.assertFalse(hasattr(graph_policy, "rates"))
        self.assertEqual({p.name for p in (classic, graph_policy, hybrid)},
                         {"TARIS Classic", "TARIS-G", "TARIS-Hybrid"})

    def test_invalid_variant_and_budget(self):
        with self.assertRaises(ValueError):
            make_agent("unknown", self.world, VariantSettings(n=4))
        with self.assertRaises(ValueError):
            make_agent("graph", self.world, VariantSettings(n=2))

    def test_three_variants_execute_with_same_world(self):
        before = source_hashes()
        case = run_case("S0", 32, days=8, initial_nodes=16,
                        settings=VariantSettings(n=4, block_days=7, repair_mode="fixed"))
        self.assertTrue(case["common_random_numbers"])
        self.assertTrue(case["world_unchanged"])
        self.assertEqual(before, source_hashes())
        self.assertEqual(set(case["variants"]), {"classic", "graph", "hybrid"})
        for result in case["variants"].values():
            self.assertTrue(result["numerical_finite"])
            self.assertEqual(result["objects"], 8)
        for kind in ("graph", "hybrid"):
            decisions = case["variants"][kind]["graph_decisions"]
            self.assertTrue(decisions)
            self.assertTrue(all(row["graph_after"] <= row["graph_before"] + 1e-12 for row in decisions))
            self.assertTrue(all(len(set(row["chosen"])) == len(row["chosen"]) for row in decisions))
            actual = case["variants"][kind]["initial_placements"]
            logged = case["variants"][kind]["placement_decisions"]
            self.assertEqual([row["nodes"] for row in actual], [row["nodes"] for row in logged])


if __name__ == "__main__":
    unittest.main(verbosity=2)
