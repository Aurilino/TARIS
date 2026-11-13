from fractions import Fraction
import gc
import unittest
import weakref

from exact_model import ExactModel
from graph_search import graph_search
from learning import NodeSpec, Observation, ObservationStore, fit
from uncertainty import RobustModel, bootstrap_weights, build_ensembles, stress_prior


def simple(risk):
    return ExactModel(["A", "B"], [
        {"probability": str(risk), "affected": ["A"]},
        {"probability": str(1 - risk), "affected": []}],
        [{"probability": 1, "affected": []}])


class UncertaintyTests(unittest.TestCase):
    def test_cache_does_not_keep_finished_ensembles_alive(self):
        def temporary():
            model = RobustModel([simple(Fraction(1, 5))])
            model.availability.own("A")
            model.availability.pair("A", "B")
            model.evaluate(["A", "B"], 1)
            return weakref.ref(model.availability)
        reference = temporary()
        gc.collect()
        self.assertIsNone(reference())

    def test_one_member_equals_original(self):
        model = simple(Fraction(1, 5))
        robust = RobustModel([model])
        for nodes in (("A",), ("A", "B")):
            self.assertEqual(robust.evaluate(nodes, 1), model.evaluate(nodes, 1))

    def test_more_models_cannot_lower_object_risk_or_minimum(self):
        point, stressed = simple(Fraction(1, 10)), simple(Fraction(2, 5))
        robust = RobustModel([point, stressed])
        self.assertGreaterEqual(robust.evaluate(["A"], 1).availability_risk,
                                point.evaluate(["A"], 1).availability_risk)
        request = dict(k=1, availability_limit="0.2", loss_limit=0,
                       free_slots={"A": 1, "B": 0})
        self.assertIsNotNone(point.search(**request).best)
        self.assertEqual(robust.search(**request).status, "proven_infeasible_in_scope")
        self.assertEqual(graph_search(robust, **request).status, "not_found")

    def test_object_envelope_is_not_reconstructed_from_maximum_edges(self):
        a = ExactModel(["A", "B"], [{"probability": 1, "affected": ["A"]}],
                       [{"probability": 1, "affected": []}])
        b = ExactModel(["A", "B"], [{"probability": 1, "affected": ["B"]}],
                       [{"probability": 1, "affected": []}])
        robust = RobustModel([a, b])
        self.assertEqual(robust.availability.own("A"), 1)
        self.assertEqual(robust.availability.own("B"), 1)
        self.assertEqual(robust.evaluate(["A", "B"], 1).availability_risk, 0)

    def test_resampling_is_reproducible_and_does_not_use_future(self):
        store = ObservationStore()
        for t in range(20):
            store.add(Observation(t, t + 1, "availability", "day", (("A", False),)))
        a = bootstrap_weights(store, 10, seed=1, block_size=2)
        b = bootstrap_weights(store, 10, seed=1, block_size=2)
        self.assertEqual(a, b)
        self.assertTrue(all(t < 10 for t in a))
        self.assertEqual(sum(a.values()), 10)
        for t in range(0, 10, 2):
            self.assertEqual(a[t], a[t + 1])

    def test_weighted_fit_ignores_unselected_and_unreceived_rows(self):
        registry = (NodeSpec("A", "site", 0),)
        store = ObservationStore()
        store.add(Observation(0, 1, "availability", "day", (("A", True),)))
        store.add(Observation(1, 10, "loss", "day", (("A", True),)))
        model = fit(store, registry, 2, window_weights={0: 0, 1: 100})
        self.assertEqual(model.sample_counts["availability", "day"], 0)
        self.assertEqual(model.sample_counts["loss", "day"], 0)
        with self.assertRaises(ValueError):
            fit(store, registry, 2, window_weights={0: -1})

    def test_ensemble_keeps_point_model_and_new_nodes(self):
        registry = (NodeSpec("A", "site", 0), NodeSpec("B", "site", 2))
        store = ObservationStore()
        store.add(Observation(0, 1, "availability", "day", (("A", False),)))
        models, metadata = build_ensembles(store, registry, 2, strength=10, replicas=2)
        self.assertEqual(len(models["combined"].models), 5)
        self.assertEqual(models["combined"].nodes, ("A", "B"))
        self.assertIsNone(metadata["confidence_level"])
        self.assertGreater(models["combined"].loss.own("B"), 0)

    def test_invalid_ensemble_and_stress(self):
        with self.assertRaises(ValueError):
            RobustModel([])
        with self.assertRaises(ValueError):
            stress_prior((NodeSpec("A", "site", 0),), float("nan"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
