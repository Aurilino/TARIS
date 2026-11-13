"""Р3: граница знания, неполные наблюдения и согласованность рисков."""
from fractions import Fraction
from itertools import combinations
import unittest

from learning import NodeSpec, Observation, ObservationStore, fit, select_strength
from run_learning import make_experiment


class LearningTests(unittest.TestCase):
    def test_future_observations_do_not_change_past_fit(self):
        registry = (NodeSpec("A", "site", 0), NodeSpec("B", "site", 0))
        past = ObservationStore()
        full = ObservationStore()
        row = Observation(0, 1, "availability", "day", (("A", True), ("B", False)))
        past.add(row)
        full.add(row)
        full.add(Observation(5, 6, "availability", "day", (("A", False), ("B", True))))
        a, b = fit(past, registry, 2), fit(full, registry, 2)
        self.assertEqual(a.probabilities, b.probabilities)
        self.assertEqual(a.sample_counts, b.sample_counts)

    def test_delayed_audit_is_not_known_at_window_end(self):
        registry = (NodeSpec("A", "site", 0),)
        store = ObservationStore()
        store.add(Observation(0, 5, "loss", "day", (("A", True),)))
        early = fit(store, registry, 2)
        empty = fit(ObservationStore(), registry, 2)
        late = fit(store, registry, 5)
        self.assertEqual(early.probabilities, empty.probabilities)
        self.assertGreater(float(late.exact_model().loss.own("A")), float(early.exact_model().loss.own("A")))

    def test_unknown_values_are_not_successes(self):
        registry = (NodeSpec("A", "site", 0),)
        store = ObservationStore()
        for t in range(10):
            store.add(Observation(t, t + 1, "availability", "day", (("A", None),)))
        a, b = fit(store, registry, 10), fit(ObservationStore(), registry, 10)
        self.assertEqual(a.probabilities, b.probabilities)
        self.assertEqual(a.sample_counts["availability", "day"], 0)

    def test_new_node_has_no_labels_and_positive_risk(self):
        registry, store, _, _ = make_experiment(0)
        learned = fit(store, registry, 160)
        self.assertEqual(sum(count for (*_, node), count in learned.label_counts.items() if node == "F"), 0)
        self.assertGreater(learned.exact_model().availability.own("F"), 0)
        self.assertGreater(learned.exact_model().loss.own("F"), 0)
        self.assertNotIn("F", fit(store, registry, 159).nodes)

    def test_pre_registration_labels_are_ignored(self):
        registry = (NodeSpec("A", "site", 5),)
        store = ObservationStore()
        store.add(Observation(0, 1, "loss", "day", (("A", True),)))
        a, b = fit(store, registry, 5), fit(ObservationStore(), registry, 5)
        self.assertEqual(a.probabilities, b.probabilities)
        self.assertEqual(a.label_counts["loss", "day", "A"], 0)

    def test_layer_and_context_are_separate(self):
        registry = (NodeSpec("A", "site", 0),)
        store = ObservationStore()
        for t in range(40):
            context = "day" if t % 2 == 0 else "night"
            store.add(Observation(t, t + 1, "availability", context, (("A", context == "night"),)))
        learned = fit(store, registry, 40, strength=2)
        day = learned.exact_model({"day": 1, "night": 0})
        night = learned.exact_model({"day": 0, "night": 1})
        self.assertGreater(night.availability.own("A"), day.availability.own("A"))
        self.assertEqual(day.loss.own("A"), night.loss.own("A"))

    def test_joint_model_learns_shared_events(self):
        registry = (NodeSpec("A", "site", 0), NodeSpec("B", "site", 0))
        store = ObservationStore()
        for t in range(100):
            down = t % 5 == 0
            store.add(Observation(t, t + 1, "availability", "day", (("A", down), ("B", down))))
        joint = fit(store, registry, 100, strength=2).exact_model({"day": 1, "night": 0})
        independent = fit(store, registry, 100, method="independent", strength=2).exact_model({"day": 1, "night": 0})
        self.assertGreater(joint.availability.pair("A", "B"), independent.availability.pair("A", "B"))

    def test_distribution_and_graph_remain_coherent(self):
        registry, store, _, _ = make_experiment(1)
        learned = fit(store, registry, 160).exact_model()
        for layer in (learned.availability, learned.loss):
            self.assertEqual(sum((s.probability for s in layer.states), Fraction()), 1)
            for i, j in combinations(learned.nodes, 2):
                pair, left, right = layer.pair(i, j), layer.own(i), layer.own(j)
                self.assertLessEqual(pair, min(left, right))
                self.assertGreaterEqual(pair, max(0, left + right - 1))
                self.assertEqual(pair, layer.risk((i, j), 1))

    def test_unsmoothed_control_uses_observed_frequencies(self):
        registry = (NodeSpec("A", "site", 0),)
        store = ObservationStore()
        for t in range(10):
            store.add(Observation(t, t + 1, "availability", "day", (("A", t == 0),)))
        fitted = fit(store, registry, 10, method="unsmoothed")
        p = float(fitted.exact_model({"day": 1, "night": 0}).availability.own("A"))
        self.assertAlmostEqual(p, 0.1)

    def test_strength_selection_ignores_future_test(self):
        registry, full, _, _ = make_experiment(2)
        historical = ObservationStore()
        for row in full.known(160):
            historical.add(row)
        a = select_strength(full, registry, train_cutoff=100, validation_cutoff=160)
        b = select_strength(historical, registry, train_cutoff=100, validation_cutoff=160)
        self.assertEqual(a, b)

    def test_invalid_observations_and_duplicate_exposure(self):
        store = ObservationStore()
        good = Observation(0, 1, "availability", "day", (("A", False),))
        store.add(good)
        for row in (good,
                    Observation(1, 1, "availability", "day", (("A", False),)),
                    Observation(2, 3, "availability", "day", (("A", 0),)),
                    Observation(2, 3, "loss", "day", (("A", None), ("A", True)))):
            with self.subTest(row=row), self.assertRaises(ValueError):
                store.add(row)

    def test_invalid_model_options(self):
        registry = (NodeSpec("A", "site", 0),)
        store = ObservationStore()
        for options in ({"strength": 0}, {"strength": float("nan")}, {"method": "unknown"}, {"iterations": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fit(store, registry, 1, **options)
        learned = fit(store, registry, 1)
        with self.assertRaises(ValueError):
            learned.exact_model({"day": "0.3", "night": "0.3"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
