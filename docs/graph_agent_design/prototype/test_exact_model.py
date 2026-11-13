"""Проверки математических свойств и известных контрпримеров."""
from fractions import Fraction
from itertools import combinations
import json
from pathlib import Path
import unittest

from exact_model import Distribution, ExactModel


class ExactModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {c["name"]: c for c in json.loads(
            Path(__file__).with_name("examples.json").read_text(encoding="utf-8"))["cases"]}

    def model(self, name):
        c = self.cases[name]
        return ExactModel(c["nodes"], c["availability"], c["loss"])

    def search(self, name):
        return self.model(name).search(**self.cases[name]["request"])

    def test_regular_schedules(self):
        sync = self.model("synchronous_schedule")
        spread = self.model("staggered_schedule")
        self.assertEqual(sync.evaluate(sync.nodes, 2).availability_risk, Fraction(1, 3))
        self.assertEqual(spread.evaluate(spread.nodes, 2).availability_risk, 0)
        self.assertEqual(self.search("synchronous_schedule").status, "proven_infeasible_in_scope")
        result = self.search("staggered_schedule")
        self.assertEqual(len(result.best.nodes), 3)
        self.assertEqual(result.checked, 4)

    def test_same_pair_graph_different_object_risk(self):
        x, y = self.model("same_graph_X"), self.model("same_graph_Y")
        for node in x.nodes:
            self.assertEqual(x.availability.own(node), y.availability.own(node))
        for i, j in combinations(x.nodes, 2):
            self.assertEqual(x.availability.pair(i, j), y.availability.pair(i, j))
        self.assertEqual(x.evaluate(x.nodes, 2).availability_risk, Fraction(3, 4))
        self.assertEqual(y.evaluate(y.nodes, 2).availability_risk, Fraction(1, 4))
        self.assertEqual(self.search("same_graph_X").status, "proven_infeasible_in_scope")
        self.assertEqual(self.search("same_graph_Y").status, "optimal_in_scope")

    def test_loss_constraint_selects_independent_site(self):
        model = self.model("shared_site_and_independent_node")
        self.assertEqual(model.evaluate(["S", "A"], 1).loss_risk, Fraction(1, 50))
        self.assertEqual(model.evaluate(["S", "B"], 1).loss_risk, Fraction(1, 1250))
        result = self.search("shared_site_and_independent_node")
        self.assertEqual(result.best.nodes, ("A", "B"))
        self.assertEqual(result.feasible_at_min_n, 2)
        self.assertEqual(result.checked, 6)

    def test_one_chunk_uses_own_risk_when_graph_is_empty(self):
        model = ExactModel(["A", "B"], [
            {"probability": "0.1", "affected": ["A"]},
            {"probability": "0.9", "affected": []}],
            [{"probability": 1, "affected": []}])
        result = model.search(k=1, availability_limit=1, loss_limit=0)
        self.assertEqual(result.best.nodes, ("B",))

    def test_capacity_and_scope_are_distinct(self):
        model = self.model("staggered_schedule")
        no_space = model.search(k=2, availability_limit=0, loss_limit=0,
                                free_slots={"A": 0, "B": 1, "C": 0})
        self.assertEqual(no_space.status, "insufficient_capacity")
        scoped = model.search(k=2, max_n=2, availability_limit=0, loss_limit=0)
        self.assertEqual(scoped.status, "proven_infeasible_in_scope")
        self.assertEqual(scoped.checked, 3)

    def test_boundary_is_exact(self):
        model = self.model("synchronous_schedule")
        allowed = model.search(k=2, availability_limit="1/3", loss_limit=0)
        rejected = model.search(k=2, availability_limit="0.3333333333333333", loss_limit=0)
        self.assertEqual(allowed.status, "optimal_in_scope")
        self.assertEqual(rejected.status, "proven_infeasible_in_scope")

    def test_invalid_distributions(self):
        for states in (
            [{"probability": "0.9", "affected": []}],
            [{"probability": "1", "affected": ["unknown"]}],
            [{"probability": "1", "affected": ["A", "A"]}],
            [{"probability": "-1", "affected": []}],
            [{"probability": True, "affected": []}],
        ):
            with self.subTest(states=states), self.assertRaises(ValueError):
                Distribution(["A"], states)

    def test_invalid_placement_and_request(self):
        model = self.model("staggered_schedule")
        with self.assertRaises(ValueError):
            model.evaluate(["A", "A"], 1)
        with self.assertRaises(ValueError):
            model.search(k=2, min_n=1, availability_limit=0, loss_limit=0)
        with self.assertRaises(ValueError):
            model.search(k=2, availability_limit=0, loss_limit=0, free_slots={"A": 1})

    def test_node_order_does_not_change_decision(self):
        case = self.cases["shared_site_and_independent_node"]
        reversed_model = ExactModel(reversed(case["nodes"]), case["availability"], case["loss"])
        self.assertEqual(reversed_model.search(**case["request"]), self.search(case["name"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
