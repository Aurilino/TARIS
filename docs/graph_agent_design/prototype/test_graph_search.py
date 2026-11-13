"""Р2: проверки безопасности результата и явных ограничений эвристики."""
from fractions import Fraction
import json
from pathlib import Path
import unittest

from compare_search import generated_cases, trap_case
from exact_model import ExactModel, placement_key
from graph_search import graph_search


def model(case):
    return ExactModel(case["nodes"], case["availability"], case["loss"])


class GraphSearchTests(unittest.TestCase):
    def test_local_minimum_is_real_and_restarts_escape(self):
        case = trap_case()
        network = model(case)
        exact = network.search(**case["request"])
        single = graph_search(network, **case["request"], seed_count=1)
        multi = graph_search(network, **case["request"], seed_count=3)
        self.assertEqual(single.best.nodes, ("A", "B"))
        self.assertEqual(single.best.graph_score, Fraction(1, 100))
        self.assertEqual(exact.best.nodes, ("C", "D"))
        self.assertEqual(multi.best, exact.best)

    def test_search_can_use_extra_chunk(self):
        case = trap_case(availability_limit="0", max_n=3)
        network = model(case)
        self.assertEqual(len(network.search(**case["request"]).best.nodes), 2)
        self.assertEqual(len(graph_search(network, **case["request"], seed_count=1).best.nodes), 3)

    def test_not_found_is_not_impossibility(self):
        case = trap_case(availability_limit="0", max_n=2)
        network = model(case)
        result = graph_search(network, **case["request"], seed_count=1)
        self.assertEqual(result.status, "not_found")
        self.assertIsNone(result.best)
        self.assertIsNotNone(network.search(**case["request"]).best)

    def test_returned_solutions_are_feasible_and_not_better_than_exact(self):
        root = Path(__file__).resolve().parent
        cases = json.loads((root / "examples.json").read_text(encoding="utf-8"))["cases"] + generated_cases(6)
        for case in cases:
            with self.subTest(case=case["name"]):
                network = model(case)
                exact = network.search(**case["request"])
                result = graph_search(network, **case["request"])
                if result.best:
                    self.assertIsNotNone(exact.best)
                    self.assertEqual(network.evaluate(result.best.nodes, case["request"]["k"]), result.best)
                    self.assertLessEqual(result.best.availability_risk, Fraction(str(case["request"]["availability_limit"])))
                    self.assertLessEqual(result.best.loss_risk, Fraction(str(case["request"]["loss_limit"])))
                    self.assertGreaterEqual(len(result.best.nodes), len(exact.best.nodes))
                    if len(result.best.nodes) == len(exact.best.nodes):
                        self.assertGreaterEqual(placement_key(result.best), placement_key(exact.best))

    def test_capacity_and_invalid_options(self):
        network = model(trap_case())
        request = dict(k=1, availability_limit=1, loss_limit=1)
        result = graph_search(network, **request, free_slots={n: 0 for n in network.nodes})
        self.assertEqual(result.status, "insufficient_capacity")
        self.assertEqual(result.graph_pairs_prepared, 0)
        for options in ({"seed_count": 0}, {"seed_count": True}, {"max_swaps": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                graph_search(network, **request, **options)

    def test_candidate_order_does_not_change_search(self):
        case = trap_case()
        a = model(case)
        b = ExactModel(reversed(case["nodes"]), case["availability"], case["loss"])
        self.assertEqual(graph_search(a, **case["request"]), graph_search(b, **case["request"]))

    def test_single_node_and_no_local_swaps(self):
        network = ExactModel(["A"], [{"probability": 1, "affected": []}],
                             [{"probability": 1, "affected": []}])
        result = graph_search(network, k=1, availability_limit=0, loss_limit=0, max_swaps=0)
        self.assertEqual(result.best.nodes, ("A",))
        self.assertEqual(result.checked, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
