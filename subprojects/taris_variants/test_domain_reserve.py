import unittest
from itertools import combinations, product
from subprojects.taris_variants.domain_reserve import minimum_layout, unconstrained_minimum, worst_surviving


def node_subset_minimum(caps, k, b):
    nodes = [domain for domain, count in enumerate(caps) for _ in range(count)]
    domains = sorted(set(nodes))
    failures = list(combinations(domains, min(b, len(domains))))
    for n in range(k, len(nodes)+1):
        for chosen in combinations(range(len(nodes)), n):
            if all(sum(nodes[node] not in failed for node in chosen) >= k for failed in failures):
                return n
    return None


class DomainReserveTests(unittest.TestCase):
    def test_number_of_domains_changes_minimum(self):
        self.assertEqual(minimum_layout([1]*4, 2, 2)["minimum_n"], 4)
        self.assertEqual(minimum_layout([1]*5, 2, 3)["minimum_n"], 5)
        self.assertEqual(minimum_layout([2]*4, 2, 3)["minimum_n"], 8)

    def test_limited_capacity_can_raise_bound_or_make_task_impossible(self):
        self.assertEqual(unconstrained_minimum(5, 2, 5), 9)
        self.assertEqual(minimum_layout([1, 1, 3, 3, 3], 5, 2)["minimum_n"], 11)
        self.assertEqual(minimum_layout([2, 2, 2, 1], 2, 3)["status"], "infeasible_capacity")
        self.assertEqual(minimum_layout([2]*4, 2, 3, max_n=7)["status"], "infeasible_budget")

    def test_exhaustive_independent_node_subsets(self):
        for d in (1, 2, 3):
            for caps in product(range(3), repeat=d):
                for k in (1, 2):
                    for b in (0, 1, 2):
                        expected = node_subset_minimum(caps, k, b)
                        result = minimum_layout(caps, k, b)
                        self.assertEqual(result["minimum_n"], expected, (caps, k, b))
                        self.assertEqual(result["feasible"], expected is not None)
                        if expected is not None:
                            self.assertEqual(sum(result["layout"]), expected)
                            self.assertTrue(all(x <= c for x, c in zip(result["layout"], caps)))

    def test_permuting_labels_preserves_minimum(self):
        a = minimum_layout([1, 1, 3, 3, 3], 5, 2)
        b = minimum_layout([3, 1, 3, 1, 3], 5, 2)
        self.assertEqual(a["minimum_n"], b["minimum_n"])

    def test_search_limit_is_unknown_not_infeasible(self):
        result = minimum_layout([3]*4, 2, 2, max_states=1)
        self.assertEqual(result["status"], "search_limit")
        self.assertIsNone(result["feasible"])
        self.assertFalse(result["certified_infeasible"])
        self.assertFalse(result["certified_minimum"])

    def test_input_scope_and_zero_capacity_domains(self):
        self.assertEqual(minimum_layout([0, 1, 1, 1, 1], 2, 2)["minimum_n"], 4)
        self.assertEqual(minimum_layout([5, 5], 2, 2)["status"], "infeasible_domains")
        for values in (([True], 1, 0), ([-1], 1, 0), ([1], 0, 0), ([1], 1, -1)):
            with self.assertRaises(ValueError):
                minimum_layout(*values)
        self.assertEqual(worst_surviving([2, 2, 1], 1), 3)


if __name__ == "__main__":
    unittest.main()
