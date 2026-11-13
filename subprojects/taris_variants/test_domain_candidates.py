import copy
from itertools import combinations
import unittest

from .domain_candidates import plan_candidates


def candidate(node, dc, **overrides):
    return {"node": node, "dc": dc, "free_bytes": 100,
            "known_online": True, "eligible": True, **overrides}


class DomainCandidatesTests(unittest.TestCase):
    def solve(self, rows, **kwargs):
        return plan_candidates(rows, fragment_bytes=10, k=2, failed_domains=3, **kwargs)

    def test_concrete_indices_survive_every_failure_subset(self):
        rows = [candidate(d*10+j, d) for d in range(4) for j in range(2)]
        result = self.solve(rows)
        self.assertEqual(result["solver"]["minimum_n"], 8)
        plan = result["plan"]
        self.assertEqual(len({r["node"] for r in plan}), 8)
        self.assertEqual({r["index"] for r in plan}, set(range(8)))
        for failed in combinations(range(4), 3):
            self.assertGreaterEqual(len({r["index"] for r in plan if r["dc"] not in failed}), 2)
        self.assertEqual(result["acknowledged_fragments"], 0)
        self.assertFalse(result["real_world_guarantee"])

    def test_snapshot_order_does_not_change_plan_and_input_is_unchanged(self):
        rows = [candidate(d, d) for d in range(5)]
        before = copy.deepcopy(rows)
        self.assertEqual(self.solve(rows), self.solve(list(reversed(rows))))
        self.assertEqual(rows, before)

    def test_large_capacity_is_one_node_and_home_is_not_a_new_domain(self):
        rows = [candidate(d, d, free_bytes=100000) for d in range(4)]
        rows += [candidate(100+j, -1) for j in range(20)]
        result = self.solve(rows)
        self.assertFalse(result["solver"]["feasible"])
        self.assertEqual(result["solver"]["capacities"], [1]*4)
        self.assertEqual(result["plan"], [])

    def test_pending_and_unavailable_do_not_enter_capacity(self):
        rows = [candidate(d, d) for d in range(5)]
        rows += [candidate(10, 5, known_online=False), candidate(11, 5, eligible=False),
                 candidate(12, 5, free_bytes=9)]
        result = self.solve(rows, pending_nodes=[4], excluded_nodes=[3])
        self.assertEqual(result["domain_ids"], [0, 1, 2])
        self.assertFalse(result["solver"]["feasible"])
        self.assertEqual(len(result["excluded"]), 5)

    def test_search_limit_is_unknown_and_budget_failure_is_explicit(self):
        rows = [candidate(d, d) for d in range(5)]
        result = self.solve(rows, max_states=1)
        self.assertIsNone(result["solver"]["feasible"])
        self.assertEqual(result["plan"], [])
        self.assertFalse(self.solve(rows, max_n=4)["solver"]["feasible"])

    def test_rejects_duplicates_truth_fields_and_invalid_types(self):
        for rows in ([candidate(0, 0), candidate(0, 1)],
                     [candidate(0, 0, alive_to=100)], [candidate(True, 0)],
                     [candidate(0, 0, known_online=1)], [candidate(0, 0, free_bytes=float("nan"))]):
            with self.assertRaises(ValueError):
                self.solve(rows)
