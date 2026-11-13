import unittest
from subprojects.taris_variants.stress_actuarial import MODES, conditional_cost, break_even_loss_price


def scenarios(cost=2., loss=3.):
    return [{"mode": mode, "resource_cost": cost, "lost_admitted_volume": loss} for mode in MODES]


class StressActuarialTests(unittest.TestCase):
    def test_unknown_weights_and_unknown_price_do_not_become_zero(self):
        self.assertIsNone(conditional_cost(scenarios(), loss_price=4.)["conditional_total_cost"])
        weights = {m: .2 for m in MODES}
        result = conditional_cost(scenarios(), weights)
        self.assertAlmostEqual(result["expected_lost_volume"], 3.)
        self.assertIsNone(result["conditional_total_cost"])
        self.assertIsNone(result["full_actuarial_reserve"])

    def test_explicit_units_weights_and_permutation(self):
        weights = {m: float(m == "far") for m in MODES}
        result = conditional_cost(list(reversed(scenarios())), weights, 4.)
        self.assertEqual(result["conditional_total_cost"], 14.)
        self.assertFalse(result["real_world_scenario_probabilities_estimated"])

    def test_invalid_weights_price_and_duplicate_rows_rejected(self):
        with self.assertRaises(ValueError):
            conditional_cost(scenarios(), {m: .1 for m in MODES}, 1.)
        with self.assertRaises(ValueError):
            conditional_cost(scenarios(), {m: .2 for m in MODES}, -1.)
        with self.assertRaises(ValueError):
            conditional_cost(scenarios()+scenarios()[:1], {m: .2 for m in MODES}, 1.)

    def test_crossing_and_equal_loss_cases(self):
        weights = {m: .2 for m in MODES}
        result = break_even_loss_price(scenarios(10., 0.), scenarios(2., 4.), weights)
        self.assertAlmostEqual(result["loss_price"], 2.)
        self.assertEqual(break_even_loss_price(scenarios(), scenarios(), weights)["status"], "equal_for_all_prices")
        self.assertEqual(break_even_loss_price(scenarios(1.), scenarios(2.), weights)["status"], "no_finite_crossing")


if __name__ == "__main__":
    unittest.main()
