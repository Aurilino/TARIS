import unittest
from subprojects.taris_variants.restricted_forecast import observed_window_volume, predict, fit_profile


class RestrictedTests(unittest.TestCase):
    def test_window_counts_partial_failed_work_but_not_other_jobs(self):
        segments = [{"job": 0, "start": 0., "end": 4., "volume": 8.},
                    {"job": 1, "start": 1., "end": 3., "volume": 100.},
                    {"job": 0, "start": 4., "end": 6., "volume": 2.}]
        self.assertEqual(observed_window_volume(segments, {0}, 2., 3., 8.), 5.)
        with self.assertRaises(ValueError):
            observed_window_volume(segments, {0}, 6., 3., 8.)

    def test_known_capacity_and_nonnegative_coefficient(self):
        self.assertEqual(predict({"known_work": 4., "capacity": 3.}, 2.), 3.)
        self.assertEqual(predict({"known_work": 0., "capacity": 3.}, 2.), 0.)
        with self.assertRaises(ValueError):
            predict({"known_work": 4., "capacity": 3.}, -1.)

    def test_fit_equal_world_weight_and_grid(self):
        # Один мир с сотней копий не получает вес в сто раз больше другого.
        row = {"seed": 1, "known_work": 1., "capacity": 10., "actual_window_volume": 1.}
        rows = [dict(row) for _ in range(100)]
        rows.append({**row, "seed": 2, "actual_window_volume": 3.})
        self.assertEqual(fit_profile(rows)["coefficient"], 2.)

    def test_no_positive_queue_is_untrained_and_boundary_reported(self):
        row = {"seed": 1, "known_work": 0., "capacity": 10., "actual_window_volume": 0.}
        self.assertEqual(fit_profile([row])["status"], "untrained")
        row.update(known_work=1., actual_window_volume=8.)
        self.assertTrue(fit_profile([row])["edge_of_grid"])


if __name__ == "__main__":
    unittest.main()
