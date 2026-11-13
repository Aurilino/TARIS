import unittest
from subprojects.taris_variants.observation_labels import label_snapshots


class LabelTests(unittest.TestCase):
    def test_clipping_and_censoring_keep_positive_work(self):
        snapshots = [{"time": 2., "jobs": [{"id": 0, "kind": "repair"}]}]
        jobs = [{"id": 0, "submitted": 0., "finished": None, "status": "write_block"}]
        segments = [{"job": 0, "start": 1., "end": 5., "volume": 8.}]
        row = label_snapshots(snapshots, jobs, segments, [], 4.)[0]
        self.assertEqual(row["observed_future_volume"], 4.)
        self.assertIsNone(row["full_future_volume"])
        self.assertIsNone(row["time_to_completion_days"])
        self.assertTrue(row["censored"])

    def test_completion_exactly_at_boundary_is_future(self):
        snapshots = [{"time": 2., "jobs": [{"id": 0, "kind": "repair"}]}]
        jobs = [{"id": 0, "submitted": 0., "finished": 2., "status": "done"}]
        row = label_snapshots(snapshots, jobs, [], [], 4.)[0]
        self.assertEqual(row["time_to_completion_days"], 0.)
        self.assertFalse(row["censored"])

    def test_exact_horizon_is_censored_and_not_zero(self):
        snapshots = [{"time": 1., "jobs": [{"id": 0, "kind": "repair"}]}]
        jobs = [{"id": 0, "submitted": 0., "finished": 4., "status": "done"}]
        segments = [{"job": 0, "start": 1., "end": 4., "volume": 3.}]
        row = label_snapshots(snapshots, jobs, segments, [], 4.)[0]
        self.assertTrue(row["censored"])
        self.assertEqual(row["observed_future_volume"], 3.)

    def test_does_not_mix_jobs_or_include_past_failed_blocks(self):
        snapshots = [{"time": 2., "jobs": [{"id": 0, "kind": "repair"}]}]
        jobs = [{"id": 0, "submitted": 0., "finished": 3., "status": "done"}]
        segments = [{"job": 0, "start": 0., "end": 1., "volume": 1.},
                    {"job": 0, "start": 2., "end": 2.5, "volume": 2.},
                    {"job": 1, "start": 2., "end": 3., "volume": 100.}]
        results = [{"segment": i, "ok": False} for i in range(3)]
        row = label_snapshots(snapshots, jobs, segments, results, 4.)[0]
        self.assertEqual(row["full_future_volume"], 2.)
        self.assertEqual(row["future_failed_blocks_observed"], 1)


if __name__ == "__main__":
    unittest.main()
