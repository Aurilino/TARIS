import unittest

from run_pipeline import run


class PipelineTests(unittest.TestCase):
    def test_learned_placement_drives_real_execution_and_reservations(self):
        result = run(12)
        for method, data in result["results"].items():
            with self.subTest(method=method):
                self.assertIsNotNone(data["placement"]["best"])
                self.assertEqual(len(data["episodes"]), 8)
                for episode in data["episodes"].values():
                    self.assertTrue(all(value <= 10 for value in episode["maximum_allocated_and_reserved"].values()))
                    self.assertTrue(all(job["status"] in ("done", "failed", "queued", "reading", "writing") for job in episode["jobs"]))
                    for job in episode["jobs"]:
                        if job["status"] == "done":
                            self.assertTrue(any(event["event"] == "write_confirmed" and event["job"] == job["id"]
                                                for event in episode["log"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
