import unittest
from subprojects.taris_variants.wave_exposure import wave_exposure


def record():
    return {"initial_writes": [[0, 0., [0, 1, 2], [0, 1, 2]]], "losses": [], "buffers": [],
        "physical_fragments": [
            {"obj": 0, "idx": 0, "node": 0, "start": 0., "physical_end": 1.},
            {"obj": 0, "idx": 1, "node": 1, "start": 0., "physical_end": None},
            {"obj": 0, "idx": 1, "node": 2, "start": 0., "physical_end": None}]}


class WaveExposureTests(unittest.TestCase):
    def test_duplicate_survivors_do_not_form_code(self):
        result = wave_exposure(record(), [0], 1., 2)
        self.assertEqual(result["immediately_vulnerable_objects"], [0])
        self.assertEqual(result["rows"][0]["rank_before"], 2)
        self.assertEqual(result["rows"][0]["rank_after_removing_targets"], 1)

    def test_full_buffer_protects_even_after_disk_rank_drops(self):
        data = record()
        data["buffers"] = [{"obj": 0, "start": .5, "end": None}]
        self.assertEqual(wave_exposure(data, [0], 1., 2)["immediately_vulnerable_objects"], [])

    def test_prior_loss_and_not_yet_created_are_excluded(self):
        data = record()
        data["losses"] = [{"obj": 0, "t": .5}]
        self.assertEqual(wave_exposure(data, [0], 1., 2)["objects_at_risk"], 0)
        data["losses"] = []
        data["initial_writes"][0][1] = 2.
        self.assertEqual(wave_exposure(data, [0], 1., 2)["objects_at_risk"], 0)

    def test_copy_dead_before_wave_is_not_counted(self):
        data = record()
        data["physical_fragments"][0]["physical_end"] = .5
        result = wave_exposure(data, [1], 1., 2)
        self.assertEqual(result["rows"][0]["rank_before"], 1)


if __name__ == "__main__":
    unittest.main()
