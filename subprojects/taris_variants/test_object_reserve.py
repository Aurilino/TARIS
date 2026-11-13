import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import unittest
from subprojects.taris_variants.object_reserve import object_reserve


def copy(idx, start=0., end=None, node=0):
    return {"obj": 0, "idx": idx, "node": node, "start": start, "physical_end": end}


def record(copies, buffers=()):
    return {"initial_writes": [[0, 0., [], []]], "physical_fragments": copies, "buffers": list(buffers)}


class ObjectReserveTests(unittest.TestCase):
    def test_complete_reserve_and_volume_units(self):
        result = object_reserve(record([copy(i) for i in range(4)]), 2, 4, 10., object_size=4.)
        self.assertEqual(result["distinct_fragment_days"], 40.)
        self.assertEqual(result["physical_size_days"], 80.)
        self.assertEqual(result["below_target_object_days"], 0.)
        self.assertEqual(result["rank_days"], {4: 10.})

    def test_duplicate_is_storage_but_not_another_code_number(self):
        result = object_reserve(record([copy(0, node=0), copy(0, node=1), copy(1, node=2)]), 2, 3, 7.)
        self.assertEqual(result["physical_copy_days"], 21.)
        self.assertEqual(result["distinct_fragment_days"], 14.)
        self.assertEqual(result["duplicate_copy_days"], 7.)
        self.assertEqual(result["bare_code_object_days"], 7.)
        self.assertEqual(result["target_shortfall_fragment_days"], 7.)

    def test_gap_before_replacement_is_measured_exactly(self):
        result = object_reserve(record([copy(0, end=2.), copy(0, start=4.), copy(1), copy(2)]), 2, 3, 6.)
        self.assertEqual(result["below_target_object_days"], 2.)
        self.assertEqual(result["bare_code_object_days"], 2.)
        self.assertEqual(result["distinct_fragment_days"], 16.)

    def test_overlap_replacement_is_not_false_extra_reserve(self):
        result = object_reserve(record([copy(0), copy(0, start=3., node=3), copy(1), copy(2)]), 2, 3, 6.)
        self.assertEqual(result["below_target_object_days"], 0.)
        self.assertEqual(result["duplicate_copy_days"], 3.)

    def test_buffer_cover_is_separate_from_disk_reserve(self):
        result = object_reserve(record([copy(0), copy(1, end=3.)],
            [{"obj": 0, "start": 3., "end": 7.}]), 2, 3, 10.)
        self.assertEqual(result["disk_deficit_object_days"], 7.)
        self.assertEqual(result["buffer_only_object_days"], 4.)
        self.assertEqual(result["target_shortfall_fragment_days"], 17.)

    def test_future_copy_does_not_add_past_reserve(self):
        result = object_reserve(record([copy(0), copy(1), copy(2, start=11.)]), 2, 3, 10.)
        self.assertEqual(result["rank_days"], {2: 10.})

    def test_corrupted_trace_is_rejected(self):
        with self.assertRaises(ValueError):
            object_reserve(record([copy(0, start=4., end=2.)]), 2, 3, 10.)

    def test_touching_buffers_are_order_independent_but_overlap_is_rejected(self):
        buffers = [{"obj": 0, "start": 5., "end": 10.}, {"obj": 0, "start": 0., "end": 5.}]
        result = object_reserve(record([copy(0)], buffers), 2, 3, 10.)
        self.assertEqual(result["buffer_only_object_days"], 10.)
        buffers[1]["end"] = 6.
        with self.assertRaises(ValueError):
            object_reserve(record([copy(0)], buffers), 2, 3, 10.)


if __name__ == "__main__":
    unittest.main()
