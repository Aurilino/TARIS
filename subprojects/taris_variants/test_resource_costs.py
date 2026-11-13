import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import unittest
from subprojects.taris_variants.resource_costs import resource_usage, observed_charges, ResourcePrices


def trace():
    return {"physical_fragments": [{"start": 0., "physical_end": None} for _ in range(2)],
        "buffers": [{"obj": 0, "start": 2., "end": 4., "size": 4.}],
        "decoded_prefixes": [{"obj": 0, "start": 1., "end": 2., "size": 2.}],
        "receiver_prefixes": [{"obj": 0, "start": 3., "end": 4., "size": 1.}],
        "segments": [{"phase": "read", "start": 0., "end": 8., "volume": 8.}]}


class ResourceCostsTests(unittest.TestCase):
    def test_explicit_price_units_and_separate_memory_components(self):
        usage = resource_usage(trace(), 10., 2, 4.)
        result = observed_charges(usage, ResourcePrices(3., .2, 10., 2.), 2, 4.)
        self.assertEqual(result["size_day_price"], 1.5)
        self.assertEqual(result["disk"], 60.)
        self.assertEqual(result["full_buffer"], 120.)
        self.assertEqual(result["decoded_prefix"], 30.)
        self.assertEqual(result["receiver_prefix"], 3.)
        self.assertAlmostEqual(result["observed_resource_cost"], 214.6)
        self.assertIsNone(result["expected_loss_cost"])
        self.assertIsNone(result["full_actuarial_reserve"])

    def test_volume_price_conversion_changes_with_code_and_size(self):
        usage = resource_usage(trace(), 10., 2, 4.)
        prices = ResourcePrices(fragment_day=3.)
        self.assertEqual(observed_charges(usage, prices, 4, 4.)["size_day_price"], 3.)
        self.assertEqual(observed_charges(usage, prices, 2, 8.)["size_day_price"], .75)

    def test_price_sensitivity_is_monotone(self):
        usage = resource_usage(trace(), 10., 2, 4.)
        a = observed_charges(usage, ResourcePrices(), 2, 4.)["observed_resource_cost"]
        b = observed_charges(usage, ResourcePrices(central_ratio=10.), 2, 4.)["observed_resource_cost"]
        c = observed_charges(usage, ResourcePrices(central_ratio=10., receiver_ratio=10.), 2, 4.)["observed_resource_cost"]
        self.assertLess(a, b)
        self.assertLess(b, c)

    def test_failed_attempts_are_charged_and_future_service_is_not(self):
        row = trace()
        row["segments"] = [{"phase": "failed_read", "start": 1., "end": 3., "volume": 2.},
                           {"phase": "write", "start": 3., "end": 7., "volume": 4.}]
        usage = resource_usage(row, 5., 2, 4.)
        self.assertEqual(usage["served_volume_by_phase"], {"failed_read": 2., "write": 2.})
        self.assertEqual(usage["scheduled_unserved_volume"], 2.)

    def test_prefix_overlapping_full_buffer_is_rejected(self):
        row = trace()
        row["decoded_prefixes"][0]["end"] = 3.
        with self.assertRaises(ValueError):
            resource_usage(row, 10., 2, 4.)

    def test_negative_price_and_overlapping_channel_are_rejected(self):
        with self.assertRaises(ValueError):
            ResourcePrices(central_ratio=-1.).validate()
        row = trace()
        row["segments"].append({"phase": "write", "start": 4., "end": 5., "volume": 1.})
        with self.assertRaises(ValueError):
            resource_usage(row, 10., 2, 4.)


if __name__ == "__main__":
    unittest.main()
