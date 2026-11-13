import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import unittest
from copy import deepcopy
from subprojects.taris_variants.conditional_work import conditional_work
from subprojects.taris_variants import test_observable_memory as memory_tests


def example():
    return {"schema": "taris-queue-observation/2", "time": 4., "k": 2,
        "transport_settings": {"stripes": 4},
        "queue_settings": {"rate_size_per_hour": 2., "proof_fraction": .01},
        "object_sizes": [{"obj": 0, "size": 2.}],
        "jobs": [{"id": 0, "kind": "repair", "obj": 0, "indices": [0, 1], "nodes": [3, 4]}],
        "local_memory": [{"job": 0, "obj": 0, "kind": "full", "size": 2., "confirmed_at": 1.}],
        "block_acknowledgements": [
            {"job": 0, "phase": "write_block", "idx": 0, "node": 3, "stripe": 3, "ok": True, "time": 2.},
            {"job": 0, "phase": "write_block", "idx": 1, "node": 4, "stripe": 1, "ok": True, "time": 3.}]}


class ConditionalWorkTests(unittest.TestCase):
    def test_hand_calculation_two_modes_and_unknowns(self):
        snapshot = example()
        before = deepcopy(snapshot)
        trust = conditional_work(snapshot, transfer_price=3.)
        restart = conditional_work(snapshot, "restart_partial", transfer_price=3.)
        self.assertAlmostEqual(trust["known_repair_volume"], .52)
        self.assertAlmostEqual(trust["conditional_repair_transfer_cost"], 1.56)
        self.assertAlmostEqual(trust["uninterrupted_transfer_hours"], .26)
        self.assertAlmostEqual(restart["known_repair_volume"], 1.02)
        self.assertIsNone(trust["expected_future_cost"])
        self.assertIsNone(trust["completion_time"])
        self.assertEqual(snapshot, before)

    def test_latest_ack_replaces_old_large_prefix(self):
        snapshot = example()
        snapshot["block_acknowledgements"].append({"job": 0, "phase": "write_block", "idx": 1,
            "node": 4, "stripe": 0, "ok": True, "time": 3.5})
        self.assertAlmostEqual(conditional_work(snapshot)["known_repair_volume"], .77)

    def test_failed_ack_does_not_erase_confirmed_prefix(self):
        snapshot = example()
        snapshot["block_acknowledgements"].append({"job": 0, "phase": "write_block", "idx": 1,
            "node": 4, "stripe": 2, "ok": False, "time": 3.5})
        self.assertAlmostEqual(conditional_work(snapshot)["known_repair_volume"], .52)

    def test_current_phase_not_added_twice(self):
        snapshot = example()
        snapshot["planned_channel_phase"] = {"volume": 100., "start": 3., "end": 20.}
        self.assertAlmostEqual(conditional_work(snapshot)["known_repair_volume"], .52)

    def test_unknown_proof_prevents_whole_queue_total(self):
        snapshot = example()
        snapshot["jobs"].append({"id": 1, "kind": "proof", "node": 5})
        result = conditional_work(snapshot)
        self.assertEqual(result["unknown_proof_jobs"], [1])
        self.assertIsNone(result["whole_queue_volume"])

    def test_runtime_prefix_and_memory_fault_change_read_work(self):
        ledger = memory_tests.ObservableMemoryTests().prefix_ledger()
        snapshot = ledger.queue_observation(.09)
        result = conditional_work(snapshot)
        held = sum(r["size"] for r in snapshot["local_memory"])
        self.assertAlmostEqual(result["jobs"][0]["read_volume"], 1.-held)
        ledger._push(.09001, "queue_buffer_fail", 0)
        ledger._process(.09002)
        after = conditional_work(ledger.queue_observation(.09002))
        self.assertAlmostEqual(after["jobs"][0]["read_volume"], 1.)

    def test_reject_future_ack_bad_price_and_duplicate_target(self):
        snapshot = example()
        snapshot["block_acknowledgements"][0]["time"] = 4.
        with self.assertRaises(ValueError):
            conditional_work(snapshot)
        with self.assertRaises(ValueError):
            conditional_work(example(), transfer_price=-1.)
        snapshot = example()
        snapshot["jobs"][0]["indices"] = [0, 0]
        with self.assertRaises(ValueError):
            conditional_work(snapshot)


if __name__ == "__main__":
    unittest.main()
