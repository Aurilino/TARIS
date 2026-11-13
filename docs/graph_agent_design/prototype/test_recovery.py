from fractions import Fraction
import unittest

from recovery import ObjectSpec, RecoveryEngine, RecoveryPolicy, StorageNode
from run_recovery import base_engine, run_cases, shared_engine


class RecoveryTests(unittest.TestCase):
    def test_write_confirmation_and_return_do_not_duplicate_unique_chunks(self):
        engine = base_engine()
        engine.run("1.5", events=[(0, {"kind": "down", "node": "C"})], policy=RecoveryPolicy())
        self.assertEqual(len(engine.physical), 3)
        self.assertEqual(engine.jobs[0].status, "writing")
        engine.run(5, events=[(4, {"kind": "up", "node": "C"})], policy=RecoveryPolicy())
        self.assertEqual(len(engine.physical), 4)
        self.assertEqual(len({idx for _, idx, _ in engine.physical}), 3)
        self.assertEqual(len(engine.jobs), 1)

    def test_bandwidth_is_shared_and_queue_changes_completion(self):
        slow = shared_engine(1).run(6, policy=RecoveryPolicy())
        fast = shared_engine(2).run(6, policy=RecoveryPolicy())
        self.assertEqual([j.finished for j in slow.jobs], [4, 4])
        self.assertEqual([j.finished for j in fast.jobs], [3, 3])
        self.assertEqual(slow.traffic, 6)
        self.assertEqual(fast.traffic, 6)
        self.assertTrue(all(value <= 1 for node, value in slow.maximum_reserved_used.items() if node in "CD"))

    def test_reservation_blocks_conflicting_requests(self):
        engine = shared_engine(1)
        self.assertTrue(engine.request("O1", 2, "C", ((0, "A"), (1, "B"))))
        self.assertFalse(engine.request("O2", 2, "C", ((0, "A"), (1, "B"))))
        self.assertFalse(engine.request("O1", 2, "D", ((0, "A"), (1, "B"))))
        self.assertEqual(engine.free_slots("C"), 0)

    def test_temporary_down_does_not_destroy_information(self):
        engine = base_engine().run(3, events=[(0, {"kind": "down", "node": "A"}),
                                             (0, {"kind": "down", "node": "B"})])
        self.assertFalse(engine.lost)
        self.assertEqual(len(engine.physical), 3)

    def test_silent_loss_not_visible_until_audit(self):
        engine = base_engine()
        before = engine.snapshot().catalog
        engine.apply("silent_loss", "C", "O", 2)
        self.assertEqual(engine.snapshot().catalog, before)
        engine.run("0.5", policy=RecoveryPolicy())
        self.assertEqual(engine.jobs, [])
        engine.apply("audit", "C")
        engine.run(4, policy=RecoveryPolicy())
        self.assertEqual(engine.jobs[0].created, Fraction(1, 2))
        self.assertEqual(engine.jobs[0].status, "done")

    def test_source_corruption_fails_without_creating_chunk(self):
        engine = base_engine().run(1, events=[(0, {"kind": "down", "node": "C"}),
            ("0.5", {"kind": "silent_loss", "node": "A", "obj": "O", "index": 0})], policy=RecoveryPolicy())
        self.assertEqual(engine.jobs[0].status, "failed")
        self.assertEqual(engine.free_slots("D"), engine.nodes["D"].capacity)
        self.assertFalse(any(node == "D" for _, _, node in engine.physical))

    def test_target_down_pauses_write(self):
        engine = base_engine().run(5, events=[(0, {"kind": "down", "node": "C"}),
            ("1.5", {"kind": "down", "node": "D"}), (3, {"kind": "up", "node": "D"})], policy=RecoveryPolicy())
        self.assertEqual(engine.jobs[0].finished, Fraction(7, 2))

    def test_buffer_is_explicit_and_can_fail(self):
        cases = run_cases()
        self.assertFalse(cases["buffer_preserves_reconstructed_object"]["lost"])
        self.assertEqual(cases["buffer_failure_after_source_loss"]["lost"], {"O": Fraction(3, 2)})

    def test_policy_receives_no_physical_state(self):
        snapshot = base_engine().snapshot()
        self.assertFalse(hasattr(snapshot, "physical"))
        self.assertFalse(hasattr(snapshot, "lost"))
        self.assertFalse(hasattr(snapshot, "engine"))

    def test_invalid_config_and_events(self):
        with self.assertRaises(ValueError):
            RecoveryEngine([StorageNode("A", "s"), StorageNode("A", "s")], [ObjectSpec("O", 1, 1)])
        with self.assertRaises(ValueError):
            base_engine().run(2, events=[(3, {"kind": "down", "node": "A"})])


if __name__ == "__main__":
    unittest.main(verbosity=2)
