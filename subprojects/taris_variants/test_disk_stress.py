import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from subprojects.taris_variants.disk_stress import DiskStressLedger, DiskWave, public_wave_targets
from subprojects.taris_variants.observable_memory import ObservableMemoryLedger
from subprojects.taris_variants.test_physical import fixture
from subprojects.taris_variants.audit_disk_stress import audit_stress_record


class DiskStressTests(unittest.TestCase):
    def test_audit_rejects_copy_surviving_scheduled_disk_death(self):
        ledger = fixture()
        record = {"stress_schedule": [{"time": 1., "nodes": [0]}],
            "disk_notifications": [{"time": 1., "node": 0, "kind": "confirmed_disk_failure", "declared_copies": [0]}],
            "physical_fragments": [{"obj": 0, "idx": 0, "node": 0, "start": 0.,
                                    "physical_end": None, "allocated": True}]}
        with self.assertRaises(AssertionError):
            audit_stress_record(record, ledger.cfg, ledger.w, ledger.ch, ledger.data)

    def test_control_full_run_identical_to_original(self):
        histories = []
        for kind in (ObservableMemoryLedger, DiskStressLedger):
            ledger = fixture(kind)
            ledger.data.new_objects = lambda epoch: np.array([0]) if epoch == 0 else np.array([], int)
            ledger.ch.reset = lambda: None
            ledger.ch.step = lambda: {"seed": "00"*32}
            policy = SimpleNamespace(n=3, name="control", alarm=lambda t, v: [], on_epoch=lambda t, v: [],
                migrate=lambda t, v: [], release=lambda t, v: [], place=lambda o, t, v: [0, 1, 2])
            result = ledger.run(policy)
            histories.append((result.metrics(), ledger.jobs, ledger.segments, ledger.initial_writes))
        self.assertEqual(*histories)

    def test_confirmed_notice_destroys_and_frees_slots(self):
        ledger = fixture(lambda w, c, d: DiskStressLedger(w, c, d, waves=[DiskWave(1., (0,))]))
        ledger._world_events()
        ledger._process(1.01)
        self.assertFalse(ledger.f_exist[0])
        self.assertEqual(ledger.f_dead[0], 1.)
        self.assertEqual(ledger.f_end[0], "сбой диска (подтверждён)")
        self.assertEqual(ledger.used[0], 0)
        self.assertIn(0, ledger.missing[0])
        self.assertEqual(ledger.disk_notifications[0]["time"], 1.)
        self.assertEqual(ledger.disk_generation[0], 1)

    def test_hidden_destruction_outside_stress_stays_hidden(self):
        ledger = fixture(lambda w, c, d: DiskStressLedger(w, c, d, waves=[DiskWave(2., (2,))]))
        ledger._world_events()
        ledger._push(1., "destroy_node", 0)
        ledger._process(1.01)
        self.assertFalse(ledger.f_exist[0])
        self.assertTrue(np.isinf(ledger.f_to[0]))
        self.assertEqual(ledger.disk_notifications, [])

    def test_stress_clears_partial_receiver_and_keeps_full_buffer(self):
        ledger = fixture(lambda w, c, d: DiskStressLedger(w, c, d, waves=[DiskWave(.11, (3,))]))
        ledger.down[0] = ledger.obs_iv[0] = np.array([0.]), np.array([8.])
        ledger._declare(0, 0., "test")
        ledger._world_events()
        ledger._repair(0., [(0, [0], [3])])
        ledger._process(.11)
        self.assertTrue(any(r["end"] is None for r in ledger.receiver_prefixes))
        ledger._process(.11001)
        self.assertTrue(all(r["end"] is not None for r in ledger.receiver_prefixes))
        self.assertIn(0, ledger.buffers)
        ledger._process(.3)
        self.assertTrue(any(not r["ok"] for r in ledger.block_results))
        writes = [r for r in ledger.segments if r["phase"] == "write_block" and r["start"] > .11]
        self.assertEqual(writes[0]["stripe"], 0)
        self.assertEqual(writes[0]["disk_generation"], 1)
        self.assertFalse(ledger.log_loss)

    def test_public_targets_ignore_future_life_and_tie_by_domain(self):
        nodes = pd.DataFrame({"join_t": [0., 0., 0., 0., 2.], "dc": [2, 1, 0, 2, 0],
                              "alive_to": [1., 9., 3., 5., 8.]})
        before = public_wave_targets(nodes)
        nodes.alive_to = 0.
        self.assertEqual(before, public_wave_targets(nodes))
        self.assertEqual(before["domains"], [2, 0, 1])
        self.assertEqual(before["groups"], [(0, 3), (2,), (1,)])

    def test_invalid_schedule_and_no_future_schedule_in_snapshot(self):
        for wave in (DiskWave(1., (0, 0)), DiskWave(-1., (0,)), DiskWave(1., (99,))):
            with self.assertRaises(ValueError):
                fixture(lambda w, c, d: DiskStressLedger(w, c, d, waves=[wave]))
        ledger = fixture(lambda w, c, d: DiskStressLedger(w, c, d, waves=[DiskWave(2., (0,))]))
        self.assertNotIn("stress_waves", ledger.queue_observation(0.))


if __name__ == "__main__":
    unittest.main()
