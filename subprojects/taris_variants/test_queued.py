import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
import unittest
import numpy as np
import pandas as pd
from subprojects.taris_variants.test_physical import fixture
from subprojects.taris_variants.queued import QueuedLedger, QueueSettings
from subprojects.taris_variants.audit_queued import audit_record
from dataclasses import asdict


def queued(settings=None):
    # Используются те же заранее известные диски, что в проверке физического режима.
    ledger = fixture(lambda world, chain, data: QueuedLedger(world, chain, data, settings))
    ledger.cfg = replace(ledger.cfg, repair_hours=0.)
    ledger.w.cfg = ledger.cfg
    # Для удобства точных примеров: объект читается сутки, один чанк пишется полсуток.
    ledger.queue_settings = settings or QueueSettings(rate_size_per_hour=1/24)
    return ledger


class QueuedTests(unittest.TestCase):
    def start_one_repair(self, ledger):
        ledger.down[0] = ledger.obs_iv[0] = np.array([.5]), np.array([4.])
        ledger._declare(0, .5, "простой > T_rep")
        ledger._repair(1., [(0, [0], [3])])
        ledger._process(1.01)

    def test_buffer_saves_after_source_and_old_copy_are_destroyed(self):
        ledger = queued()
        self.start_one_repair(ledger)
        ledger._push(2.1, "destroy_node", 0)
        ledger._push(2.1, "destroy_node", 1)
        ledger._process(2.2)
        self.assertIn(0, ledger.buffers)
        self.assertEqual(ledger.obj_exist[0], 1)
        self.assertFalse(ledger.log_loss)
        ledger._process(2.6)
        self.assertFalse(ledger.buffers)
        self.assertFalse(ledger.log_loss)
        self.assertEqual(ledger.obj_exist[0], 2)
        self.assertAlmostEqual(ledger.queue_report(2.6)["buffer_only_object_days"], .4)

    def test_duplicate_index_write_cannot_replace_full_buffer(self):
        ledger = queued()
        self.start_one_repair(ledger)
        ledger._push(2.1, "destroy_node", 1)
        ledger._push(2.1, "destroy_node", 2)
        ledger._process(2.2)
        self.assertFalse(ledger.log_loss)
        ledger._process(2.6)
        self.assertEqual(ledger.operation_outcomes[-1][-1], "written")
        self.assertEqual(ledger.obj_exist[0], 1)
        self.assertEqual(ledger.log_loss, [(0, 2.5)])

    def test_buffer_failure_causes_loss_when_disk_code_is_insufficient(self):
        ledger = queued()
        self.start_one_repair(ledger)
        ledger._push(2.1, "destroy_node", 1)
        ledger._push(2.1, "destroy_node", 2)
        ledger._push(2.2, "queue_buffer_fail", 0)
        ledger._process(2.6)
        self.assertEqual(ledger.log_loss, [(0, 2.2)])
        self.assertEqual(ledger.operation_outcomes[-1][-1], "failed")

    def test_target_failure_discards_buffer_after_attempt(self):
        ledger = queued()
        self.start_one_repair(ledger)
        ledger._push(2.1, "destroy_node", 1)
        ledger._push(2.1, "destroy_node", 2)
        ledger.down[3] = np.array([2.2]), np.array([3.])
        ledger._process(2.6)
        self.assertEqual(ledger.log_loss, [(0, 2.5)])
        self.assertEqual(ledger.reserved.sum(), 0)

    def test_source_outage_inside_read_is_not_hidden_by_recovery(self):
        ledger = queued()
        self.start_one_repair(ledger)
        ledger.down[1] = np.array([1.2]), np.array([1.3])
        ledger._process(2.6)
        self.assertFalse(ledger.buffer_intervals)
        self.assertEqual(ledger.jobs[0]["status"], "read_failed")
        self.assertEqual(ledger.reserved.sum(), 0)

    def test_duplicate_requests_do_not_duplicate_pending_work(self):
        ledger = queued()
        self.start_one_repair(ledger)
        count = len(ledger.jobs)
        ledger._repair(1.1, [(0, [0], [3])])
        self.assertEqual(len(ledger.jobs), count)
        self.assertEqual(ledger.reserved[3], 1)

    def test_finite_queue_rejects_additional_work_without_leaking_slots(self):
        ledger = queued(QueueSettings(rate_size_per_hour=1/24, max_jobs=1))
        self.start_one_repair(ledger)
        ledger._declare(1, 1.1, "простой > T_rep")
        ledger._repair(1.2, [(0, [1], [4])])
        self.assertGreater(ledger.queue_rejections["repair_queue_full"], 0)
        self.assertEqual(ledger.reserved[4], 0)

    def test_small_buffer_rejects_repair(self):
        ledger = queued(QueueSettings(buffer_capacity_size=.5))
        ledger._declare(0, .5, "простой > T_rep")
        ledger._repair(1., [(0, [0], [3])])
        self.assertEqual(ledger.queue_rejections["buffer_too_small"], 1)
        self.assertEqual(ledger.reserved.sum(), 0)

    def test_channel_budget_is_clipped_at_horizon(self):
        ledger = queued()
        self.start_one_repair(ledger)
        report = ledger.queue_report(1.5)
        self.assertAlmostEqual(report["serviced_volume"]["read"], .5)
        self.assertNotIn("write", report["serviced_volume"])
        self.assertEqual(report["unfinished_jobs"], 1)

    def test_fifo_and_paid_return_verification(self):
        ledger = queued()
        self.start_one_repair(ledger)
        # Узел 0 вернулся раньше, но proof ждёт занятого сервера.
        ledger.down[0] = ledger.obs_iv[0] = np.array([.5]), np.array([1.1])
        ledger._return_node(0, 1.1)
        self.assertFalse(ledger.returns)
        ledger._process(3.)
        self.assertTrue(ledger.returns or ledger.cleanup)
        self.assertGreater(ledger.queue_report(3.)["serviced_volume"]["proof"], 0.)
        starts = [j["started"] for j in ledger.jobs]
        self.assertEqual(starts, sorted(starts))
        segments = ledger.segments
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(segments, segments[1:])))

    def test_zero_proof_setup_does_not_break_world_event_dispatch(self):
        ledger = queued(replace(QueueSettings(), proof_setup_hours=0.))
        ledger._declare(0, .5, "простой > T_rep")
        ledger._push(1., "observed_return", 0)
        ledger._process(2.)
        self.assertEqual(len(ledger.returns), 1)

    def test_invalid_settings_are_rejected(self):
        for settings in (QueueSettings(rate_size_per_hour=0), QueueSettings(max_jobs=0),
                         QueueSettings(buffer_capacity_size=float("nan")), QueueSettings(proof_fraction=0)):
            with self.assertRaises(ValueError):
                settings.validate()

    def test_independent_audit_of_buffer_only_intervals_and_losses(self):
        for destroyed, fail_buffer, expected_loss in (([0, 1], False, 0), ([1, 2], False, 1), ([0, 1], True, 1)):
            with self.subTest(destroyed=destroyed, fail_buffer=fail_buffer):
                ledger = queued()
                self.start_one_repair(ledger)
                for node in destroyed:
                    ledger._push(2.1, "destroy_node", node)
                if fail_buffer:
                    ledger._push(2.2, "queue_buffer_fail", 0)
                ledger._process(8.)
                ledger.ch.true_down = pd.DataFrame([
                    {"node": node, "start": a, "end": b} for node, (starts, ends) in ledger.down.items()
                    for a, b in zip(starts, ends)])
                record = {"queue_settings": asdict(ledger.queue_settings), "metrics": ledger.queue_report(8.),
                    "jobs": ledger.jobs, "segments": ledger.segments, "buffers": ledger.buffer_intervals,
                    "queue_events": ledger.queue_events, "repair_outcomes": ledger.operation_outcomes,
                    "losses": [{"obj": obj, "t": t} for obj, t in ledger.log_loss],
                    "reserved_by_node": ledger.reserved.tolist(), "allocated_slots_by_node": ledger.used.tolist(),
                    "physical_fragments": [{"obj": ledger.f_obj[f], "idx": ledger.f_idx[f], "node": ledger.f_node[f],
                        "start": ledger.f_from[f], "physical_end": ledger.f_dead[f] if np.isfinite(ledger.f_dead[f]) else None,
                        "allocated": ledger.allocated[f]} for f in range(len(ledger.f_obj))]}
                audited = audit_record(record, ledger.cfg, ledger.w, ledger.ch, ledger.data)
                self.assertEqual(audited["irrecoverable_objects"], expected_loss)


if __name__ == "__main__":
    unittest.main()
