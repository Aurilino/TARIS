import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
from dataclasses import asdict
import unittest
import numpy as np
import pandas as pd
from subprojects.taris_variants.test_physical import fixture
from subprojects.taris_variants.queued import QueueSettings
from subprojects.taris_variants.resumable import ResumableLedger, TransportSettings
from subprojects.taris_variants.audit_resumable import audit_record


def resumable(stripes=4):
    ledger = fixture(lambda world, chain, data: ResumableLedger(world, chain, data,
        QueueSettings(rate_size_per_hour=1/24), TransportSettings(stripes=stripes, retry_hours=1.)))
    ledger.cfg = replace(ledger.cfg, repair_hours=0.)
    ledger.w.cfg = ledger.cfg
    return ledger


def start(ledger):
    ledger.down[0] = ledger.obs_iv[0] = np.array([.5]), np.array([7.])
    ledger._declare(0, .5, "простой > T_rep")
    ledger._repair(1., [(0, [0], [3])])


class ResumableTests(unittest.TestCase):
    def test_read_keeps_verified_prefix_after_interruption(self):
        ledger = resumable()
        ledger.down[1] = np.array([1.3]), np.array([1.4])
        start(ledger)
        ledger._process(3.)
        attempts = [s["stripe"] for s in ledger.segments if s["phase"] == "read_block"]
        self.assertEqual(attempts, [0, 1, 1, 2, 3])
        self.assertEqual(len(ledger.buffer_intervals), 1)
        self.assertEqual(sum(r[-1] == "written" for r in ledger.operation_outcomes), 1)
        self.assertFalse(ledger.buffers)

    def test_write_keeps_prefix_across_ordinary_outage(self):
        ledger = resumable()
        ledger.down[3] = np.array([2.15]), np.array([2.6])
        start(ledger)
        ledger._process(4.)
        attempts = [s["stripe"] for s in ledger.segments if s["phase"] == "write_block"]
        self.assertEqual(attempts, [0, 1, 1, 2, 3])
        self.assertEqual(ledger.operation_outcomes[-1][-1], "written")

    def test_destruction_resets_receiver_prefix(self):
        ledger = resumable()
        start(ledger)
        ledger._push(2.15, "destroy_node", 3)
        ledger._process(4.)
        attempts = [s["stripe"] for s in ledger.segments if s["phase"] == "write_block"]
        self.assertEqual(attempts, [0, 1, 0, 1, 2, 3])
        self.assertTrue(any(s["disk_generation"] == 1 for s in ledger.segments if s["phase"] == "write_block"))

    def test_sources_can_change_between_decoded_stripes(self):
        ledger = resumable()
        ledger._add_fragment(0, 3, 4, 0.)
        start(ledger)
        ledger._push(1.3, "destroy_node", 1)
        ledger._process(4.)
        reads = [s for s in ledger.segments if s["phase"] == "read_block"]
        self.assertEqual(reads[0]["sources"], [1, 2])
        self.assertTrue(any(s["sources"] == [2, 3] for s in reads))
        self.assertTrue(ledger.buffer_intervals)

    def test_last_buffer_is_retained_after_duplicate_index_write(self):
        ledger = resumable()
        start(ledger)
        ledger._push(2.1, "destroy_node", 1)
        ledger._push(2.1, "destroy_node", 2)
        ledger._process(3.)
        self.assertIn(0, ledger.buffers)
        self.assertFalse(ledger.log_loss)
        self.assertEqual(ledger.obj_exist[0], 1)
        # Агент узнаёт отрицательную целостность из закрывающей проверки и даёт новые цели.
        ledger._repair(3., [(0, [1, 2], [4, 5])])
        ledger._process(5.)
        self.assertFalse(ledger.buffers)
        self.assertFalse(ledger.log_loss)
        self.assertGreaterEqual(ledger.obj_exist[0], 2)
        self.assertEqual(ledger.buffer_intervals[0]["reason"], "confirmed_disk_code")

    def test_failure_of_retained_buffer_is_real_loss(self):
        ledger = resumable()
        start(ledger)
        ledger._push(2.1, "destroy_node", 1)
        ledger._push(2.1, "destroy_node", 2)
        ledger._push(3., "queue_buffer_fail", 0)
        ledger._process(4.)
        self.assertEqual(ledger.log_loss, [(0, 3.)])

    def test_memory_failure_discards_read_prefix(self):
        ledger = resumable()
        start(ledger)
        ledger._push(1.3, "queue_buffer_fail", 0)
        ledger._process(4.)
        attempts = [s["stripe"] for s in ledger.segments if s["phase"] == "read_block"]
        self.assertEqual(attempts[:3], [0, 1, 0])
        self.assertTrue(all(r["end"] is not None for r in ledger.decoded_prefixes))

    def test_close_check_uses_channel_and_k_distinct_indices(self):
        ledger = resumable()
        start(ledger)
        ledger._process(4.)
        checks = [s for s in ledger.segments if s["phase"] == "close_proof"]
        self.assertTrue(checks)
        self.assertAlmostEqual(sum(s["volume"] for s in checks), .01)
        self.assertEqual(len({ledger.f_idx[f] for f in checks[-1]["sources"]}), 2)

    def test_retained_buffer_blocks_following_fifo_work(self):
        ledger = resumable()
        ledger._add_fragment(0, 0, 4, 0.)
        ledger.down[4] = ledger.obs_iv[4] = np.array([.5]), np.array([3.])
        ledger._declare(3, .5, "простой > T_rep")
        start(ledger)
        ledger._push(2.1, "destroy_node", 1)
        ledger._push(2.1, "destroy_node", 2)
        ledger._process(3.)
        ledger._return_node(4, 3.)
        ledger._process(4.)
        self.assertTrue(ledger.waiting)
        self.assertIsNone(ledger.jobs[ledger.waiting[0]]["started"])
        self.assertIn(0, ledger.buffers)

    def test_interrupted_final_read_stripe_cannot_invent_full_buffer(self):
        ledger = resumable()
        start(ledger)
        ledger._push(1.9, "destroy_node", 1)
        ledger._push(1.9, "destroy_node", 2)
        ledger._process(3.)
        self.assertFalse(ledger.buffer_intervals)
        self.assertEqual(ledger.log_loss, [(0, 1.9)])

    def test_invalid_transport_settings(self):
        for settings in (TransportSettings(stripes=0), TransportSettings(stripes=1.5),
                         TransportSettings(retry_hours=0), TransportSettings(retry_hours=float("nan"))):
            with self.assertRaises(ValueError):
                settings.validate()

    def test_independent_audit_of_retention_and_memory_failures(self):
        for mode, expected in (("retained_and_repaired", 0), ("buffer_failure", 1), ("prefix_failure", 1)):
            with self.subTest(mode=mode):
                ledger = resumable()
                start(ledger)
                for node in (1, 2):
                    ledger._push(2.1, "destroy_node", node)
                if mode == "prefix_failure":
                    ledger._push(1.3, "queue_buffer_fail", 0)
                if mode == "buffer_failure":
                    ledger._push(3., "queue_buffer_fail", 0)
                ledger._process(3.)
                if mode == "retained_and_repaired":
                    ledger._repair(3., [(0, [1, 2], [4, 5])])
                ledger._process(8.)
                ledger.ch.true_down = pd.DataFrame([
                    {"node": node, "start": a, "end": b} for node, (starts, ends) in ledger.down.items()
                    for a, b in zip(starts, ends)])
                record = {"queue_settings": asdict(ledger.queue_settings), "transport_settings": asdict(ledger.transport),
                    "metrics": ledger.queue_report(8.), "jobs": ledger.jobs, "segments": ledger.segments,
                    "buffers": ledger.buffer_intervals, "block_results": ledger.block_results, "close_checks": ledger.close_checks,
                    "decoded_prefixes": ledger.decoded_prefixes, "receiver_prefixes": ledger.receiver_prefixes,
                    "queue_events": ledger.queue_events, "losses": [{"obj": obj, "t": t} for obj, t in ledger.log_loss],
                    "reserved_by_node": ledger.reserved.tolist(), "allocated_slots_by_node": ledger.used.tolist(),
                    "physical_fragments": [{"obj": ledger.f_obj[f], "idx": ledger.f_idx[f], "node": ledger.f_node[f],
                        "start": ledger.f_from[f], "physical_end": ledger.f_dead[f] if np.isfinite(ledger.f_dead[f]) else None,
                        "allocated": ledger.allocated[f]} for f in range(len(ledger.f_obj))]}
                audited = audit_record(record, ledger.cfg, ledger.w, ledger.ch, ledger.data)
                self.assertEqual(audited["irrecoverable_objects"], expected)


if __name__ == "__main__":
    unittest.main()
