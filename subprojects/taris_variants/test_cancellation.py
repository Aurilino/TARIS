import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import replace
import unittest
from copy import deepcopy
import numpy as np
from subprojects.taris_variants.test_physical import fixture
from subprojects.taris_variants.cancel_obsolete import CancellationLedger
from subprojects.taris_variants.physical import PhysicalLedger
from subprojects.taris_variants.queued import QueueSettings
from subprojects.taris_variants.resumable import TransportSettings
from subprojects.taris_variants.audit_cancellation import audit_cancellations


def ledger():
    result = fixture(lambda w, c, d: CancellationLedger(w, c, d,
        QueueSettings(rate_size_per_hour=1/24), TransportSettings(stripes=4)))
    result.cfg = replace(result.cfg, repair_hours=2.)
    result.w.cfg = result.cfg
    return result


class CancellationTests(unittest.TestCase):
    def test_auditor_rejects_cancellation_of_still_missing_index(self):
        L = ledger()
        L.down[0] = L.obs_iv[0] = np.array([.5]), np.array([1.01])
        L._declare(0, .5, "простой > T_rep")
        L._repair(1., [(0, [0], [3])])
        PhysicalLedger._return_node(L, 0, 1.01)
        L._process(2.)
        record = {"obsolete_cancellations": L.obsolete_cancellations, "metrics": L.queue_report(2.),
                  "jobs": L.jobs, "repair_outcomes": L.operation_outcomes, "returns": L.returns, "buffers": L.buffer_intervals}
        audit_cancellations(record)
        bad = deepcopy(record)
        bad["obsolete_cancellations"][0]["observed_missing"] = [0]
        with self.assertRaises(AssertionError):
            audit_cancellations(bad)

    def test_return_before_read_cancels_without_transfer(self):
        L = ledger()
        L.down[0] = L.obs_iv[0] = np.array([.5]), np.array([1.01])
        L._declare(0, .5, "простой > T_rep")
        L._repair(1., [(0, [0], [3])])
        PhysicalLedger._return_node(L, 0, 1.01)
        L._process(2.)
        self.assertEqual(L.jobs[0]["status"], "obsolete_before_read")
        self.assertFalse(L.segments)
        self.assertEqual(L.used.sum(), 3)
        self.assertEqual(L.reserved.sum(), 0)
        self.assertNotIn(0, L.obsolete_cancellations[0]["observed_missing"])

    def test_only_obsolete_part_of_request_is_removed(self):
        L = ledger()
        L.down[0] = L.obs_iv[0] = np.array([.5]), np.array([1.01])
        L.down[1] = L.obs_iv[1] = np.array([.5]), np.array([4.])
        L._declare(0, .5, "простой > T_rep")
        L._declare(1, .5, "простой > T_rep")
        L._repair(1., [(0, [0, 1], [3, 4])])
        PhysicalLedger._return_node(L, 0, 1.01)
        L._process(1.2)
        job = next(j for j in L.jobs if j["kind"] == "repair")
        self.assertEqual(job["indices"], [1])
        self.assertEqual(job["nodes"], [4])
        self.assertEqual(L.reserved[3], 0)
        self.assertEqual(L.reserved[4], 1)

    def test_partial_read_may_be_cancelled_but_paid_work_is_retained(self):
        L = ledger()
        L.down[0] = L.obs_iv[0] = np.array([.5]), np.array([1.4])
        L._declare(0, .5, "простой > T_rep")
        L._repair(1., [(0, [0], [3])])
        L._process(1.4)
        PhysicalLedger._return_node(L, 0, 1.4)
        L._process(2.)
        self.assertEqual(L.jobs[0]["status"], "obsolete_before_read")
        self.assertGreater(L.queue_report(2.)["serviced_volume"]["read_block"], 0.)
        self.assertFalse(L.buffer_intervals)
        self.assertTrue(all(r["end"] is not None for r in L.decoded_prefixes))

    def test_complete_buffer_still_requires_paid_close_proof(self):
        L = ledger()
        L.down[0] = L.obs_iv[0] = np.array([.5]), np.array([2.2])
        L._declare(0, .5, "простой > T_rep")
        L._repair(1., [(0, [0], [3])])
        L._process(2.2)
        self.assertIn(0, L.buffers)
        PhysicalLedger._return_node(L, 0, 2.2)
        L._process(4.)
        self.assertEqual(L.buffer_intervals[0]["reason"], "confirmed_disk_code")
        self.assertFalse(L.obsolete_cancellations)
        self.assertTrue(L.close_checks[-1]["ok"])


if __name__ == "__main__":
    unittest.main()
