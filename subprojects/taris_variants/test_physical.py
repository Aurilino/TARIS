import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
from taris_sim.config import Config
from taris_sim.ledger import Ledger, NetworkView
from subprojects.taris_variants.physical import PhysicalLedger


def fixture(kind=PhysicalLedger):
    """Без случайных событий: известные диски, пинги и объекты."""
    cfg = Config(horizon_days=8, k=2, repair_hours=2)
    nodes = pd.DataFrame({"join_t": [0.]*6, "owner": range(6),
                          "alive_to": [np.inf]*6, "leave_t": [np.inf]*6})
    down = pd.DataFrame(columns=["node", "start", "end", "source"])
    outages = pd.DataFrame(columns=["node", "start", "end", "cls", "destroyed"])
    world = SimpleNamespace(cfg=cfg, nodes=nodes, outages=outages, silent=pd.DataFrame(columns=["node", "t"]))
    chain = SimpleNamespace(observed=down.drop(columns="source"), true_down=down,
                            audits=pd.DataFrame({"node": [0], "t": [6.], "passed_avail": [True]}))
    data = SimpleNamespace(capacity=np.ones(6, int)*10,
                          objects=pd.DataFrame({"created_epoch": [0], "size": [1.]}))
    ledger = kind(world, chain, data)
    ledger._reset_state()
    ledger.obj_n[0] = 3
    for idx, node in enumerate((0, 1, 2)):
        ledger._add_fragment(0, idx, node, 0.)
    return ledger


class PhysicalLedgerTests(unittest.TestCase):
    def test_controlled_full_run_separates_timeout_loss_from_outage(self):
        outcomes = {}
        for kind in (Ledger, PhysicalLedger):
            ledger = fixture(kind)
            ledger.data.new_objects = lambda epoch: np.array([0]) if epoch == 0 else np.array([], int)
            ledger.ch.reset = lambda: None
            ledger.ch.step = lambda: {"seed": "00" * 32}
            ledger.w.outages = pd.DataFrame({"node": [0, 1], "start": [.2, .2],
                "end": [4., 4.], "cls": ["total", "total"], "destroyed": [False, False]})
            for node in (0, 1):
                ledger.down[node] = ledger.obs_iv[node] = np.array([.2]), np.array([4.])
            policy = SimpleNamespace(n=3, name="controlled", alarm=lambda t, v: [],
                on_epoch=lambda t, v: [], migrate=lambda t, v: [], release=lambda t, v: [],
                place=lambda obj, t, v: [0, 1, 2])
            result = ledger.run(policy)
            outcomes[kind] = len(result.losses)
            if kind is PhysicalLedger:
                self.assertEqual(len(ledger.returns), 2)
                self.assertAlmostEqual(ledger.physical_report(8.)["unavailable_object_days"], 3.8)
                self.assertEqual(result.metrics()["schema"], "taris-physical-metrics/1")
                self.assertEqual(result.metrics()["physical_lost_objects"], 0)
                with self.assertRaises(NotImplementedError):
                    result.unavailability()
        self.assertEqual(outcomes, {Ledger: 1, PhysicalLedger: 0})

    def initial_run(self, down_nodes):
        ledger = fixture()
        ledger.data.new_objects = lambda epoch: np.array([0]) if epoch == 0 else np.array([], int)
        ledger.ch.reset = lambda: None
        ledger.ch.step = lambda: {"seed": "00" * 32}
        for node in down_nodes:
            ledger.down[node] = np.array([0.]), np.array([1.])
        policy = SimpleNamespace(n=3, name="fixture", alarm=lambda t, v: [], on_epoch=lambda t, v: [],
            migrate=lambda t, v: [], release=lambda t, v: [], place=lambda obj, t, v: [0, 1, 2])
        ledger.run(policy, epochs=1)
        return ledger

    def test_initial_proposal_is_not_automatically_a_successful_write(self):
        ledger = self.initial_run([0])
        self.assertEqual(ledger.initial_writes[0][3], [1, 2])
        self.assertEqual(ledger.missing[0], {0})
        self.assertFalse(ledger.log_loss)
        self.assertEqual(ledger.physical_report(1.)["initial_failed_writes"], 1)

    def test_incomplete_initial_admission_is_separate_from_later_loss(self):
        ledger = self.initial_run([0, 1])
        report = ledger.physical_report(1.)
        self.assertEqual(report["initial_incomplete_objects"], 1)
        self.assertEqual(report["lost_admitted_objects"], 0)
        self.assertEqual(report["unavailable_object_days"], 1.)

    def test_timeout_is_not_physical_loss(self):
        physical, legacy = fixture(), fixture(Ledger)
        for ledger in (physical, legacy):
            ledger._declare(0, 2., "простой > T_rep")
            ledger._declare(1, 2., "простой > T_rep")
        self.assertEqual(len(legacy.log_loss), 1)
        self.assertFalse(physical.log_loss)
        self.assertEqual(physical.obj_exist[0], 3)
        self.assertEqual(physical.used.sum(), 3)
        self.assertEqual(NetworkView(physical).missing(0), {0, 1})

    def test_long_outage_recovers_and_counts_unavailability(self):
        ledger = fixture()
        for node in (0, 1):
            ledger.down[node] = np.array([1.]), np.array([4.])
            ledger.obs_iv[node] = np.array([1.]), np.array([4.])
            ledger._declare(node, 2., "простой > T_rep")
            ledger._return_node(node, 4.)
        self.assertEqual(ledger.missing[0], set())
        self.assertEqual(len(ledger.returns), 2)
        self.assertFalse(ledger.log_loss)
        report = ledger.physical_report(8.)
        self.assertEqual(report["unavailable_object_days"], 3.)
        self.assertEqual(report["physical_fragment_days"], 24.)

    def test_true_destruction_of_retired_copy_causes_loss(self):
        ledger = fixture()
        ledger._declare(0, 1., "простой > T_rep")
        ledger._push(2., "destroy_node", 0)
        ledger._push(3., "destroy_node", 1)
        ledger._process(4.)
        self.assertEqual(ledger.log_loss, [(0, 3.)])

    def test_silent_loss_still_hits_retired_copy(self):
        ledger = fixture()
        ledger._declare(0, 1., "простой > T_rep")
        ledger._push(2., "silent", (0, 0))
        ledger._process(3.)
        self.assertFalse(ledger.f_exist[0])
        self.assertEqual(ledger.f_dead[0], 2.)

    def test_duplicate_index_does_not_add_reconstruction_rank(self):
        ledger = fixture()
        ledger._add_fragment(0, 0, 3, 1.)
        ledger._destroy(1, 2.)
        ledger._destroy(2, 3.)
        self.assertEqual(ledger.obj_exist[0], 1)
        self.assertEqual(ledger.log_loss, [(0, 3.)])
        self.assertEqual(ledger.physical_report(8.)["unavailable_object_days"], 5.)

    def test_repair_reserves_capacity_and_cleans_redundant_copy(self):
        ledger = fixture()
        ledger._declare(0, 1., "простой > T_rep")
        ledger._repair(2., [(0, [0], [3])])
        self.assertEqual(ledger.reserved[3], 1)
        self.assertEqual(ledger.used[3], 1)
        ledger._process(3.)
        self.assertEqual(ledger.reserved.sum(), 0)
        self.assertEqual(ledger.operation_outcomes[0][-1], "written")
        self.assertFalse(ledger.f_exist[0])
        self.assertEqual(ledger.used.sum(), 3)
        self.assertEqual(ledger.obj_exist[0], 3)
        self.assertFalse(ledger.log_loss)

    def test_down_old_copy_keeps_its_slot_after_repair(self):
        ledger = fixture()
        ledger.down[0] = ledger.obs_iv[0] = np.array([1.]), np.array([5.])
        ledger._declare(0, 2., "простой > T_rep")
        ledger._repair(3., [(0, [0], [3])])
        ledger._process(4.)
        self.assertEqual(ledger.used.sum(), 4)
        ledger._return_node(0, 5.)
        self.assertEqual(ledger.used.sum(), 3)

    def test_repair_cannot_use_less_than_k_sources(self):
        ledger = fixture()
        for f in (0, 1):
            ledger.down[f] = np.array([1.]), np.array([5.])
            ledger._declare(f, 2., "простой > T_rep")
        ledger._repair(3., [(0, [0, 1], [3, 4])])
        self.assertFalse(ledger.log_repair)
        self.assertEqual(ledger.repair_rejections["sources_unavailable"], 1)

    def test_failed_target_does_not_materialize_fragment(self):
        ledger = fixture()
        ledger._declare(0, 1., "простой > T_rep")
        ledger.down[3] = np.array([2.05]), np.array([4.])
        ledger._repair(2., [(0, [0], [3])])
        ledger._process(3.)
        self.assertEqual(len(ledger.f_obj), 3)
        self.assertEqual(ledger.operation_outcomes[0][-1], "failed")
        self.assertEqual(ledger.reserved.sum(), 0)

    def test_sources_must_survive_until_completion_without_buffer(self):
        ledger = fixture()
        ledger._declare(0, 1., "простой > T_rep")
        ledger._repair(2., [(0, [0], [3])])
        ledger._push(2.01, "destroy_node", 1)
        ledger._process(3.)
        self.assertEqual(ledger.operation_outcomes[0][-1], "failed")
        self.assertEqual(len(ledger.f_obj), 3)

    def test_repeated_requests_cannot_overbook_one_slot(self):
        ledger = fixture()
        ledger.capacity[3] = 1
        ledger._declare(0, 1., "простой > T_rep")
        ledger._repair(2., [(0, [0], [3]), (0, [0], [3])])
        self.assertEqual(ledger.reserved[3], 1)
        self.assertEqual(len(ledger.log_repair), 1)

    def test_event_at_horizon_is_not_processed_early(self):
        ledger = fixture()
        ledger._push(8., "destroy_node", 0)
        ledger._process(8.)
        self.assertTrue(ledger.f_exist[0])

    def test_return_while_repair_pending_prevents_duplicate_write(self):
        ledger = fixture()
        ledger._declare(0, 1., "простой > T_rep")
        ledger._repair(2., [(0, [0], [3])])
        ledger._return_node(0, 2.01)
        ledger._process(3.)
        self.assertEqual(len(ledger.f_obj), 3)
        self.assertEqual(ledger.operation_outcomes[0][-1], "already_returned")
        self.assertEqual(ledger.used.sum(), 3)


if __name__ == "__main__":
    unittest.main()
