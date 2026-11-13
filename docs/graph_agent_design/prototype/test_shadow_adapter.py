import unittest

from learning import NodeSpec, fit
from shadow_adapter import ChainObservationAdapter, PublicSlice


class ShadowAdapterTests(unittest.TestCase):
    def test_unregistered_is_unknown_and_loss_is_not_invented(self):
        adapter = ChainObservationAdapter((NodeSpec("A", "s", 0), NodeSpec("B", "s", 1)))
        adapter.ingest(PublicSlice(0, 1 / 24, ("A",), ("A",)))
        row = adapter.store.known(1)[0]
        self.assertEqual(dict(row.values), {"A": True, "B": None})
        self.assertEqual(row.layer, "availability")
        learned = fit(adapter.store, adapter.registry, 1)
        self.assertEqual(learned.sample_counts["loss", "night"], 0)
        self.assertGreater(learned.exact_model().loss.own("A"), 0)

    def test_future_slice_does_not_change_past_fit(self):
        registry = (NodeSpec("A", "s", 0),)
        adapter = ChainObservationAdapter(registry)
        adapter.ingest(PublicSlice(0, 1 / 24, ("A",), ()))
        first = fit(adapter.store, registry, 1)
        adapter.ingest(PublicSlice(1 / 24, 2 / 24, ("A",), ("A",)))
        self.assertEqual(first.probabilities, fit(adapter.store, registry, 1).probabilities)

    def test_invalid_slice_and_gap(self):
        registry = (NodeSpec("A", "s", 0),)
        for payload in (PublicSlice(0, 1 / 24, (), ("A",)),
                        PublicSlice(0, 1 / 24, ("X",), ()),
                        PublicSlice(1 / 24, 2 / 24, ("A",), ())):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                ChainObservationAdapter(registry).ingest(payload)


if __name__ == "__main__":
    unittest.main(verbosity=2)
