from fractions import Fraction
import unittest

from crisis_trials import (ForecastRecoveryPolicy, WarningForecaster, WarningOutcome,
                           crisis_engine, overload_engine, run_cases, training_records)
from recovery import RecoveryPolicy


class CrisisTests(unittest.TestCase):
    def test_forecast_uses_only_received_labels(self):
        records = training_records()
        a = WarningForecaster(records, 40)
        b = WarningForecaster(records + [WarningOutcome(50, 51, "S0", False)], 40)
        self.assertEqual(a.counts, b.counts)
        self.assertEqual(a.probability("S0"), Fraction(33, 42))
        self.assertEqual(a.probability("S1"), Fraction(9, 42))

    def test_predictable_and_sudden_are_different(self):
        _, cases = run_cases()
        self.assertEqual(cases["predictable_reactive"]["lost"], {"O": Fraction(5, 2)})
        self.assertFalse(cases["predictable_forecast"]["lost"])
        self.assertTrue(cases["sudden_forecast_n3"]["lost"])
        self.assertFalse(cases["sudden_forecast_n4"]["lost"])

    def test_warning_does_not_reveal_future_event(self):
        engine = crisis_engine()
        engine.apply("warning", site="S0", duration=3)
        snapshot = engine.snapshot()
        self.assertEqual(snapshot.alerts, ((Fraction(), "S0", Fraction(3)),))
        self.assertFalse(hasattr(snapshot, "shocks"))
        self.assertEqual(len(engine.physical), 3)

    def test_false_warnings_can_cost_traffic(self):
        _, cases = run_cases()
        self.assertEqual(cases["frequent_false_warning_ignored"]["traffic_chunk_units"], 0)
        self.assertEqual(cases["credible_warning_but_no_crisis"]["traffic_chunk_units"], 3)

    def test_overload_is_generated_by_execution_not_scripted(self):
        unguarded = overload_engine(2).run(9, policy=RecoveryPolicy())
        guarded = overload_engine(1).run(9, policy=RecoveryPolicy())
        self.assertTrue(any(row["event"] == "overload" for row in unguarded.log))
        self.assertFalse(any(row["event"] == "overload" for row in guarded.log))
        self.assertGreater(max(j.finished for j in unguarded.jobs), max(j.finished for j in guarded.jobs))
        self.assertEqual(unguarded.traffic, guarded.traffic)

    def test_warning_expires(self):
        forecaster = WarningForecaster(training_records(), 40)
        engine = crisis_engine()
        engine.apply("warning", site="S0", duration=1)
        engine.run(1)
        self.assertEqual(engine.snapshot().alerts, ())
        self.assertEqual(ForecastRecoveryPolicy(forecaster)(engine.snapshot()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
