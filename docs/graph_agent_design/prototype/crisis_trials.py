"""Р5: наблюдаемые предупреждения, внезапные кризисы и перегрузка от работ."""
from dataclasses import dataclass, replace
from fractions import Fraction
import json
from pathlib import Path

from recovery import ObjectSpec, RecoveryEngine, RecoveryPolicy, StorageNode
from run_examples import encode
from run_recovery import summarize


@dataclass(frozen=True)
class WarningOutcome:
    time: int
    observed_at: int
    site: str
    crisis: bool


class WarningForecaster:
    def __init__(self, records, cutoff):
        self.counts = {}
        for row in records:
            if type(row.crisis) is not bool or not 0 <= row.time < row.observed_at:
                raise ValueError("Некорректная метка предупреждения")
            if row.observed_at <= cutoff:
                hits, total = self.counts.get(row.site, (0, 0))
                self.counts[row.site] = hits + int(row.crisis), total + 1

    def probability(self, site):
        hits, total = self.counts.get(site, (0, 0))
        return Fraction(hits + 1, total + 2)


class ForecastRecoveryPolicy(RecoveryPolicy):
    def __init__(self, forecaster, threshold=Fraction(3, 5), risk_model=None):
        super().__init__(risk_model)
        self.forecaster, self.threshold = forecaster, threshold

    def __call__(self, view):
        avoid_sites = {site for _, site, _ in view.alerts
                       if self.forecaster.probability(site) >= self.threshold}
        sites = dict(view.sites)
        protected_view = replace(view, up=tuple(node for node in view.up if sites[node] not in avoid_sites))
        return super().__call__(protected_view)


def training_records():
    # Прошлые исходы предупреждений. Инференс не получает расписание будущих шоков.
    records = []
    for t in range(40):
        records.append(WarningOutcome(t, t + 1, "S0", t % 5 != 0))
        records.append(WarningOutcome(t, t + 1, "S1", t % 5 == 0))
    return records


def crisis_engine(n=3):
    engine = RecoveryEngine([StorageNode(node, f"S{i}") for i, node in enumerate("ABCDE")],
                            [ObjectSpec("O", 2, n)])
    for idx, node in enumerate("ABCDE"[:n]):
        engine.add_chunk("O", idx, node)
    return engine


def overload_engine(parallel):
    engine = RecoveryEngine([StorageNode("A", "source"), StorageNode("B", "source"),
                             StorageNode("C", "destination"), StorageNode("D", "destination")],
                            [ObjectSpec("O1", 2, 3), ObjectSpec("O2", 2, 3)],
                            parallel_limit=parallel, overload_flow_limit=2,
                            overload_grace=1, overload_cooldown=2)
    for obj in ("O1", "O2"):
        engine.add_chunk(obj, 0, "A")
        engine.add_chunk(obj, 1, "B")
    return engine


def run_cases():
    forecaster = WarningForecaster(training_records(), 40)
    shocks = [(2, {"kind": "destroy", "node": "A"}),
              ("2.5", {"kind": "destroy", "node": "B"})]
    warning = [(0, {"kind": "warning", "site": "S0", "duration": 3})]
    cases = {}
    cases["predictable_reactive"] = summarize(crisis_engine().run(7, events=warning + shocks, policy=RecoveryPolicy(), decision_interval="0.5"))
    cases["predictable_forecast"] = summarize(crisis_engine().run(7, events=warning + shocks, policy=ForecastRecoveryPolicy(forecaster), decision_interval="0.5"))
    cases["sudden_forecast_n3"] = summarize(crisis_engine().run(7, events=shocks, policy=ForecastRecoveryPolicy(forecaster), decision_interval="0.5"))
    cases["sudden_forecast_n4"] = summarize(crisis_engine(4).run(7, events=shocks, policy=ForecastRecoveryPolicy(forecaster), decision_interval="0.5"))
    false_warning = [(0, {"kind": "warning", "site": "S1", "duration": 3})]
    cases["frequent_false_warning_ignored"] = summarize(crisis_engine().run(5, events=false_warning, policy=ForecastRecoveryPolicy(forecaster)))
    cases["credible_warning_but_no_crisis"] = summarize(crisis_engine().run(5, events=warning, policy=ForecastRecoveryPolicy(forecaster)))
    cases["unlimited_recovery_overload"] = summarize(overload_engine(2).run(9, policy=RecoveryPolicy()))
    cases["throttled_recovery"] = summarize(overload_engine(1).run(9, policy=RecoveryPolicy()))
    return forecaster, cases


def main():
    forecaster, cases = run_cases()
    output = Path(__file__).with_name("crisis_results.json")
    output.write_text(json.dumps({"schema": "taris-crisis-trials/1",
        "training": {site: str(forecaster.probability(site)) for site in ("S0", "S1")},
        "cases": cases}, ensure_ascii=False, indent=2, default=encode) + "\n", encoding="utf-8")
    for name, case in cases.items():
        print(name, "потери:", case["lost"], "трафик:", case["traffic_chunk_units"],
              "перегрузки:", sum(event["event"] == "overload" for event in case["log"]))


if __name__ == "__main__":
    main()
