"""I4.7b: внешний контролируемый аудит условного расчёта."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from copy import deepcopy
import hashlib
import json
from math import isclose
import numpy as np
from subprojects.taris_variants.observable_memory import ObservableMemoryLedger
from subprojects.taris_variants.conditional_work import conditional_work
from subprojects.taris_variants.test_physical import fixture


def build(interruption=False):
    ledger = fixture(ObservableMemoryLedger)
    ledger.down[0] = ledger.obs_iv[0] = np.array([0.]), np.array([8.])
    if interruption:
        ledger.down[1] = ledger.obs_iv[1] = np.array([.0901]), np.array([.092])
    ledger._declare(0, 0., "controlled_outage")
    ledger._repair(0., [(0, [0], [3])])
    return ledger


def served_after(segments, t, end):
    return sum(r["volume"] * max(0., min(end, r["end"])-max(t, r["start"]))
               / (r["end"]-r["start"]) for r in segments)


def evaluate(t, interruption=False):
    ledger = build(interruption)
    ledger._process(t)
    snapshot = ledger.queue_observation(t)
    forecast = conditional_work(snapshot)
    phase = snapshot["planned_channel_phase"]
    already_served = 0. if phase is None else phase["volume"]*(t-phase["start"])/(phase["end"]-phase["start"])
    ledger._process(.3)
    if any(job["finished"] is None for job in ledger.jobs):
        raise AssertionError("Контролируемая очередь цензурирована")
    actual = served_after(ledger.segments, t, .3)
    failed = sum(not row["ok"] for row in ledger.block_results)
    residual = actual-(forecast["known_repair_volume"]-already_served)
    if not interruption and not isclose(residual, 0., abs_tol=1e-10):
        raise AssertionError("Не сошёлся объём продолжения без новых отказов")
    if interruption and not (failed > 0 and residual > 0):
        raise AssertionError("Не наблюдается оплаченный повтор")
    return {"time": t, "interruption": interruption, "snapshot": snapshot,
        "conditional": forecast, "actual_future_served": actual,
        "already_served_unacknowledged": already_served, "extra_vs_uninterrupted": residual,
        "failed_blocks": failed, "unfinished_jobs": 0, "segments": deepcopy(ledger.segments),
        "block_results": deepcopy(ledger.block_results)}


def run():
    cases = [evaluate(t) for t in (.09, .105, .11)] + [evaluate(.09, True)]
    assert cases[0]["snapshot"] == cases[-1]["snapshot"], "Будущий простой изменил текущий снимок"
    paths = [Path(__file__), Path(__file__).with_name("conditional_work.py"),
             Path(__file__).with_name("observable_memory.py"), Path(__file__).with_name("observable_queue.py")]
    result = {"schema": "taris-continuation-check/1", "random_worlds": 0,
        "identical_past_snapshots": True, "uncensored": True, "cases": cases,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    output = Path(__file__).with_name("continuation_i47b")
    output.mkdir(exist_ok=True)
    (output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Сверка продолжения I4.7b", "", "| Время | Новый простой | Условный объём | Фактический будущий | Уже обслужено в блоке | Дополнительный объём |",
             "|---:|---|---:|---:|---:|---:|"]
    for row in cases:
        lines.append(f'| {row["time"]} | {row["interruption"]} | {row["conditional"]["known_repair_volume"]:.9f} | '
                     f'{row["actual_future_served"]:.9f} | {row["already_served_unacknowledged"]:.9f} | '
                     f'{row["extra_vs_uninterrupted"]:.9f} |')
    lines += ["", "Все задания завершены. Снимки до будущего прерывания одинаковы. Без нового отказа объёмы сходятся "
              "с поправкой на уже обслуженную часть блока. При прерывании появляется оплаченный повтор. "
              "Это контролируемая проверка, не оценка ожидаемых затрат и не вероятностная калибровка."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("I4.7b: 4 продолжения проверены; наблюдаемое прошлое совпадает")


if __name__ == "__main__":
    run()
