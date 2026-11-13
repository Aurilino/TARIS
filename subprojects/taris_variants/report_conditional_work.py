"""Контролируемый runtime-пример I4.7a; не новая случайная серия."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import hashlib
import json
import numpy as np
from subprojects.taris_variants.observable_memory import ObservableMemoryLedger
from subprojects.taris_variants.conditional_work import conditional_work
from subprojects.taris_variants.test_physical import fixture


def run():
    rows = []
    for failure in (False, True):
        ledger = fixture(ObservableMemoryLedger)
        ledger.down[0] = ledger.obs_iv[0] = np.array([0.]), np.array([8.])
        ledger._declare(0, 0., "controlled_outage")
        ledger._repair(0., [(0, [0], [3])])
        if failure:
            ledger._push(.09001, "queue_buffer_fail", 0)
        times = (.09, .09002) if failure else (.09, .105, .11, .12)
        for t in times:
            ledger._process(t)
            snapshot = ledger.queue_observation(t)
            rows.append({"local_fault": failure, "snapshot": snapshot,
                "scenarios": [conditional_work(snapshot, mode) for mode in ("trust_ack", "restart_partial")]})
    paths = [Path(__file__), Path(__file__).with_name("conditional_work.py"),
             Path(__file__).with_name("observable_queue.py"), Path(__file__).with_name("observable_memory.py"),
             Path(__file__).with_name("test_physical.py")]
    result = {"schema": "taris-conditional-work-example/1", "random_worlds": 0,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "cases": rows}
    output = Path(__file__).with_name("conditional_work_i47a")
    output.mkdir(exist_ok=True)
    (output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# Условная работа I4.7a", "", "Контролируемые runtime-снимки; новых случайных миров нет.", "",
             "| Отказ памяти | Время, сутки | Сценарий | Объём repair | Цена передачи | Неизвестные proof |",
             "|---|---:|---|---:|---:|---:|"]
    for row in rows:
        for scenario in row["scenarios"]:
            lines.append(f'| {row["local_fault"]} | {row["snapshot"]["time"]:.5f} | {scenario["mode"]} | '
                f'{scenario["known_repair_volume"]:.6f} | {scenario["conditional_repair_transfer_cost"]:.6f} | '
                f'{len(scenario["unknown_proof_jobs"])} |')
    lines += ["", "Условная цена — только передача при цене 0,1 за объём. Ожидание, CPU, память во времени, "
              "будущий ущерб и полный резерв не оценены. Текущая фаза условно прерывается; её неподтверждённый блок повторяется."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("I4.7a: 6 runtime-снимков, 12 условных сценариев сохранены")


if __name__ == "__main__":
    run()
