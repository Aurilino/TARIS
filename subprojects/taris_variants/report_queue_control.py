"""I6: полная цена прогресса очереди, не только число потерь."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import hashlib
import json
from math import isclose
from subprojects.taris_variants.object_reserve import object_reserve
from subprojects.taris_variants.resource_costs import resource_usage, observed_charges, ResourcePrices


def main():
    folder = Path(__file__).parent
    source = folder/"series_i6_replay"
    audit = json.loads((source/"audit.json").read_text(encoding="utf-8"))
    assert audit["executions"] == 24
    rows = []
    for name, digest in audit["input_hashes"].items():
        path = source/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        assert case["config"]["k"] == 2 and case["config"]["object_size"] == 1.
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                m = record["metrics"]
                usage = resource_usage(record, 45., 2, 1.)
                stock = object_reserve(record, 2, 4, 45.)
                assert isclose(usage["served_volume"], sum(m["serviced_volume"].values()), abs_tol=1e-8)
                assert isclose(stock["physical_copy_days"], m["physical_fragment_days"], abs_tol=1e-8)
                rows.append({"mode": case["queue_control_mode"], "profile": profile, "variant": variant,
                    "losses": m["irrecoverable_objects"], "unfinished_jobs": m["unfinished_jobs"],
                    "finished_repairs": sum(j["kind"] == "repair" and j["finished"] is not None for j in record["jobs"]),
                    "finished_proofs": sum(j["kind"] == "proof" and j["finished"] is not None for j in record["jobs"]),
                    "confirmed_proofs": sum(j["status"] == "proof_completed" for j in record["jobs"]),
                    "restored_map_entries": len(record["returns"]), "job_statuses": m["job_statuses"],
                    "read_wait_pause_count": m["read_wait_pause_count"], "proof_bypass_starts": m["proof_bypass_starts"],
                    "parked_owner_days": m["parked_owner_days"], "parked_read_wait_days": m["parked_read_wait_days"],
                    "queue_rejections": m["queue_rejections"], "resources": usage, "stock": stock,
                    "costs": observed_charges(usage, ResourcePrices(), 2, 1.)})
    assert len(rows) == 24
    output = folder/"queue_evaluation_i6"
    output.mkdir(exist_ok=True)
    result = {"schema": "taris-queue-control-evaluation/1", "worlds": 1, "executions": 24,
        "input_hashes": audit["input_hashes"], "audit_hash": hashlib.sha256((source/"audit.json").read_bytes()).hexdigest(),
        "source_hashes": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
                          ("report_queue_control.py", "object_reserve.py", "resource_costs.py")},
        "rows": rows, "full_actuarial_reserve": None, "independent_holdout": False}
    (output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# I6 — очередь и цена восстановления", "", "Диагностический replay одного уже открытого мира103, close. Не независимая проверка.", "",
        "| Правило | Канал | Вариант | Хвост | Завершено proof | Возвращено в карту | Передача | Ниже n=4, объекто-сутки | Цена ресурсов |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["mode"]} | {row["profile"]} | {row["variant"]} | {row["unfinished_jobs"]} | '
            f'{row["confirmed_proofs"]} | {row["restored_map_entries"]} | {row["resources"]["served_volume"]:.6f} | '
            f'{row["stock"]["below_target_object_days"]:.6f} | {row["costs"]["observed_resource_cost"]:.6f} |')
    lines += ["", "Потерь0во всех режимах. Снятие блокировки повышает передачу и расходы, поэтому прогресс нельзя "
        "оценивать только уменьшением бюджета. Метрика физической доступности не равна готовности пользовательского чтения "
        "через текущую карту. Стоимость ущерба, будущего хвоста и полный резерв неизвестны."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("I6: очередь, возвращённые карты, физический запас и ресурсы сохранены")


if __name__ == "__main__":
    main()
