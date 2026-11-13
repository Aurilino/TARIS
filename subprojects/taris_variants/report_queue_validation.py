"""I6.2: описательное сравнение зафиксированных правил на новых мирах."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
import hashlib
import json
from math import isclose
from subprojects.taris_variants.object_reserve import object_reserve
from subprojects.taris_variants.resource_costs import resource_usage, observed_charges, ResourcePrices
from subprojects.taris_variants.report_disk_stress import preloss_stock


def main():
    folder = Path(__file__).parent
    directory = folder/"series_i62"
    audit_path = directory/"audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["executions"] == 108 and audit["worlds"] == 3 and audit["frozen_source_subset_verified"]
    rows, groups = [], defaultdict(list)
    for name, digest in audit["input_hashes"].items():
        path = directory/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        assert case["config"]["k"] == 2 and case["config"]["object_size"] == 1. and case["settings"]["n"] == 4
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                m = record["metrics"]
                usage = resource_usage(record, 45., 2, 1.)
                stock = object_reserve(record, 2, 4, 45.)
                assert isclose(usage["served_volume"], sum(m["serviced_volume"].values()), abs_tol=1e-8)
                assert isclose(stock["physical_copy_days"], m["physical_fragment_days"], abs_tol=1e-8)
                row = {"seed": case["seed"], "stress_mode": case["stress_mode"], "mode": case["drain_mode"],
                    "profile": profile, "variant": variant, "objects": m["objects"],
                    "losses": m["irrecoverable_objects"], "lost_admitted_volume": m["lost_admitted_objects"],
                    "initial_incomplete_objects": m["initial_incomplete_objects"],
                    "unfinished_jobs": m["unfinished_jobs"], "job_statuses": m["job_statuses"],
                    "confirmed_proofs": sum(j["status"] == "proof_completed" for j in record["jobs"]),
                    "restored_map_entries": len(record["returns"]),
                    "waiting_read_jobs_at_end": sum(j["finished"] is None and j["status"] == "waiting_read_sources"
                                                     for j in record["jobs"]),
                    "pressure_fallback_starts": m["pressure_fallback_starts"],
                    "queue_rejections": m["queue_rejections"], "resources": usage, "stock": stock,
                    "preloss_stock": preloss_stock(record, 45.),
                    "costs": observed_charges(usage, ResourcePrices(), 2, 1.)}
                rows.append(row)
                groups[row["stress_mode"], row["mode"], profile, variant].append(row)
    assert len(rows) == 108 and len(groups) == 36
    summaries = []
    for (stress, mode, profile, variant), group in sorted(groups.items()):
        assert sorted(r["seed"] for r in group) == [201, 202, 203]
        summaries.append({"stress_mode": stress, "mode": mode, "profile": profile, "variant": variant,
            "worlds": 3, "mean_losses": sum(r["losses"] for r in group)/3,
            "mean_lost_admitted_volume": sum(r["lost_admitted_volume"] for r in group)/3,
            "loss_range": [min(r["losses"] for r in group), max(r["losses"] for r in group)],
            "mean_unfinished_jobs": sum(r["unfinished_jobs"] for r in group)/3,
            "mean_restored_map_entries": sum(r["restored_map_entries"] for r in group)/3,
            "mean_served_volume": sum(r["resources"]["served_volume"] for r in group)/3,
            "mean_resource_cost": sum(r["costs"]["observed_resource_cost"] for r in group)/3,
            "mean_below_target_days": sum(r["stock"]["below_target_object_days"] for r in group)/3,
            "mean_preloss_below_target_days": sum(r["preloss_stock"]["preloss_below_target_days"] for r in group)/3,
            "total_pressure_fallback_starts": sum(r["pressure_fallback_starts"] for r in group),
            "total_waiting_read_jobs_at_end": sum(r["waiting_read_jobs_at_end"] for r in group)})
    lookup = {(r["seed"], r["stress_mode"], r["mode"], r["profile"], r["variant"]): r for r in rows}
    contrasts = []
    for seed in (201, 202, 203):
        for stress in ("close", "far"):
            for profile in ("ample", "constrained"):
                for variant in ("classic", "graph", "hybrid"):
                    current = lookup[seed, stress, "pressure", profile, variant]
                    for baseline in ("legacy", "combined"):
                        previous = lookup[seed, stress, baseline, profile, variant]
                        contrasts.append({"seed": seed, "stress_mode": stress, "profile": profile, "variant": variant,
                            "comparison": "pressure_minus_"+baseline,
                            "loss_difference": current["losses"]-previous["losses"],
                            "unfinished_difference": current["unfinished_jobs"]-previous["unfinished_jobs"],
                            "below_target_difference": current["stock"]["below_target_object_days"]-previous["stock"]["below_target_object_days"],
                            "resource_cost_difference": current["costs"]["observed_resource_cost"]-previous["costs"]["observed_resource_cost"]})
    output = folder/"queue_validation_i62"
    output.mkdir(exist_ok=True)
    result = {"schema": "taris-queue-validation-evaluation/1", "worlds": 3, "executions": 108,
        "input_hashes": audit["input_hashes"], "audit_hash": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "source_hashes": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
                          ("report_queue_validation.py", "object_reserve.py", "resource_costs.py", "report_disk_stress.py")},
        "rows": rows, "summary": summaries, "paired_contrasts": contrasts,
        "expected_loss_cost": None, "full_actuarial_reserve": None, "independent_holdout": True}
    (output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# I6.2 — независимая проверка очереди", "", "Три новых мира201–203,108исполнений. Средние и диапазоны по мирам, не по суточным строкам.", "",
        "| Волны | Правило | Канал | Агент | Потери: среднее [диапазон] | Средний хвост | Ниже4 до потери, объекто-сутки | Средняя цена ресурсов |",
        "|---|---|---|---|---:|---:|---:|---:|"]
    for row in summaries:
        lo, hi = row["loss_range"]
        lines.append(f'| {row["stress_mode"]} | {row["mode"]} | {row["profile"]} | {row["variant"]} | '
            f'{row["mean_losses"]:.3f} [{lo}–{hi}] | {row["mean_unfinished_jobs"]:.3f} | '
            f'{row["mean_preloss_below_target_days"]:.4f} | {row["mean_resource_cost"]:.4f} |')
    lines += ["", "Цена не включает ущерб, инфраструктурную премию и будущий хвост. Ожидаемый ущерб и полный резерв неизвестны. "
        "Все индивидуальные миры и парные различия сохранены в JSON; знак улучшения очереди не заменяет оценку сохранности."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("I6.2: индивидуальные миры,36сводок и72парных различия сохранены")


if __name__ == "__main__":
    main()
