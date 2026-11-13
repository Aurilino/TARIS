"""I5: физическая сохранность и фактические ресурсы, без цены неизвестного ущерба."""
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
from subprojects.taris_variants.audit_physical import rank_timeline


def preloss_stock(record, end, k=2, target_n=4):
    losses = {r["obj"]: r["t"] for r in record["losses"]}
    copies = defaultdict(list)
    for row in record["physical_fragments"]:
        copies[row["obj"]].append(row)
    totals = {"preloss_object_days": 0., "preloss_below_target_days": 0.,
              "preloss_bare_code_days": 0., "preloss_disk_deficit_days": 0.}
    for obj, created, *_ in record["initial_writes"]:
        stop = min(end, losses.get(obj, end))
        intervals = defaultdict(list)
        for row in copies[obj]:
            a = row["start"]
            b = min(stop, row["physical_end"] if row["physical_end"] is not None else stop)
            if a < b:
                intervals[row["idx"]].append((a, b))
        for a, b, rank in rank_timeline(intervals, created, stop):
            totals["preloss_object_days"] += b-a
            totals["preloss_below_target_days"] += (b-a)*(rank < target_n)
            totals["preloss_bare_code_days"] += (b-a)*(rank == k)
            totals["preloss_disk_deficit_days"] += (b-a)*(rank < k)
    return totals


def main():
    folder = Path(__file__).parent
    directory = folder/"series_i5_pilot"
    audit = json.loads((directory/"audit.json").read_text(encoding="utf-8"))
    rows = []
    for name, digest in audit["inputs"].items():
        path = directory/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        end, k = case["config"]["horizon_days"], case["config"]["k"]
        assert case["config"]["object_size"] == 1. and k == 2 and case["settings"]["n"] == 4
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                stock = object_reserve(record, k, 4, end)
                usage = resource_usage(record, end, k, 1.)
                m = record["metrics"]
                assert isclose(stock["physical_copy_days"], m["physical_fragment_days"], abs_tol=1e-8)
                assert isclose(stock["disk_deficit_object_days"], m["disk_deficit_object_days"], abs_tol=1e-8)
                assert isclose(stock["buffer_only_object_days"], m["buffer_only_object_days"], abs_tol=1e-8)
                assert isclose(usage["served_volume"], sum(m["serviced_volume"].values()), abs_tol=1e-8)
                losses = record["losses"]
                waves = case["stress_schedule"]
                at_first = sum(r["t"] == 15. for r in losses)
                at_second = sum(r["t"] == waves[1]["time"] for r in losses) if waves else 0
                rows.append({"seed": case["seed"], "mode": case["stress_mode"], "profile": profile,
                    "variant": variant, "objects": m["objects"], "losses": m["irrecoverable_objects"],
                    "lost_admitted_objects": m["lost_admitted_objects"],
                    "initial_incomplete_objects": m["initial_incomplete_objects"],
                    "losses_at_first_wave": at_first, "losses_at_second_wave": at_second,
                    "losses_other_times": len(losses)-at_first-at_second,
                    "unfinished_jobs": m["unfinished_jobs"], "stock": stock,
                    "preloss_stock": preloss_stock(record, end, k), "resources": usage,
                    "costs": {str(ratio): observed_charges(usage, ResourcePrices(central_ratio=ratio), k, 1.)
                              for ratio in (1., 10.)}})
    lookup = {(r["seed"], r["mode"], r["profile"], r["variant"]): r for r in rows}
    contrasts = []
    for seed in sorted({r["seed"] for r in rows}):
        for profile in ("ample", "constrained"):
            for variant in ("classic", "graph", "hybrid"):
                close = lookup[seed, "close", profile, variant]
                far = lookup[seed, "far", profile, variant]
                contrasts.append({"seed": seed, "profile": profile, "variant": variant,
                    "far_minus_close_losses": far["losses"]-close["losses"],
                    "far_minus_close_preloss_bare_code_days": far["preloss_stock"]["preloss_bare_code_days"]
                        -close["preloss_stock"]["preloss_bare_code_days"]})
    output = folder/"stress_evaluation_i5"
    output.mkdir(exist_ok=True)
    result = {"schema": "taris-disk-stress-evaluation/1", "worlds": audit["worlds"],
        "input_hashes": audit["inputs"], "audit_hash": hashlib.sha256((directory/"audit.json").read_bytes()).hexdigest(),
        "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows,
        "paired_contrasts": contrasts, "expected_loss_cost": None, "full_actuarial_reserve": None}
    (output/"results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# I5 — стресс-пилот сохранности", "", "Один внешний мир, три парные ветви. Не оценка вероятности катастрофы.", "",
        "| Ветвь | Канал | Вариант | Потерь | На первой волне | На второй | До потери: ровно k, объекто-сутки | Только буфер, объекто-сутки | Передача | Хвост | Цена ресурсов RAM×1 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f'| {row["mode"]} | {row["profile"]} | {row["variant"]} | {row["losses"]} | '
            f'{row["losses_at_first_wave"]} | {row["losses_at_second_wave"]} | '
            f'{row["preloss_stock"]["preloss_bare_code_days"]:.6f} | {row["stock"]["buffer_only_object_days"]:.6f} | '
            f'{row["resources"]["served_volume"]:.6f} | {row["unfinished_jobs"]} | '
            f'{row["costs"]["1.0"]["observed_resource_cost"]:.6f} |')
    lines += ["", "До потери — экспозиция только до первого безвозвратного исхода объекта. Полный запас в JSON включает "
        "также время после потери и не смешивается с этой экспозицией. Наблюдённая цена ресурсов не включает ущерб: "
        "потеря копий уменьшает хранение, поэтому меньшая цена сама по себе не означает эффективность. "
        "Полный будущий ущерб и резерв неизвестны. Немедленное обнаружение дисковой ошибки — допущение пилота."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("I5: объектовый запас, потери и ресурсы сохранены")


if __name__ == "__main__":
    main()
