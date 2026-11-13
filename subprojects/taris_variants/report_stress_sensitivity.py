"""I5.1/2: описательные средние по мирам, условная стоимость с неизвестными входами."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import defaultdict
from dataclasses import asdict
import argparse
import hashlib
import json
from math import isclose
from subprojects.taris_variants.object_reserve import object_reserve
from subprojects.taris_variants.resource_costs import resource_usage, observed_charges, ResourcePrices
from subprojects.taris_variants.report_disk_stress import preloss_stock
from subprojects.taris_variants.stress_actuarial import MODES, conditional_cost, break_even_loss_price
from subprojects.taris_variants.wave_exposure import wave_exposure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--loss-price", type=float)
    args = parser.parse_args()
    folder = Path(__file__).parent
    directory = folder/"series_i51"
    audit_path = directory/"audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["worlds"] == 5 and audit["executions"] == 150
    weights = json.loads(args.weights.read_text(encoding="utf-8")) if args.weights else None
    rows, groups = [], defaultdict(list)
    for name, digest in audit["inputs"].items():
        path = directory/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        case = json.loads(path.read_text(encoding="utf-8"))["case"]
        cfg = case["config"]
        assert cfg["k"] == 2 and cfg["object_size"] == 1. and case["settings"]["n"] == 4
        for profile, variants in case["results"].items():
            for variant, record in variants.items():
                end = cfg["horizon_days"]
                stock = object_reserve(record, 2, 4, end)
                usage = resource_usage(record, end, 2, 1.)
                m = record["metrics"]
                wave_times = [w["time"] for w in case["stress_schedule"]]
                stage_counts = {"losses_at_first_wave": sum(r["t"] == 15. for r in record["losses"]) if wave_times else 0,
                    "losses_at_second_wave": sum(r["t"] == wave_times[1] for r in record["losses"]) if wave_times else 0}
                exposure = wave_exposure(record, case["stress_schedule"][0]["nodes"], 15., 2) if wave_times else None
                if exposure is not None:
                    assert set(exposure["immediately_vulnerable_objects"]) == {r["obj"] for r in record["losses"] if r["t"] == 15.}
                assert isclose(stock["physical_copy_days"], m["physical_fragment_days"], abs_tol=1e-8)
                assert isclose(stock["disk_deficit_object_days"], m["disk_deficit_object_days"], abs_tol=1e-8)
                assert isclose(usage["served_volume"], sum(m["serviced_volume"].values()), abs_tol=1e-8)
                row = {"seed": case["seed"], "mode": case["stress_mode"], "profile": profile, "variant": variant,
                    "objects": m["objects"], "lost_admitted_volume": m["lost_admitted_objects"]*cfg["object_size"],
                    "losses": m["irrecoverable_objects"], "initial_incomplete_objects": m["initial_incomplete_objects"],
                    "unavailable_object_days": m["unavailable_object_days"], "unfinished_jobs": m["unfinished_jobs"],
                    "stock": stock, "preloss_stock": preloss_stock(record, end), "resources": usage,
                    "first_wave_exposure": exposure, **stage_counts,
                    "costs": {str(ratio): observed_charges(usage, ResourcePrices(central_ratio=ratio), 2, 1.)
                              for ratio in (1., 10.)}}
                rows.append(row)
                groups[row["mode"], profile, variant].append(row)
    summaries = []
    for (mode, profile, variant), group in sorted(groups.items(), key=lambda item:
            (MODES.index(item[0][0]), item[0][1], item[0][2])):
        assert sorted(r["seed"] for r in group) == [101, 102, 103, 104, 105]
        summaries.append({"mode": mode, "profile": profile, "variant": variant, "worlds": 5,
            "mean_losses": sum(r["losses"] for r in group)/5,
            "min_losses": min(r["losses"] for r in group), "max_losses": max(r["losses"] for r in group),
            "mean_lost_admitted_volume": sum(r["lost_admitted_volume"] for r in group)/5,
            "mean_losses_at_first_wave": sum(r["losses_at_first_wave"] for r in group)/5,
            "mean_losses_at_second_wave": sum(r["losses_at_second_wave"] for r in group)/5,
            "mean_served_volume": sum(r["resources"]["served_volume"] for r in group)/5,
            "mean_preloss_bare_code_days": sum(r["preloss_stock"]["preloss_bare_code_days"] for r in group)/5,
            "mean_unfinished_jobs": sum(r["unfinished_jobs"] for r in group)/5,
            "mean_resource_cost": {str(ratio): sum(r["costs"][str(ratio)]["observed_resource_cost"] for r in group)/5
                                   for ratio in (1., 10.)}})
    scenario_sets, mixtures, crossings = {}, [], []
    for profile in ("ample", "constrained"):
        for variant in ("classic", "graph", "hybrid"):
            for ratio in (1., 10.):
                scenarios = [{"mode": row["mode"], "resource_cost": row["mean_resource_cost"][str(ratio)],
                              "lost_admitted_volume": row["mean_lost_admitted_volume"]}
                             for row in summaries if row["profile"] == profile and row["variant"] == variant]
                scenario_sets[profile, variant, ratio] = scenarios
                mixtures.append({"profile": profile, "variant": variant, "central_price_ratio": ratio,
                    **conditional_cost(scenarios, weights, args.loss_price)})
    for mode in MODES:
        for variant in ("classic", "graph", "hybrid"):
            point_weights = {name: float(name == mode) for name in MODES}
            crossings.append({"mode": mode, "variant": variant, "central_price_ratio": 1.,
                "comparison": "ample_vs_constrained_given_this_branch_no_infrastructure_premium",
                **break_even_loss_price(scenario_sets["ample", variant, 1.], scenario_sets["constrained", variant, 1.], point_weights)})
    lookup = {(r["seed"], r["mode"], r["profile"], r["variant"]): r for r in rows}
    contrasts = []
    for seed in range(101, 106):
        for variant in ("classic", "graph", "hybrid"):
            for mode in MODES:
                a, b = lookup[seed, mode, "ample", variant], lookup[seed, mode, "constrained", variant]
                contrasts.append({"seed": seed, "mode": mode, "variant": variant,
                    "constrained_minus_ample_lost_volume": b["lost_admitted_volume"]-a["lost_admitted_volume"]})
    output = folder/"stress_sensitivity_i51"
    output.mkdir(exist_ok=True)
    output_path = output/("conditional_weights.json" if args.weights or args.loss_price is not None else "results.json")
    prices = {str(ratio): asdict(ResourcePrices(central_ratio=ratio)) for ratio in (1., 10.)}
    result = {"schema": "taris-stress-sensitivity-evaluation/1", "worlds": 5, "executions": len(rows),
        "input_hashes": audit["inputs"], "audit_hash": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "source_hashes": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
            ("report_stress_sensitivity.py", "report_disk_stress.py", "object_reserve.py", "resource_costs.py", "stress_actuarial.py", "wave_exposure.py")},
        "rows": rows, "summary": summaries, "paired_channel_contrasts": contrasts,
        "supplied_weights": weights, "supplied_loss_price": args.loss_price, "resource_prices": prices,
        "conditional_mixtures": mixtures, "conditional_branch_crossings": crossings,
        "real_world_scenario_probabilities_estimated": False, "full_actuarial_reserve": None}
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# I5.1 — интервал волн и сохранность", "", "Пять внешних миров,150исполнений. Среднее и диапазон относятся к мирам, не к суточным строкам.", "",
        "| Ветвь | Канал | Вариант | Среднее потерь | Диапазон | Ровно k до потери, объекто-сутки | Средний хвост | Средняя цена ресурсов RAM×1 |",
        "|---|---|---|---:|---|---:|---:|---:|"]
    for row in summaries:
        lines.append(f'| {row["mode"]} | {row["profile"]} | {row["variant"]} | {row["mean_losses"]:.2f} | '
            f'{row["min_losses"]}–{row["max_losses"]} | {row["mean_preloss_bare_code_days"]:.4f} | '
            f'{row["mean_unfinished_jobs"]:.2f} | {row["mean_resource_cost"]["1.0"]:.4f} |')
    lines += ["", "Цена ресурсов не включает ущерб и инфраструктурную премию быстрого канала. Реальные вероятности "
        "кризисов и цена информации не оценены. По умолчанию условная денежная смесь неизвестна; равные веса не подставляются. "
        "Точки равенства по цене информации в JSON условны для каждой ветви и не являются рекомендацией."]
    if output_path.name == "results.json":
        (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("I5.1/2: сводка по пяти мирам и условные актуарные компоненты сохранены")


if __name__ == "__main__":
    main()
