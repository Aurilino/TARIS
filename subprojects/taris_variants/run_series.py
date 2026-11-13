"""I2: возобновляемая парная серия бюджетов и режимов восстановления."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
from dataclasses import asdict
import hashlib
import json
from statistics import mean, median
import traceback

from subprojects.taris_variants.agents import VariantSettings
from subprojects.taris_variants.run_compare import run_case, source_hashes


def core_hashes():
    folder = Path(__file__).resolve().parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ("agents.py", "risk_graph.py", "run_compare.py")}


def fingerprint(case):
    return hashlib.sha256(json.dumps({"world": case["world_hashes"], "input": case["input_hashes"]},
                                     sort_keys=True).encode()).hexdigest()


def summarize(cases):
    grouped = {}
    for case in cases:
        group = (case["settings"]["n"], case["settings"]["repair_mode"])
        grouped.setdefault(group, []).append(case)
    result = []
    for (n, mode), rows in sorted(grouped.items()):
        variants = {}
        for kind in ("classic", "graph", "hybrid"):
            values = [r["variants"][kind] for r in rows]
            caps = [v["placement_risk"] for v in values if v.get("placement_risk") is not None]
            exposure = sum(v["full_exposure"]["object_days"] for v in values)
            unavailable = sum(v["full_exposure"]["unavailable_object_days"] for v in values)
            test_exposure = sum(v["post_warmup_cohort"]["object_days"] for v in values)
            test_unavailable = sum(v["post_warmup_cohort"]["unavailable_object_days"] for v in values)
            variants[kind] = {
                "runs": len(values), "objects": sum(v["objects"] for v in values),
                "lost_objects": sum(v["lost_objects"] for v in values),
                "repair_traffic": sum(v["repair_traffic"] for v in values),
                "fragment_days": sum(v["fragment_days"] for v in values),
                "object_days": exposure, "unavailable_object_days": unavailable,
                "unavailability_share": unavailable / exposure if exposure else None,
                "post_warmup_unavailability_share": test_unavailable / test_exposure if test_exposure else None,
                "runtime_warning_count": sum(sum(v["runtime_warnings"].values()) for v in values),
                "mean_seconds": mean(v["seconds"] for v in values),
                "actuarial_checked": sum(c["checked"] for c in caps) if caps else None,
                "actuarial_violations": sum(c["violations"] for c in caps) if caps else None,
                "actuarial_unscored": sum(c["unscored"] for c in caps) if caps else None,
                "short_placements": sum(v.get("short_placements", 0) for v in values),
                "graph_placement_checks": sum(len(v.get("placement_decisions", [])) for v in values),
                "graph_availability_exceeded": sum(
                    row["estimated_unavailability"] > case["settings"].get("availability_limit", 0.08)
                    for case in rows for row in case["variants"][kind].get("placement_decisions", [])),
            }
        paired = {}
        for kind in ("graph", "hybrid"):
            metrics = {}
            for metric in ("lost_objects", "repair_traffic", "fragment_days"):
                deltas = [r["variants"][kind][metric] - r["variants"]["classic"][metric] for r in rows]
                metrics[metric] = {"mean_delta": mean(deltas), "median_delta": median(deltas),
                    "min_delta": min(deltas), "max_delta": max(deltas),
                    "lower": sum(x < -1e-12 for x in deltas), "equal": sum(abs(x) <= 1e-12 for x in deltas),
                    "higher": sum(x > 1e-12 for x in deltas)}
            deltas = [r["variants"][kind]["full_exposure"]["unavailability_share"]
                      - r["variants"]["classic"]["full_exposure"]["unavailability_share"] for r in rows]
            metrics["unavailability_share"] = {"mean_delta": mean(deltas),
                "lower": sum(x < -1e-12 for x in deltas), "equal": sum(abs(x) <= 1e-12 for x in deltas),
                "higher": sum(x > 1e-12 for x in deltas)}
            paired[kind] = metrics
        result.append({"n": n, "repair_mode": mode, "cases": len(rows),
                       "distinct_worlds": len({fingerprint(r) for r in rows}),
                       "variants": variants, "paired_vs_classic": paired})
    return result


def ablations(cases):
    groups = {}
    for case in cases:
        key = (case["scenario"], case["seed"], case["settings"]["n"])
        groups.setdefault(key, {})[case["settings"]["repair_mode"]] = case
    result = []
    for key, modes in sorted(groups.items()):
        if set(modes) != {"fixed", "thiele"}:
            continue
        a, b = modes["fixed"], modes["thiele"]
        if fingerprint(a) != fingerprint(b):
            raise AssertionError("Режимы восстановления получили разные миры")
        entries = {}
        for kind in ("classic", "graph", "hybrid"):
            av, bv = a["variants"][kind], b["variants"][kind]
            entries[kind] = {metric + "_thiele_minus_fixed": bv[metric] - av[metric]
                             for metric in ("lost_objects", "repair_traffic", "fragment_days")}
            entries[kind]["unavailability_thiele_minus_fixed"] = (
                bv["full_exposure"]["unavailability_share"] - av["full_exposure"]["unavailability_share"])
            if kind == "graph":
                entries[kind]["control_identical"] = (av["initial_placements"] == bv["initial_placements"]
                    and all(abs(value) < 1e-12 for value in entries[kind].values()))
        result.append({"scenario": key[0], "seed": key[1], "n": key[2], "variants": entries})
    return result


def render(report):
    lines = ["# I2. Парное сравнение трёх вариантов", "", "Дата: 10 ноября 2026 года.", "",
        f"Завершено случаев: {report['completed_cases']}; неуспешных: {len(report['failures'])}. "
        f"Исполнений политик: {report['completed_cases'] * 3}. Различных миров событий: {report['distinct_worlds']}.", "",
        "Один случай — сценарий, зерно, бюджет и режим. Повторы одного мира при разных бюджетах, режимах и отсутствии шоков не являются независимыми мирами.", "",
        "| n/k | Режим | Вариант | Потери | Трафик | Фрагменто-сутки* | Недоступность за весь горизонт | После разогрева** |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for group in report["summary"]:
        for kind, v in group["variants"].items():
            lines.append(f"| {group['n']}/{report['k']} | {group['repair_mode']} | {kind} | {v['lost_objects']} | "
                f"{v['repair_traffic']:.2f} | {v['fragment_days']:.0f} | {v['unavailability_share']:.5%} | "
                f"{v['post_warmup_unavailability_share']:.5%} |")
    lines += ["", "* Сумма суточных чисел хранимых фрагментов — дискретная оценка объёма хранения во времени.",
        "", "** Объекты, созданные во второй половине прогона; обучение остаётся последовательным по прошлым данным.",
        "", "Недоступность учитывает весь срок от создания до конца прогона: после потери объект считается недоступным. Отдельно сохраняются прежняя метрика до потери и объекто-сутки потерь.",
        "", "## Соблюдение оценочных ограничений", "",
        "| n/k | Режим | Вариант | Актуарно проверено | Нарушений актуарного cap | Без оценки до обучения |",
        "|---|---|---|---:|---:|---:|"]
    for group in report["summary"]:
        for kind in ("classic", "hybrid"):
            v = group["variants"][kind]
            lines.append(f"| {group['n']}/{report['k']} | {group['repair_mode']} | {kind} | "
                         f"{v['actuarial_checked']} | {v['actuarial_violations']} | {v['actuarial_unscored']} |")
    lines += ["", "TARIS-G не имеет актуарной проверки потери; отсутствие такого счётчика не означает нулевой риск. "
        "Холодный старт учитывается отдельно от прошедших проверку. При недостаточном фиксированном бюджете исходный agent_capped допускает размещение с отметкой нарушения; нулевые реализованные потери не отменяют нарушение требования.",
        "", "## Контроль режимов", "",
        f"Пар режимов: {len(report['ablations'])}. "
        f"Контроль TARIS-G неизменен: {all(a['variants']['graph']['control_identical'] for a in report['ablations'])}.",
        f"Исходники и общая случайность проверены: {report['integrity_passed']}.",
        "", "## Реализовавшиеся шоки", ""]
    for row in report["shock_counts"]:
        lines.append(f"- {row['scenario']} s{row['seed']}: шоков {row['shocks']}, с уничтожением {row['destructive']}.")
    lines += ["", "## Границы вывода", "",
        "Числа относятся к выбранной короткой серии S0/S2. Библиотека кризисов S6 и прогноз не включены. При одинаковом зерне S0 и S2 могут совпасть, если общих шоков не было.",
        "", "Начальный бюджет одинаков внутри каждой группы, но фактические затраты хранения и ремонта могут различаться. Строгое сравнение при одинаковых фрагменто-сутках пока не выполнено.",
        "", "Метрика потерь использует текущий Ledger, включая административное признание после длительного простоя. Это не чистая оценка физического уничтожения информации.",
        "", "Нулевые потери не доказывают превосходства. Численные предупреждения GLM и конечность коэффициентов сохранены по каждому прогону. Победитель заранее не назначается; статистическая значимость на малом числе различных миров не заявляется.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="I2: бюджеты и абляция момента ремонта")
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--budgets", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("series_i2"))
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.days < 8 or args.nodes < 8 or any(n <= 2 for n in args.budgets) or any(s < 0 for s in args.seeds):
        parser.error("Некорректный горизонт, размер сети, бюджет или зерно")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"main": source_hashes(), "integration": core_hashes()}
    cases, failures = [], []
    for scenario in ("S0", "S2"):
        for seed in args.seeds:
            for n in args.budgets:
                for mode in ("fixed", "thiele"):
                    job = {"scenario": scenario, "seed": seed, "days": args.days, "nodes": args.nodes,
                           "n": n, "k": 2, "repair_mode": mode}
                    path = args.output_dir / f"{scenario}_s{seed}_n{n}_{mode}.json"
                    saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
                    if saved and (saved["job"] != job or saved["provenance"] != provenance):
                        raise ValueError(f"Несовместимый кэш: {path}; задайте новую папку результатов")
                    if saved is None or saved["status"] == "failed" and args.retry_failed:
                        try:
                            settings = VariantSettings(n=n, block_days=7, reserve_horizon_days=30, repair_mode=mode)
                            case = run_case(scenario, seed, args.days, args.nodes, settings)
                            saved = {"status": "completed", "job": job, "provenance": provenance, "case": case}
                        except Exception as exc:
                            saved = {"status": "failed", "job": job, "provenance": provenance,
                                     "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                        path.write_text(json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                    if saved["status"] == "completed":
                        cases.append(saved["case"])
                    else:
                        failures.append({"job": job, "error": saved["error"]})
                    print(f"Случаев завершено: {len(cases)}, ошибок: {len(failures)}", flush=True)
    integrity = provenance == {"main": source_hashes(), "integration": core_hashes()}
    integrity = integrity and all(c["common_random_numbers"] and c["world_unchanged"] for c in cases)
    shocks = {(c["scenario"], c["seed"]): {"scenario": c["scenario"], "seed": c["seed"],
              "shocks": c["realized_shocks"], "destructive": c["realized_destructive_shocks"]} for c in cases}
    report = {"schema": "taris-series-i2/1", "k": 2, "days": args.days,
        "seeds": args.seeds, "budgets": args.budgets, "completed_cases": len(cases),
        "distinct_worlds": len({fingerprint(c) for c in cases}), "integrity_passed": integrity,
        "provenance": provenance, "failures": failures, "shock_counts": list(shocks.values()),
        "summary": summarize(cases), "ablations": ablations(cases),
        "case_files": sorted(p.name for p in args.output_dir.glob("S*_s*_n*_*.json"))}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "RESULTS.md").write_text(render(report), encoding="utf-8")
    print("Сводка:", (args.output_dir / "RESULTS.md").resolve())
    if failures or not integrity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
