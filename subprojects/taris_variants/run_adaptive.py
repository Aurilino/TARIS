"""I3: фиксированные и адаптивные размеры при общих проверках требований."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import warnings
import traceback
from time import perf_counter
import numpy as np

from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from subprojects.taris_variants.agents import VariantSettings
from subprojects.taris_variants.adaptive import AdaptiveAgent, SizingRequirements
from subprojects.taris_variants.run_compare import exposure_metrics, table_hash, source_hashes


def run_case(seed, profile, days=45, nodes=24, scenario_name="S2"):
    bounds = {"fixed3": (3, 3), "fixed4": (4, 4), "adaptive": (3, 6)}
    if profile not in bounds:
        raise ValueError("Неизвестный профиль")
    low, high = bounds[profile]
    requirements = SizingRequirements(min_n=low, max_n=high)
    settings = VariantSettings(n=high, block_days=7, reserve_horizon_days=30)
    cfg = scenario(scenario_name, seed=seed, horizon_days=days, train_days=days // 2,
        n_initial_nodes=nodes, node_join_rate=0.1, n_dc=8, n_providers=4,
        n_regions=3, blocks_per_epoch=1, k=2)
    world = World(cfg)
    hashes = {"nodes": table_hash(world.nodes), "outages": table_hash(world.outages), "silent": table_hash(world.silent)}
    variants, fingerprints = {}, []
    for variant in ("classic", "graph", "hybrid"):
        chain, data = Chain(world), DataStream(cfg, len(world.nodes))
        fingerprints.append({"observed": table_hash(chain.observed), "audit": table_hash(chain.audits),
            "objects": table_hash(data.objects), "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()})
        agent = AdaptiveAgent(variant, world, settings, requirements)
        started = perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            result = Ledger(world, chain, data).run(agent)
        if not np.isfinite(agent.w).all() or agent.model.fitted and not all(
                np.isfinite(getattr(agent.model, name)).all() for name in ("beta", "gamma")):
            raise ArithmeticError("Неконечная модель в адаптивном режиме")
        decisions = agent.placement_log
        chosen_sizes = [d["n"] for d in decisions if d["nodes"]]
        initial = result.fragments[result.fragments.t_from == result.fragments.obj.map(result.objects.created_epoch)]
        actual = {int(obj): list(map(int, rows.node)) for obj, rows in initial.groupby("obj")}
        if any(actual.get(d["obj"], []) != d["nodes"] for d in decisions):
            raise AssertionError("Выбранный размер не совпадает с исполненным размещением")
        rejected = {d["obj"] for d in decisions if not d["nodes"]}
        variants[variant] = {
            "objects_requested": len(result.objects), "objects_admitted": len(chosen_sizes),
            "initial_rejected_objects": len(rejected),
            "lost_objects_ledger": len(result.losses),
            "lost_admitted_objects": sum(int(obj) not in rejected for obj in result.losses.obj),
            "size_counts": dict(sorted(Counter(chosen_sizes).items())),
            "mean_initial_n": float(np.mean(chosen_sizes)) if chosen_sizes else None,
            "status_counts": dict(agent.status_counts), "placement_risk": agent.placement_risk,
            "requirements_passed_objects": sum(d["requirements_passed"] for d in decisions),
            "scored_objects": sum(d["scored"] for d in decisions),
            "repair_operations": len(result.repairs), "repair_traffic": float(result.repairs.traffic.sum()),
            "fragment_days": float(result.daily.fragments.sum()),
            "full_exposure": exposure_metrics(result, days),
            "post_warmup_cohort": exposure_metrics(result, days, cfg.train_days),
            "runtime_warnings": dict(Counter(str(w.message) for w in captured)),
            "seconds": perf_counter() - started, "placement_decisions": decisions,
            "repair_requests": agent.repair_request_log,
            "repair_best_effort_requests": sum(not r["requirements_passed"] for r in agent.repair_request_log),
            "final_sizes": list(map(int, result.objects.n)),
        }
        mean_n = variants[variant]["mean_initial_n"]
        size_text = f"{mean_n:.2f}" if mean_n is not None else "нет приёма"
        print(f"s{seed} {profile} {variant}: n={size_text}, "
              f"статусы={variants[variant]['status_counts']}", flush=True)
    unchanged = hashes == {"nodes": table_hash(world.nodes), "outages": table_hash(world.outages), "silent": table_hash(world.silent)}
    return {"seed": seed, "profile": profile, "config": json.loads(cfg.to_json()),
        "settings": asdict(settings), "requirements": asdict(requirements),
        "world_hashes": hashes, "input_hashes": fingerprints[0], "world_unchanged": unchanged,
        "common_random_numbers": all(f == fingerprints[0] for f in fingerprints), "variants": variants,
        "realized_shocks": len(world.shocks),
        "realized_destructive_shocks": int(world.shocks["destroyed"].sum()) if len(world.shocks) else 0}


def summarize(cases):
    summary = []
    for profile in sorted({c["profile"] for c in cases}):
        rows = [c for c in cases if c["profile"] == profile]
        for variant in ("classic", "graph", "hybrid"):
            values = [r["variants"][variant] for r in rows]
            statuses = Counter()
            for value in values:
                statuses.update(value["status_counts"])
            exposure = sum(v["full_exposure"]["object_days"] for v in values)
            unavailable = sum(v["full_exposure"]["unavailable_object_days"] for v in values)
            total_admitted = sum(v["objects_admitted"] for v in values)
            summary.append({"profile": profile, "variant": variant, "cases": len(rows),
                "objects_requested": sum(v["objects_requested"] for v in values),
                "objects_admitted": total_admitted,
                "mean_n": sum(v["mean_initial_n"] * v["objects_admitted"] for v in values
                              if v["mean_initial_n"] is not None) / total_admitted if total_admitted else None,
                "passed": sum(v["requirements_passed_objects"] for v in values),
                "scored": sum(v["scored_objects"] for v in values), "statuses": dict(statuses),
                "lost_admitted": sum(v["lost_admitted_objects"] for v in values),
                "rejected": sum(v["initial_rejected_objects"] for v in values),
                "fragment_days": sum(v["fragment_days"] for v in values),
                "repair_traffic": sum(v["repair_traffic"] for v in values),
                "unavailability_share": unavailable / exposure if exposure else None})
    return summary


def render(report):
    lines = ["# I3. Выбор начального числа чанков", "", "Дата: 10 ноября 2026 года.", "",
        "Общие проверки потери за окно и недоступности. Фиксированные профили используют тот же новый контроллер, что адаптивный; это не переопределение результатов I2.", "",
        "| Профиль | Вариант | Среднее n | Прошли / оценены | Без оценки | Не найдено, best effort | Потери принятых объектов | Фрагменто-сутки | Трафик |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        mean_n = f"{row['mean_n']:.2f}" if row["mean_n"] is not None else "—"
        lines.append(f"| {row['profile']} | {row['variant']} | {mean_n} | {row['passed']} / {row['scored']} | "
            f"{row['statuses'].get('unscored_cold_start', 0)} | {row['statuses'].get('no_feasible_found_best_effort', 0)} | "
            f"{row['lost_admitted']} | {row['fragment_days']:.0f} | {row['repair_traffic']:.2f} |")
    lines += ["", f"Изоляция и общая случайность: {report['integrity_passed']}. Ошибок: {len(report['failures'])}.",
        f"Миры совпадают между профилями: {report['paired_profiles']}. Различных миров: {report['distinct_worlds']}.",
        "", "Cold start не считается выполненным требованием. Best effort тоже не считается выполненной защитой. Отказ приёма учитывается отдельно: исходный Ledger записывает пустое начальное размещение как потерю, поэтому отдельно считаются потери уже принятых объектов.",
        "", "Минимальность числа чанков не доказана: это первый размер с допустимым предложением в ограниченном поиске. Фиксированные профили не расширяют запас при нарушении; адаптивный может использовать от трёх до шести чанков.",
        "", "Все три адаптивных варианта используют одинаковый тип актуарного датчика для общей проверки потери; G не использует Тиле и не выбирает момент ремонта по резерву. Classic дополнительно проверяет доступность общим модулем, но не ранжирует ноды по графовому критерию. Это новые явно обозначенные sizing-профили.",
        "", "Ограничение потери относится непосредственно к окну T_rep + repair_hours, не является линейно пересчитанной гарантией на 30 суток. Буфер, новая физическая семантика и консервативный ансамбль ещё не перенесены.",
        "", "Размер выбирается при создании объекта. Последующее уменьшение/увеличение кода существующего объекта пока не включено; ремонт восстанавливает выбранный размер. Сравнение малого числа миров не доказывает общую эффективность.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="I3: начальный размер кода по требованиям")
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--seeds", type=int, nargs="+", default=[6, 7, 8])
    parser.add_argument("--profiles", nargs="+", choices=["fixed3", "fixed4", "adaptive"], default=["fixed3", "fixed4", "adaptive"])
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("series_i3"))
    args = parser.parse_args()
    if args.days < 8 or args.nodes < 8 or any(s < 0 for s in args.seeds):
        parser.error("Некорректные параметры")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    core = Path(__file__).resolve().parent
    provenance = {"main": source_hashes(), "integration": {name: hashlib.sha256((core / name).read_bytes()).hexdigest()
        for name in ("adaptive.py", "agents.py", "risk_graph.py", "run_compare.py", "run_adaptive.py")}}
    cases, failures = [], []
    for seed in args.seeds:
        for profile in args.profiles:
            job = {"seed": seed, "profile": profile, "days": args.days, "nodes": args.nodes}
            path = args.output_dir / f"s{seed}_{profile}.json"
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["job"] != job or saved["provenance"] != provenance:
                    raise ValueError("Несовместимый кэш: " + str(path))
            else:
                try:
                    saved = {"status": "completed", "job": job, "provenance": provenance,
                             "case": run_case(seed, profile, args.days, args.nodes)}
                except Exception as exc:
                    saved = {"status": "failed", "job": job, "provenance": provenance,
                             "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                path.write_text(json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            if saved["status"] == "completed":
                cases.append(saved["case"])
            else:
                failures.append({"job": job, "error": saved["error"]})
                print("Ошибка:", failures[-1], flush=True)
    fingerprints = {}
    for case in cases:
        fingerprint = json.dumps({"world": case["world_hashes"], "inputs": case["input_hashes"]}, sort_keys=True)
        fingerprints.setdefault(case["seed"], set()).add(fingerprint)
    paired = all(len(values) == 1 for values in fingerprints.values())
    integration_after = {name: hashlib.sha256((core / name).read_bytes()).hexdigest()
                         for name in provenance["integration"]}
    integrity = provenance == {"main": source_hashes(), "integration": integration_after}
    integrity = integrity and paired and all(c["world_unchanged"] and c["common_random_numbers"] for c in cases)
    report = {"schema": "taris-sizing-i3/1", "completed_cases": len(cases), "failures": failures,
        "provenance": provenance, "summary": summarize(cases), "integrity_passed": integrity,
        "paired_profiles": paired, "distinct_worlds": len({f for values in fingerprints.values() for f in values}),
        "shock_counts": [{"seed": c["seed"], "count": c["realized_shocks"], "destructive": c["realized_destructive_shocks"]}
                         for c in cases if c["profile"] == args.profiles[0]]}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "RESULTS.md").write_text(render(report), encoding="utf-8")
    print("Сводка:", (args.output_dir / "RESULTS.md").resolve())
    if failures or not integrity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
