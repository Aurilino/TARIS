"""I3.2: новые миры, защитный слой и фиксированные контрольные размеры."""
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
import os
import traceback
import warnings
from time import perf_counter
import numpy as np

from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from subprojects.taris_variants.agents import VariantSettings
from subprojects.taris_variants.adaptive import AdaptiveAgent, SizingRequirements
from subprojects.taris_variants.conservative import ConservativeAgent, GuardSettings
from subprojects.taris_variants.run_compare import exposure_metrics, source_hashes, table_hash
from subprojects.taris_variants.run_adaptive import summarize


def run_case(seed, profile, days=45, nodes=24):
    ranges = {"point": (3, 6), "guard": (3, 6), "fixed4": (4, 4), "fixed6": (6, 6)}
    low, high = ranges[profile]
    requirements = SizingRequirements(min_n=low, max_n=high)
    settings = VariantSettings(n=high, block_days=7, reserve_horizon_days=30)
    guard_settings = GuardSettings()
    cfg = scenario("S2", seed=seed, horizon_days=days, train_days=days // 2,
        n_initial_nodes=nodes, node_join_rate=0.1, n_dc=8, n_providers=4,
        n_regions=3, blocks_per_epoch=1, k=2)
    world = World(cfg)
    hashes = {"nodes": table_hash(world.nodes), "outages": table_hash(world.outages), "silent": table_hash(world.silent)}
    fingerprints, results = [], {}
    for variant in ("classic", "graph", "hybrid"):
        chain, data = Chain(world), DataStream(cfg, len(world.nodes))
        fingerprints.append({"observed": table_hash(chain.observed), "audit": table_hash(chain.audits),
            "objects": table_hash(data.objects), "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()})
        agent = (ConservativeAgent(variant, world, settings, requirements, guard_settings) if profile == "guard"
                 else AdaptiveAgent(variant, world, settings, requirements))
        started = perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            result = Ledger(world, chain, data).run(agent)
        if not np.isfinite(agent.w).all() or agent.model.fitted and not all(
                np.isfinite(getattr(agent.model, attr)).all() for attr in ("beta", "gamma")):
            raise ArithmeticError("Неконечная базовая модель риска")
        decisions = agent.placement_log
        sizes = [d["n"] for d in decisions if d["nodes"]]
        initial = result.fragments[result.fragments.t_from == result.fragments.obj.map(result.objects.created_epoch)]
        actual = {int(obj): list(map(int, rows.node)) for obj, rows in initial.groupby("obj")}
        if any(actual.get(d["obj"], []) != d["nodes"] for d in decisions):
            raise AssertionError("Записанный размер не совпадает с решением")
        rejected = {d["obj"] for d in decisions if not d["nodes"]}
        results[variant] = {
            "objects_requested": len(result.objects), "objects_admitted": len(sizes),
            "initial_rejected_objects": len(rejected), "lost_objects_ledger": len(result.losses),
            "lost_admitted_objects": sum(int(obj) not in rejected for obj in result.losses.obj),
            "size_counts": dict(sorted(Counter(sizes).items())), "mean_initial_n": float(np.mean(sizes)) if sizes else None,
            "status_counts": dict(agent.status_counts), "placement_risk": agent.placement_risk,
            "requirements_passed_objects": sum(d["requirements_passed"] for d in decisions),
            "scored_objects": sum(d["scored"] for d in decisions), "repair_operations": len(result.repairs),
            "repair_traffic": float(result.repairs.traffic.sum()), "fragment_days": float(result.daily.fragments.sum()),
            "full_exposure": exposure_metrics(result, days), "post_warmup_cohort": exposure_metrics(result, days, cfg.train_days),
            "runtime_warnings": dict(Counter(str(w.message) for w in captured)), "seconds": perf_counter() - started,
            "placement_decisions": decisions, "repair_requests": agent.repair_request_log,
            "repair_best_effort_requests": sum(not row["requirements_passed"] for row in agent.repair_request_log),
            "final_sizes": list(map(int, result.objects.n)),
            "guard_diagnostics": agent.guard_rates.diagnostics if profile == "guard" else None}
        print(f"s{seed} {profile} {variant}: n={results[variant]['mean_initial_n']}, "
              f"прошли={results[variant]['requirements_passed_objects']}, "
              f"статусы={results[variant]['status_counts']}", flush=True)
    unchanged = hashes == {"nodes": table_hash(world.nodes), "outages": table_hash(world.outages), "silent": table_hash(world.silent)}
    return {"seed": seed, "profile": profile, "config": json.loads(cfg.to_json()),
        "settings": asdict(settings), "requirements": asdict(requirements),
        "guard_settings": asdict(guard_settings) if profile == "guard" else None,
        "world_hashes": hashes, "input_hashes": fingerprints[0], "world_unchanged": unchanged,
        "common_random_numbers": all(value == fingerprints[0] for value in fingerprints), "variants": results,
        "realized_shocks": len(world.shocks),
        "realized_destructive_shocks": int(world.shocks["destroyed"].sum()) if len(world.shocks) else 0}


def render(report):
    lines = ["# I3.2. Консервативная проверка перед выбором размера", "", "Дата: 10 ноября 2026 года.", "",
        "Профили: point — прежний адаптивный контроллер; guard — тот же контроллер с защитными сценариями; fixed4/fixed6 — контрольные фиксированные размеры с точечным проверяющим модулем.", "",
        "| Профиль | Вариант | Среднее n | Прошли / оценены | Cold start | Best effort | Потери принятых | Хранение | Трафик |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["summary"]:
        n = f"{row['mean_n']:.2f}" if row["mean_n"] is not None else "—"
        lines.append(f"| {row['profile']} | {row['variant']} | {n} | {row['passed']} / {row['scored']} | "
            f"{row['statuses'].get('unscored_cold_start', 0)} | {row['statuses'].get('no_feasible_found_best_effort', 0)} | "
            f"{row['lost_admitted']} | {row['fragment_days']:.0f} | {row['repair_traffic']:.2f} |")
    lines += ["", f"Успешных случаев: {report['completed_cases']}; ошибок: {len(report['failures'])}. "
        f"Изоляция: {report['integrity_passed']}; совпадение миров между профилями: {report['paired_profiles']}.",
        "", "Прошедшая проверка — решение в защитном наборе моделей, не объявленная гарантия 95% для объекта. "
        "0,95 — условный квантиль отдельной Gamma-модели интенсивности. Ансамбль доступности конечен. "
        "Покрытие и цена проверяются внешним аудитом, а холодный старт и best effort не считаются выполненной защитой.",
        "", "Общий лимит потери — 0,00004 за 26 часов в текущей модели Ledger. Физическая потеря и длительный простой ещё не разделены. "
        "Размер выбирается при создании, а не меняется автоматически весь срок жизни. Предыдущие результаты не перезаписываются.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="I3.2: новые зёрна и защитный слой")
    parser.add_argument("--seeds", type=int, nargs="+", default=[9, 10, 11])
    parser.add_argument("--profiles", nargs="+", choices=["point", "guard", "fixed4", "fixed6"],
                        default=["point", "guard", "fixed4", "fixed6"])
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("series_i32"))
    args = parser.parse_args()
    if args.days < 8 or args.nodes < 8 or any(seed < 0 for seed in args.seeds):
        parser.error("Некорректные параметры")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    folder = Path(__file__).resolve().parent
    sources = ("conservative.py", "adaptive.py", "agents.py", "risk_graph.py", "run_guarded.py", "run_compare.py")
    provenance = {"main": source_hashes(), "integration": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in sources}}
    provenance["runtime"] = {"python": sys.version, "numpy": np.__version__,
        "threads": {key: os.environ.get(key) for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    cases, failures = [], []
    for seed in args.seeds:
        for profile in args.profiles:
            job = {"seed": seed, "profile": profile, "days": args.days, "nodes": args.nodes}
            path = args.output_dir / f"s{seed}_{profile}.json"
            if path.exists():
                saved = json.loads(path.read_text(encoding="utf-8"))
                if saved["job"] != job or saved["provenance"] != provenance:
                    raise ValueError("Несовместимый кэш; выберите другую папку: " + str(path))
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
    groups = {}
    for case in cases:
        value = json.dumps({"world": case["world_hashes"], "inputs": case["input_hashes"]}, sort_keys=True)
        groups.setdefault(case["seed"], set()).add(value)
    paired = all(len(values) == 1 for values in groups.values())
    after = {"main": source_hashes(), "integration": {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in sources},
             "runtime": provenance["runtime"]}
    integrity = provenance == after and paired and all(c["world_unchanged"] and c["common_random_numbers"] for c in cases)
    report = {"schema": "taris-guard-i32/1", "completed_cases": len(cases), "failures": failures,
        "provenance": provenance, "summary": summarize(cases), "integrity_passed": integrity,
        "paired_profiles": paired, "distinct_worlds": len({v for values in groups.values() for v in values}),
        "guard_settings": asdict(GuardSettings())}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "RESULTS.md").write_text(render(report), encoding="utf-8")
    print("Сводка:", (args.output_dir / "RESULTS.md").resolve())
    if failures or not integrity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
