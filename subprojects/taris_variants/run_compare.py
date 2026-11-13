"""Изолированный запуск трёх вариантов в неизменённом Ledger."""
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
from time import perf_counter
import numpy as np

from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from subprojects.taris_variants.agents import VariantSettings, make_agent


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "taris_sim").glob("*.py")}


def table_hash(table):
    return hashlib.sha256(table.to_json(orient="split", double_precision=15).encode()).hexdigest()


def exposure_metrics(result, days, cohort_start=0):
    """Недоступность на всём горизонте, включая время после потери объекта."""
    objects = result.objects[result.objects["created_epoch"] >= cohort_start]
    shares = result.unavailability(n_objects=len(result.objects)).set_index("obj")["unavail_share"]
    exposure = unavailable = lost_days = 0.0
    for obj, row in objects.iterrows():
        created = float(row["created_epoch"])
        lost = min(float(row["lost_t"]), float(days))
        before = max(0.0, lost - created)
        after = max(0.0, days - max(created, lost))
        if before > 0 and obj not in shares.index:
            raise ValueError("Нет оценки недоступности существовавшего объекта")
        exposure += days - created
        unavailable += float(shares.get(obj, 0.0)) * before + after
        lost_days += after
    return {"objects": len(objects), "lost_objects": int((objects["lost_t"] < days).sum()),
            "object_days": exposure, "lost_object_days": lost_days,
            "unavailable_object_days": unavailable,
            "unavailability_share": unavailable / exposure if exposure else None}


def run_case(scenario_name, seed, days=40, initial_nodes=32, settings=None, k=2):
    settings = settings or VariantSettings(n=4, block_days=7, reserve_horizon_days=30)
    cfg = scenario(scenario_name, seed=seed, horizon_days=days, train_days=max(1, days // 2),
                   n_initial_nodes=initial_nodes, node_join_rate=0.1, n_dc=8,
                   n_providers=4, n_regions=3, blocks_per_epoch=1, k=k)
    world = World(cfg)
    original_world = {"nodes": table_hash(world.nodes), "outages": table_hash(world.outages),
                      "silent": table_hash(world.silent)}
    variants = {}
    fingerprints = []
    for variant in ("classic", "graph", "hybrid"):
        chain, data = Chain(world), DataStream(cfg, len(world.nodes))
        fingerprints.append({"observed": table_hash(chain.observed), "audits": table_hash(chain.audits),
                             "objects": table_hash(data.objects),
                             "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()})
        policy = make_agent(variant, world, settings)
        started = perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            result = Ledger(world, chain, data).run(policy)
        seconds = perf_counter() - started
        fitted_model = getattr(policy, "model", None)
        numerical_finite = bool(np.isfinite(policy.w).all())
        if fitted_model is not None and fitted_model.fitted:
            numerical_finite = numerical_finite and all(np.isfinite(getattr(fitted_model, field)).all()
                                                       for field in ("beta", "gamma"))
        if not numerical_finite:
            raise ArithmeticError(f"Неконечная модель риска: {variant}")
        unavailability = result.unavailability(n_objects=len(result.objects))
        initial = result.fragments[
            result.fragments["t_from"] == result.fragments["obj"].map(result.objects["created_epoch"])]
        variants[variant] = {
            "name": policy.name, "objects": len(result.objects), "lost_objects": len(result.losses),
            "repair_operations": len(result.repairs), "repair_traffic": float(result.repairs["traffic"].sum()),
            "fragment_days": float(result.daily["fragments"].sum()),
            "final_fragments": int(result.daily["fragments"].iloc[-1]),
            "object_unavailability_mean_before_loss": float(unavailability["unavail_share"].mean()) if len(unavailability) else None,
            "unavailability_objects_evaluated": len(unavailability),
            "full_exposure": exposure_metrics(result, days),
            "post_warmup_cohort": exposure_metrics(result, days, cfg.train_days),
            "short_placements": len(result.short_placements),
            "risk_fits": len(getattr(policy, "fits", [])),
            "graph_stats": getattr(policy, "graph_stats", None),
            "placement_risk": getattr(policy, "placement_risk", None),
            "seconds": seconds, "losses": result.losses.to_dict("records"),
            "numerical_finite": numerical_finite,
            "runtime_warnings": dict(Counter(str(w.message) for w in captured)),
            "graph_decisions": getattr(policy, "decision_log", []),
            "placement_decisions": getattr(policy, "placement_log", []),
            "repair_requests": getattr(policy, "repair_request_log", []),
            "initial_placements": [{"obj": int(obj), "nodes": list(map(int, rows["node"]))}
                                   for obj, rows in initial.groupby("obj", sort=True)],
        }
        print(f"{scenario_name} s{seed} {variant}: потерь={len(result.losses)}, "
              f"ремонтов={len(result.repairs)}, время={seconds:.2f} с", flush=True)
    after = {"nodes": table_hash(world.nodes), "outages": table_hash(world.outages), "silent": table_hash(world.silent)}
    return {"scenario": scenario_name, "seed": seed, "config": json.loads(cfg.to_json()),
            "settings": asdict(settings), "world_unchanged": original_world == after,
            "common_random_numbers": all(value == fingerprints[0] for value in fingerprints),
            "world_hashes": original_world, "input_hashes": fingerprints[0], "variants": variants,
            "realized_shocks": len(world.shocks),
            "realized_destructive_shocks": int(world.shocks["destroyed"].sum()) if len(world.shocks) else 0}


def main():
    parser = argparse.ArgumentParser(description="Classic / TARIS-G / TARIS-Hybrid")
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--scenarios", nargs="+", choices=["S0", "S2"], default=["S0", "S2"])
    parser.add_argument("--nodes", type=int, default=32)
    parser.add_argument("--repair-mode", choices=["thiele", "fixed"], default="thiele")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--fragments", type=int, default=4)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--reserve-horizon", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("comparison_results.json"))
    args = parser.parse_args()
    if args.days < 8 or args.nodes < 8 or args.k < 1 or any(seed < 0 for seed in args.seeds):
        parser.error("Нужно не менее восьми суток и восьми начальных нод; зёрна неотрицательны")
    before = source_hashes()
    integration_sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in Path(__file__).resolve().parent.glob("*.py")}
    settings = VariantSettings(n=args.fragments, block_days=args.block_days,
        reserve_horizon_days=args.reserve_horizon, repair_mode=args.repair_mode)
    settings.validate(args.k)
    cases = [run_case(name, seed, args.days, args.nodes, settings, args.k)
             for name in args.scenarios for seed in args.seeds]
    report = {"schema": "taris-three-variants/1", "stage": "fixed-budget integration pilot",
              "main_sources_unchanged": before == source_hashes(), "cases": cases,
              "main_source_hashes": before, "integration_source_hashes": integration_sources,
              "limitations": ["Все потери определяются действующим Ledger, включая признание после T_rep",
                              "Граф описывает доступность, а не обученную физическую потерю",
                              f"Все варианты имеют одинаковый начальный бюджет n={settings.n}, k={args.k}",
                              "Очереди и прогноз из автономного прототипа в этот Ledger не переносились"]}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not report["main_sources_unchanged"] or not all(c["common_random_numbers"] and c["world_unchanged"] for c in cases):
        raise AssertionError("Нарушена изоляция или общая случайность")
    print("Отчёт:", args.output.resolve())


if __name__ == "__main__":
    main()
