"""Парное сравнение семантики Ledger и PhysicalLedger, без правок основного кода."""
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
import numpy as np
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from subprojects.taris_variants.agents import VariantSettings, make_agent
from subprojects.taris_variants.physical import PhysicalLedger
from subprojects.taris_variants.run_compare import source_hashes, table_hash, exposure_metrics


def provenance():
    folder = Path(__file__).resolve().parent
    return {"main": source_hashes(), "integration": {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in ("physical.py", "run_physical.py", "run_compare.py", "agents.py", "risk_graph.py")}}


def run_case(seed, days=45, nodes=24):
    cfg = scenario("S2", seed=seed, horizon_days=days, train_days=max(1, days//2),
        n_initial_nodes=nodes, node_join_rate=0.1, n_dc=8, n_providers=4,
        n_regions=3, blocks_per_epoch=1, k=2)
    settings = VariantSettings(n=4, block_days=7, reserve_horizon_days=30)
    world = World(cfg)
    world_hashes = {key: table_hash(getattr(world, key)) for key in ("nodes", "outages", "silent")}
    results, fingerprints = {}, []
    for mode, executor in (("administrative", Ledger), ("physical", PhysicalLedger)):
        results[mode] = {}
        for variant in ("classic", "graph", "hybrid"):
            chain, data = Chain(world), DataStream(cfg, len(world.nodes))
            fingerprints.append({"observed": table_hash(chain.observed), "audits": table_hash(chain.audits),
                "objects": table_hash(data.objects), "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()})
            agent = make_agent(variant, world, settings)
            ledger = executor(world, chain, data)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", RuntimeWarning)
                result = ledger.run(agent)
            if not np.isfinite(agent.w).all():
                raise ArithmeticError("Неконечные веса политики")
            model = getattr(agent, "model", None)
            if model is not None and model.fitted:
                if not all(np.isfinite(getattr(model, field)).all() for field in ("beta", "gamma")):
                    raise ArithmeticError("Неконечные коэффициенты модели")
            if np.any(ledger.used < 0) or np.any(ledger.used > ledger.capacity):
                raise AssertionError("Нарушение ёмкости")
            incomplete = (sum(len(r[3]) < cfg.k for r in ledger.initial_writes) if mode == "physical"
                          else int((result.objects.n < cfg.k).sum()))
            record = {"losses": result.losses.to_dict("records"),
                "lost_objects": len(result.losses), "initial_incomplete": incomplete,
                "runtime_warnings": dict(Counter(str(w.message) for w in captured)),
                "repair_traffic": float(result.repairs.traffic.sum()),
                "repairs": result.repairs.to_dict("records"),
                "objects": result.objects.replace({np.inf: None}).reset_index().to_dict("records")}
            if mode == "physical":
                for obj in range(len(ledger.obj_frags)):
                    refs = Counter(ledger.f_idx[f] for f in range(len(ledger.f_obj))
                                   if ledger.f_obj[f] == obj and ledger.f_exist[f])
                    assert refs == ledger.references[obj] and len(refs) == ledger.obj_exist[obj]
                allocated = np.bincount([ledger.f_node[f] for f, yes in enumerate(ledger.allocated) if yes], minlength=ledger.N)
                assert np.array_equal(allocated + ledger.reserved, ledger.used)
                record["physical_metrics"] = ledger.physical_report(float(days))
                record["initial_writes"] = ledger.initial_writes
                record["initial_write_rejections"] = ledger.initial_write_rejections
                record["declarations"] = ledger.declarations
                record["returns"] = ledger.returns
                record["cleanup"] = ledger.cleanup
                record["repair_outcomes"] = ledger.operation_outcomes
                record["physical_fragments"] = [{"obj": int(ledger.f_obj[f]), "idx": int(ledger.f_idx[f]),
                    "node": int(ledger.f_node[f]), "start": float(ledger.f_from[f]),
                    "physical_end": float(ledger.f_dead[f]) if np.isfinite(ledger.f_dead[f]) else None,
                    "allocated": bool(ledger.allocated[f])} for f in range(len(ledger.f_obj))]
            else:
                record["administrative_metrics"] = exposure_metrics(result, days)
            results[mode][variant] = record
            print(f"s{seed} {mode} {variant}: потерь={len(result.losses)}, трафик={record['repair_traffic']}", flush=True)
    unchanged = world_hashes == {key: table_hash(getattr(world, key)) for key in world_hashes}
    return {"seed": seed, "config": json.loads(cfg.to_json()), "settings": asdict(settings),
        "world_hashes": world_hashes, "world_unchanged": unchanged,
        "common_random_numbers": all(v == fingerprints[0] for v in fingerprints),
        "input_hashes": fingerprints[0], "results": results,
        "realized_shocks": len(world.shocks),
        "realized_destructive_shocks": int(world.shocks.destroyed.sum()) if len(world.shocks) else 0}


def summarize(cases):
    rows = []
    for variant in ("classic", "graph", "hybrid"):
        for mode in ("administrative", "physical"):
            records = [c["results"][mode][variant] for c in cases]
            metrics = [r["physical_metrics" if mode == "physical" else "administrative_metrics"] for r in records]
            exposure = sum(m["object_days"] for m in metrics)
            unavailable = sum(m["unavailable_object_days"] for m in metrics)
            row = {"mode": mode, "variant": variant, "objects": sum(m["objects"] for m in metrics),
                "lost_objects": sum(r["lost_objects"] for r in records),
                "initial_incomplete": sum(r["initial_incomplete"] for r in records),
                "repair_traffic": sum(r["repair_traffic"] for r in records),
                "object_days": exposure, "unavailable_object_days": unavailable,
                "unavailability_share": unavailable / exposure if exposure else None}
            if mode == "physical":
                row.update({key: sum(m[key] for m in metrics) for key in (
                    "physical_fragment_days", "declarations", "live_at_declaration", "returns",
                    "redundant_deleted", "pending_writes")})
                row["repair_outcomes"] = dict(sum((Counter(m["repair_outcomes"]) for m in metrics), Counter()))
            rows.append(row)
    return rows


def render(report):
    lines = ["# I4.1. Два режима учёта потери", "", "Дата: 10 ноября 2026 года.", "",
        "Старый режим считает административную потерю, новый — безвозвратную утрату разных номеров чанков. Эти значения имеют разные определения.", "",
        "| Режим | Политика | Объектов | Потери | Неполный приём | Недоступность, объекто-сутки | Трафик ремонта |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in report["summary"]:
        lines.append(f"| {r['mode']} | {r['variant']} | {r['objects']} | {r['lost_objects']} | {r['initial_incomplete']} | {r['unavailable_object_days']:.6f} | {r['repair_traffic']:.2f} |")
    lines += ["", f"Завершено случаев: {len(report['cases'])}. Изоляция и общие случайные числа: {report['integrity_passed']}.",
        "", "Недоступность нового режима учитывает целые копии вне активной карты (возможность физического чтения), поэтому не является SLA пользовательского поиска. "
        "Время после потери включено. В обоих режимах учитываются физические условия чтения, но у старого режима после административной потери всё оставшееся время считается недоступным.",
        "", "Источник истины — журналы исполнителя. Ни условный оконный cap, ни 95%-я гарантия этим пилотом не подтверждаются. "
        "Модель без буфера требует k источников на старте и на завершении ремонта; общей очереди и миграции пока нет.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[12, 13, 14])
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("series_i41_final"))
    args = parser.parse_args()
    if args.days < 8 or args.nodes < 8 or any(seed < 0 for seed in args.seeds):
        parser.error("Некорректные параметры")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before, cases = provenance(), []
    for seed in args.seeds:
        job = {"seed": seed, "days": args.days, "nodes": args.nodes}
        path = args.output_dir / f"s{seed}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["job"] != job or saved["provenance"] != before:
                raise ValueError("Несовместимый кэш; выберите другую папку")
        else:
            saved = {"job": job, "provenance": before, "case": run_case(seed, args.days, args.nodes)}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        cases.append(saved["case"])
    integrity = before == provenance() and all(c["world_unchanged"] and c["common_random_numbers"] for c in cases)
    report = {"schema": "taris-physical-i41/1", "provenance": before,
        "cases": [{"seed": c["seed"], "world_hashes": c["world_hashes"]} for c in cases],
        "integrity_passed": integrity, "summary": summarize(cases)}
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.output_dir / "RESULTS.md").write_text(render(report), encoding="utf-8")
    print("Сводка:", args.output_dir / "RESULTS.md", flush=True)
    if not integrity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
