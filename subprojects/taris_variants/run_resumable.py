"""I4.3: возобновляемые префиксы на парных и новых мирах."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
from subprojects.taris_variants.agents import VariantSettings, make_agent
from subprojects.taris_variants.queued import QueueSettings, QueuedLedger
from subprojects.taris_variants.resumable import ResumableLedger, TransportSettings
from subprojects.taris_variants.run_compare import source_hashes, table_hash


def provenance():
    folder = Path(__file__).resolve().parent
    return {"main": source_hashes(), "integration": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name in ("resumable.py", "queued.py", "physical.py", "run_resumable.py", "run_compare.py", "agents.py", "risk_graph.py")}}


def run_case(seed, days=45, nodes=24):
    cfg = scenario("S2", seed=seed, horizon_days=days, train_days=max(1, days//2),
        n_initial_nodes=nodes, node_join_rate=.1, n_dc=8, n_providers=4, n_regions=3, blocks_per_epoch=1, k=2)
    settings = VariantSettings(n=4, block_days=7, reserve_horizon_days=30)
    world = World(cfg)
    world_hashes = {key: table_hash(getattr(world, key)) for key in ("nodes", "outages", "silent")}
    results, fingerprints = {}, []
    for profile, rate in (("ample", 2.), ("constrained", .02)):
        results[profile] = {}
        qsettings = QueueSettings(rate_size_per_hour=rate)
        for variant in ("classic", "graph", "hybrid"):
            chain, data = Chain(world), DataStream(cfg, len(world.nodes))
            fingerprint = {"observed": table_hash(chain.observed), "audits": table_hash(chain.audits),
                "true_down": table_hash(chain.true_down), "objects": table_hash(data.objects),
                "capacity": hashlib.sha256(data.capacity.tobytes()).hexdigest()}
            fingerprints.append(fingerprint)
            agent, ledger = make_agent(variant, world, settings), ResumableLedger(world, chain, data, qsettings)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", RuntimeWarning)
                result = ledger.run(agent)
            if not np.isfinite(agent.w).all():
                raise ArithmeticError("Неконечные веса политики")
            model = getattr(agent, "model", None)
            if model is not None and model.fitted:
                if not all(np.isfinite(getattr(model, field)).all() for field in ("beta", "gamma")):
                    raise ArithmeticError("Неконечные коэффициенты модели")
            assert np.all(ledger.used >= 0) and np.all(ledger.used <= ledger.capacity)
            allocated = np.bincount([ledger.f_node[f] for f, yes in enumerate(ledger.allocated) if yes], minlength=ledger.N)
            assert np.array_equal(allocated+ledger.reserved, ledger.used)
            metrics = result.metrics()
            assert metrics["max_outstanding_jobs"] <= qsettings.max_jobs
            assert metrics["peak_central_size"] <= qsettings.buffer_capacity_size+1e-8
            record = {"metrics": metrics, "queue_settings": asdict(qsettings),
                "transport_settings": asdict(ledger.transport), "decoded_prefixes": ledger.decoded_prefixes,
                "receiver_prefixes": ledger.receiver_prefixes, "block_results": ledger.block_results,
                "close_checks": ledger.close_checks,
                "target_additions": ledger.target_additions,
                "runtime_warnings": dict(Counter(str(w.message) for w in captured)),
                "losses": result.losses.to_dict("records"), "initial_writes": ledger.initial_writes,
                "buffers": ledger.buffer_intervals, "jobs": ledger.jobs, "segments": ledger.segments,
                "queue_events": ledger.queue_events, "repair_outcomes": ledger.operation_outcomes,
                "declarations": ledger.declarations, "returns": ledger.returns, "cleanup": ledger.cleanup,
                "reserved_by_node": ledger.reserved.tolist(), "allocated_slots_by_node": ledger.used.tolist(),
                "physical_fragments": [{"obj": int(ledger.f_obj[f]), "idx": int(ledger.f_idx[f]),
                    "node": int(ledger.f_node[f]), "start": float(ledger.f_from[f]),
                    "physical_end": float(ledger.f_dead[f]) if np.isfinite(ledger.f_dead[f]) else None,
                    "allocated": bool(ledger.allocated[f])} for f in range(len(ledger.f_obj))]}
            results[profile][variant] = record
            print(f"s{seed} {profile} {variant}: потерь={metrics['irrecoverable_objects']}, "
                  f"хвост={metrics['unfinished_jobs']}, бюджет={sum(metrics['serviced_volume'].values()):.3f}", flush=True)
    return {"seed": seed, "config": json.loads(cfg.to_json()), "settings": asdict(settings),
        "world_hashes": world_hashes, "world_unchanged": world_hashes == {
            key: table_hash(getattr(world, key)) for key in world_hashes},
        "input_hashes": fingerprints[0], "common_random_numbers": all(v == fingerprints[0] for v in fingerprints),
        "results": results, "realized_shocks": len(world.shocks),
        "realized_destructive_shocks": int(world.shocks.destroyed.sum()) if len(world.shocks) else 0}


def summarize(cases):
    rows = []
    fields = ("objects", "irrecoverable_objects", "initial_incomplete_objects", "lost_admitted_objects",
        "object_days", "unavailable_object_days", "disk_deficit_object_days", "buffer_only_object_days",
        "physical_fragment_days", "buffer_object_days", "unfinished_jobs", "queued_jobs", "open_buffers",
        "started_wait_days_sum", "started_jobs", "pending_wait_days_sum", "decoded_prefix_size_days",
        "receiver_prefix_size_days", "block_attempts", "failed_blocks", "successful_close_checks", "full_buffers_created")
    for profile in ("ample", "constrained"):
        for variant in ("classic", "graph", "hybrid"):
            metrics = [c["results"][profile][variant]["metrics"] for c in cases]
            row = {"profile": profile, "variant": variant, **{key: sum(m[key] for m in metrics) for key in fields}}
            row["serviced_volume"] = dict(sum((Counter(m["serviced_volume"]) for m in metrics), Counter()))
            row["queue_rejections"] = dict(sum((Counter(m["queue_rejections"]) for m in metrics), Counter()))
            row["job_statuses"] = dict(sum((Counter(m["job_statuses"]) for m in metrics), Counter()))
            rows.append(row)
    return rows


def render(report):
    lines = ["# I4.3. Возобновляемая передача и удержание буфера", "", "Дата: 10 ноября 2026 года.", "",
        "| Канал | Стратегия | Объектов | Безвозвратные потери | Неполный приём | Дисковый дефицит, сут. | Спасает только буфер, сут. | Бюджет передачи | Хвост заданий |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["summary"]:
        lines.append(f"| {r['profile']} | {r['variant']} | {r['objects']} | {r['irrecoverable_objects']} | "
            f"{r['initial_incomplete_objects']} | {r['disk_deficit_object_days']:.6f} | {r['buffer_only_object_days']:.6f} | "
            f"{sum(r['serviced_volume'].values()):.3f} | {r['unfinished_jobs']} |")
    lines += ["", f"Миров: {len(report['cases'])}; изоляция и общие случайные числа: {report['integrity_passed']}.",
        "", "Бюджет передачи включает неуспешные попытки и прошедшую часть фазы на горизонте. Это не точные доставленные байты. "
        "Первичные записи не конкурируют за этот канал. Здоровый буфер освобождается только после оплаченной проверки k разных копий.",
        "", "Хвост суммируется по прогонам, а не является одновременной длиной одной очереди. "
        "Полные журналы содержат ожидание начатых заданий и отдельно возраст ещё ожидающих. Редкая сохранность этой серией не подтверждается.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[15, 16, 17, 18, 19, 20])
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("series_i43"))
    args = parser.parse_args()
    if args.days < 8 or args.nodes < 8 or any(seed < 0 for seed in args.seeds):
        parser.error("Некорректные параметры")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before, cases = provenance(), []
    for seed in args.seeds:
        job = {"seed": seed, "days": args.days, "nodes": args.nodes}
        path = args.output_dir/f"s{seed}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["job"] != job or saved["provenance"] != before:
                raise ValueError("Несовместимый кэш; используйте другую папку")
        else:
            saved = {"job": job, "provenance": before, "case": run_case(seed, args.days, args.nodes)}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        cases.append(saved["case"])
    integrity = before == provenance() and all(c["world_unchanged"] and c["common_random_numbers"] for c in cases)
    report = {"schema": "taris-resume-i43/1", "provenance": before, "integrity_passed": integrity,
        "cases": [{"seed": c["seed"], "world_hashes": c["world_hashes"]} for c in cases], "summary": summarize(cases)}
    (args.output_dir/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.output_dir/"RESULTS.md").write_text(render(report), encoding="utf-8")
    print("Сводка:", args.output_dir/"RESULTS.md", flush=True)
    if not integrity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
