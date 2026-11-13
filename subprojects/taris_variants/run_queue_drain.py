"""I6.1: причинный replay освобождения занятого служебного места."""
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
from subprojects.taris_variants.cancel_obsolete import CancellationLedger
from subprojects.taris_variants.shared_channel import SharedChannelLedger
from subprojects.taris_variants.targeted_proof import TargetedProofLedger
from subprojects.taris_variants.run_compare import source_hashes, table_hash
from subprojects.taris_variants.observable_memory import ObservableMemoryLedger
from subprojects.taris_variants.observable_queue import ObservingPolicy
from subprojects.taris_variants.conditional_work import conditional_work
from subprojects.taris_variants.observation_labels import label_snapshots
from subprojects.taris_variants.audit_queue_drain_stress import audit_stress_record
from subprojects.taris_variants.queue_drain import QueueDrainLedger
from subprojects.taris_variants.queue_control import QueueControlLedger, QueueControlSettings
from subprojects.taris_variants.disk_stress import DiskStressLedger, DiskWave, public_wave_targets
from subprojects.taris_variants.audit_cancellation import audit_cancellations


def provenance():
    folder = Path(__file__).resolve().parent
    return {"main": source_hashes(), "integration": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name in ("run_queue_drain.py", "queue_drain.py", "audit_queue_drain.py", "audit_queue_drain_stress.py", "EXPERIMENT_I61.md", "run_queue_control.py", "queue_control.py", "audit_queue_control.py", "audit_queue_stress.py", "EXPERIMENT_I6_QUEUE.md", "run_stress_sensitivity.py", "EXPERIMENT_I51.md", "run_disk_stress.py", "disk_stress.py", "audit_disk_stress.py", "run_observed.py", "observable_queue.py", "observable_memory.py", "conditional_work.py", "observation_labels.py", "audit_targeted_proof.py", "audit_cancellation.py", "audit_physical.py", "targeted_proof.py", "shared_channel.py", "cancel_obsolete.py", "resumable.py", "queued.py", "physical.py", "run_targeted_proof.py", "run_compare.py", "agents.py", "risk_graph.py")}}


def run_case(seed=103, days=45, nodes=24, stress_mode="close", control_mode="combined", drain_mode="original"):
    control = QueueControlSettings(control_mode in ("reserve", "combined"), control_mode in ("bypass", "combined"))
    cfg = scenario("S2", seed=seed, horizon_days=days, train_days=max(1, days//2),
        n_initial_nodes=nodes, node_join_rate=.1, n_dc=8, n_providers=4, n_regions=3, blocks_per_epoch=1, k=2)
    settings = VariantSettings(n=4, block_days=7, reserve_horizon_days=30)
    world = World(cfg)
    targets = public_wave_targets(world.nodes)
    if stress_mode == "control":
        waves = ()
    elif stress_mode in ("close", "after_decision", "medium", "far"):
        waves = (DiskWave(15., tuple(sorted(targets["groups"][0]+targets["groups"][1]))),
                 DiskWave(15.+{"close": .5, "after_decision": 1.5, "medium": 3., "far": 7.}[stress_mode], targets["groups"][2]))
    else:
        raise ValueError("Неизвестная стресс-ветвь")
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
            agent, ledger = make_agent(variant, world, settings), QueueDrainLedger(world, chain, data, qsettings, waves=waves, control=control, drain_irrelevant_proof=drain_mode == "pressure")
            observer = ObservingPolicy(agent, ledger)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", RuntimeWarning)
                result = ledger.run(observer)
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
            assert metrics["targeted_proof_starts"] == metrics["proof_bypass_starts"]
            record = {"metrics": metrics, "queue_settings": asdict(qsettings),
                "transport_settings": asdict(ledger.transport), "decoded_prefixes": ledger.decoded_prefixes,
                "receiver_prefixes": ledger.receiver_prefixes, "block_results": ledger.block_results,
                "close_checks": ledger.close_checks,
                "target_additions": ledger.target_additions,
                "obsolete_cancellations": ledger.obsolete_cancellations,
                "parked_intervals": ledger.parked_intervals, "scheduler_events": ledger.scheduler_events,
                "proof_selection_log": ledger.proof_selection_log,
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
            snapshots = observer.snapshots
            record["observations"] = snapshots
            record["conditional_work"] = [{"time": row["time"], "scenarios": [conditional_work(row, mode)
                for mode in ("trust_ack", "restart_partial")]} for row in snapshots]
            record["posthoc_labels"] = label_snapshots(snapshots, ledger.jobs, ledger.segments, ledger.block_results, float(days))
            record["drain_irrelevant_proof"] = ledger.drain_irrelevant_proof
            record["control_settings"] = asdict(control)
            record["read_wait_pauses"] = ledger.read_wait_pauses
            record["repair_admissions"] = ledger.repair_admissions
            record["stress_schedule"] = [asdict(wave) for wave in waves]
            record["disk_notifications"] = ledger.disk_notifications
            record["physical_transport_audit"] = audit_stress_record(record, cfg, world, chain, data)
            audit_cancellations(record)
            if drain_mode == "original":
                old_path = Path(__file__).with_name("series_i6_replay")/f"s{seed}_{stress_mode}_combined.json"
                old_saved = json.loads(old_path.read_text(encoding="utf-8"))
                assert old_saved["provenance"]["main"] == source_hashes()
                old_case = old_saved["case"]
                assert old_case["world_hashes"] == world_hashes and old_case["input_hashes"] == fingerprint
                old_record = old_case["results"][profile][variant]
                keys = ("jobs", "segments", "block_results", "scheduler_events", "buffers", "decoded_prefixes",
                        "receiver_prefixes", "proof_selection_log", "losses", "initial_writes", "physical_fragments",
                        "queue_events", "repair_outcomes", "declarations", "returns", "cleanup", "observations", "read_wait_pauses", "repair_admissions")
                for key in keys:
                    assert json.loads(json.dumps(record[key], allow_nan=False)) == old_record[key], key
                for key, value in old_record["metrics"].items():
                    if key != "schema":
                        assert metrics[key] == value, key
                record["baseline_identical"] = True
                record["baseline_input_hash"] = hashlib.sha256(old_path.read_bytes()).hexdigest()
            results[profile][variant] = record
            print(f"s{seed} {stress_mode} {drain_mode} {profile} {variant}: потерь={metrics['irrecoverable_objects']}, "
                  f"хвост={metrics['unfinished_jobs']}, бюджет={sum(metrics['serviced_volume'].values()):.3f}", flush=True)
    return {"seed": seed, "drain_mode": drain_mode, "queue_control_mode": control_mode, "control_settings": asdict(control), "stress_mode": stress_mode, "public_targets": targets, "stress_schedule": [asdict(wave) for wave in waves], "config": json.loads(cfg.to_json()), "settings": asdict(settings),
        "world_hashes": world_hashes, "world_unchanged": world_hashes == {
            key: table_hash(getattr(world, key)) for key in world_hashes},
        "input_hashes": fingerprints[0], "common_random_numbers": all(v == fingerprints[0] for v in fingerprints),
        "results": results, "realized_shocks": len(world.shocks),
        "realized_destructive_shocks": int(world.shocks.destroyed.sum()) if len(world.shocks) else 0}


def summarize(cases):
    return [{"seed": case["seed"], "drain_mode": case["drain_mode"], "queue_control_mode": case["queue_control_mode"], "stress_mode": case["stress_mode"], "profile": profile,
             "variant": variant, **record["metrics"]}
            for case in cases for profile, variants in case["results"].items() for variant, record in variants.items()]


def render(report):
    lines = ["# I6.1. Освобождение служебного места", "", "Дата: 12 ноября 2026 года.", "",
        "| Мир | Правило | Канал | Стратегия | Объектов | Безвозвратные потери | Неполный приём | Дисковый дефицит, сут. | Спасает только буфер, сут. | Бюджет передачи | Хвост заданий |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in report["summary"]:
        lines.append(f"| {r['seed']} | {r['drain_mode']} | {r['profile']} | {r['variant']} | {r['objects']} | {r['irrecoverable_objects']} | "
            f"{r['initial_incomplete_objects']} | {r['disk_deficit_object_days']:.6f} | {r['buffer_only_object_days']:.6f} | "
            f"{sum(r['serviced_volume'].values()):.3f} | {r['unfinished_jobs']} |")
    lines += ["", f"Ветвей: {len(report['cases'])}; изоляция и общие случайные числа: {report['integrity_passed']}.",
        "", "Бюджет передачи включает неуспешные попытки и прошедшую часть фазы на горизонте. Это не точные доставленные байты. "
        "Первичные записи не конкурируют за этот канал. Здоровый буфер освобождается только после оплаченной проверки k разных копий.",
        "", "Каждая строка — отдельное исполнение одного мира; ветви одного мира не являются независимыми мирами. "
        "Полные журналы содержат ожидание начатых заданий и отдельно возраст ещё ожидающих. Редкая сохранность этой серией не подтверждается.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drain-modes", nargs="+", choices=("original", "pressure"), default=["original", "pressure"])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("series_i61_replay"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before, cases = provenance(), []
    for drain_mode in args.drain_modes:
        job = {"seed": 103, "days": 45, "nodes": 24, "stress_mode": "close", "queue_control_mode": "combined", "drain_mode": drain_mode}
        path = args.output_dir/f"s103_close_{drain_mode}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["job"] != job or saved["provenance"] != before:
                raise ValueError("Несовместимый кэш; используйте другую папку")
        else:
            saved = {"job": job, "provenance": before, "case": run_case(drain_mode=drain_mode)}
            path.write_text(json.dumps(saved, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        cases.append(saved["case"])
    integrity = before == provenance() and all(c["world_unchanged"] and c["common_random_numbers"] for c in cases)
    report = {"schema": "taris-queue-drain-replay/1", "provenance": before, "integrity_passed": integrity,
        "cases": [{"seed": c["seed"], "drain_mode": c["drain_mode"], "world_hashes": c["world_hashes"]} for c in cases],
        "summary": summarize(cases)}
    (args.output_dir/"summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (args.output_dir/"RESULTS.md").write_text(render(report), encoding="utf-8")
    print("Сводка:", args.output_dir/"RESULTS.md", flush=True)
    if not integrity:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
