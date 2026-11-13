"""Сопоставимый прогон baseline-политик и TARIS на конфигурации MAIN.

По умолчанию: S0/S2, зёрна 1–10, 1095 суток. Сценарные параметры MAIN
не переопределяются. Baseline сохраняют собственную логику восстановления;
режимы reserve_* используют общий порог R_min=2.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from taris_sim.config import scenario
from taris_sim.scenario import MAIN, make_policy


BASELINES = (
    "uniform13", "crush13", "copysets13", "filecoin13", "sia13", "storj13",
    "dynamic_ec13", "aec13", "gbm13", "k1_pg13", "k2_laplace13", "k2_nuts13", "oracle13",
)
RESERVE_POLICIES = (
    "reserve_oracle", "reserve_uniform", "reserve_agent", "reserve_agent_greedy",
    "reserve_agent_capped", "reserve_tariff",
)
POLICIES = BASELINES + RESERVE_POLICIES
OUT = os.path.join(ROOT, "runs", "step5_comparison")


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def _make_reserve(name, world, probe):
    from taris_sim.policy import ReservePolicy

    placement = {
        "reserve_oracle": ("oracle", "true"),
        "reserve_uniform": ("uniform", "true"),
        "reserve_agent": ("agent", "agent"),
        "reserve_agent_greedy": ("agent_greedy", "agent"),
        "reserve_agent_capped": ("agent_capped", "agent"),
        "reserve_tariff": ("tariff", "agent"),
    }
    kind, rates = placement[name]
    extra = {"tariff_settings": probe} if kind == "tariff" else {}
    return ReservePolicy(world, 13, mode="fixed", R_fixed=2, rates=rates,
                         placement=kind, **extra)


def run_one(job):
    scn, seed, policy_name, horizon, outdir, nuts = job
    path = os.path.join(outdir, scn, f"{policy_name}_s{seed}.pkl")
    if os.path.isfile(path):
        try:
            with open(path, "rb") as fh:
                saved = pickle.load(fh)
            if saved.get("meta", {}).get("horizon_days") == horizon:
                return (scn, seed, policy_name, saved["metrics"], saved.get("diagnostics", {}), -1.0, None)
        except Exception:
            pass

    started = time.time()
    try:
        cfg = scenario(scn, world_mode="matched", seed=seed,
                       horizon_days=horizon, **MAIN)
        from taris_sim.world import World
        from taris_sim.chain import Chain
        from taris_sim.data import DataStream
        from taris_sim.ledger import Ledger
        from scripts.run_tariff_placement import PROBE

        world = World(cfg)
        ledger = Ledger(world, Chain(world), DataStream(cfg, len(world.nodes)))
        if policy_name in RESERVE_POLICIES:
            policy = _make_reserve(policy_name, world, PROBE)
        elif policy_name == "k2_nuts13":
            from taris_sim.policy import K2NUTSPolicy
            policy = K2NUTSPolicy(world, 13, **nuts)
        else:
            policy = make_policy(policy_name, world)

        result = ledger.run(policy)
        metrics = result.metrics()
        metrics.update(seed=int(seed), scenario=scn, policy=policy_name,
                       runtime_seconds=round(time.time() - started, 2))
        diagnostics = {}
        for key in ("fit_count", "fit_seconds", "divergences", "max_rhat", "min_ess_bulk",
                    "sigma_node", "sigma_cell", "migration_candidates", "target_n",
                    "subnet_relaxations", "disqualified", "spill_count"):
            if hasattr(policy, key):
                value = getattr(policy, key)
                diagnostics[key] = float(value) if isinstance(value, (np.floating, float)) else int(value)
        if hasattr(policy, "placement_risk"):
            diagnostics["placement_risk"] = dict(policy.placement_risk)
        if hasattr(policy, "fits"):
            diagnostics["fit_records"] = len(policy.fits)
        if hasattr(policy, "tariff_values"):
            tv = policy.tariff_values
            diagnostics["tariff_eligible_share"] = (
                float(tv["financial"]["new_data_eligible"].mean()) if tv else None)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"meta": {"world": "MAIN", "seed": int(seed), "scenario": scn,
                                  "horizon_days": int(horizon), "policy": policy_name},
                         "metrics": metrics, "diagnostics": diagnostics,
                         "losses": result.losses, "repairs": result.repairs,
                         "daily": result.daily, "objects": result.objects}, fh)
        return (scn, seed, policy_name, metrics, diagnostics, time.time() - started, None)
    except Exception:
        return (scn, seed, policy_name, None, None, time.time() - started, traceback.format_exc())


def parse_seeds(value):
    if "-" in value:
        lo, hi = map(int, value.split("-", 1))
        return range(lo, hi + 1)
    return [int(x) for x in value.split(",") if x]


def main():
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--scenarios", default="S0,S2")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--horizon", type=int, default=1095)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--nuts-fit-interval", type=int, default=365)
    ap.add_argument("--nuts-draws", type=int, default=500)
    ap.add_argument("--nuts-tune", type=int, default=500)
    ap.add_argument("--nuts-chains", type=int, default=4)
    ap.add_argument("--nuts-target-accept", type=float, default=0.99)
    args = ap.parse_args()
    policies = [x for x in args.policies.split(",") if x]
    unknown = set(policies) - set(POLICIES)
    if unknown:
        ap.error(f"неизвестные политики: {', '.join(sorted(unknown))}")
    scenarios = [x for x in args.scenarios.split(",") if x]
    nuts = {"fit_interval": args.nuts_fit_interval, "draws": args.nuts_draws,
            "tune": args.nuts_tune, "chains": args.nuts_chains,
            "target_accept": args.nuts_target_accept}
    jobs = [(s, seed, p, args.horizon, args.out, nuts)
            for s in scenarios for seed in parse_seeds(args.seeds) for p in policies]
    os.makedirs(args.out, exist_ok=True)
    failures = 0
    print(f"MAIN; {len(jobs)} задач; S={','.join(scenarios)}; seeds={args.seeds}; "
          f"horizon={args.horizon}; workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            scn, seed, name, metrics, diag, elapsed, err = future.result()
            if err:
                failures += 1
                print(f"ERROR {done}/{len(jobs)} {scn} {name} s{seed} ({elapsed:.1f}s)\n{err}", flush=True)
            elif elapsed < 0:
                print(f"SKIP {done}/{len(jobs)} {scn} {name} s{seed}", flush=True)
            else:
                print(f"DONE {done}/{len(jobs)} {scn} {name} s{seed}: "
                      f"loss={metrics['Потеряно объектов']} "
                      f"traffic={metrics['Трафик восстановления в год']:.1f}/y "
                      f"({elapsed:.1f}s)", flush=True)
    print(f"Finished {len(jobs)} jobs; failures={failures}; output={args.out}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
