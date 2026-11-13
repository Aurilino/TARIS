"""Парный сравнительный прогон baseline-политик шага 4.

Пример: python scripts/run_step4.py --workers 4 --seeds 1-3 --scenarios S0,S2
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.scenario import make_policy


DEFAULT_POLICIES = ("uniform13", "crush13", "copysets13", "filecoin13", "sia13", "storj13",
                    "dynamic_ec13", "aec13", "gbm13", "k1_pg13", "k2_laplace13", "oracle13")


def _one(job):
    policy_name, scn, seed, outdir, nuts = job
    t0 = time.time()
    path = os.path.join(outdir, scn, f"{policy_name}_s{seed}.pkl")
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                obj = pickle.load(fh)
            return (policy_name, scn, seed, obj.get("metrics"), obj.get("diagnostics", {}), -1.0, None)
        except Exception:
            pass
    try:
        cfg = scenario(scn, world_mode="matched", seed=seed)
        world = World(cfg)
        chain = Chain(world)
        data = DataStream(cfg, len(world.nodes))
        ledger = Ledger(world, chain, data)
        if policy_name == "k2_nuts13":
            from taris_sim.policy import K2NUTSPolicy
            policy = K2NUTSPolicy(world, 13, **nuts)
        else:
            policy = make_policy(policy_name, world)
        result = ledger.run(policy)
        metrics = result.metrics()
        metrics.update(seed=int(seed), scenario=scn, policy=policy_name,
                       runtime_seconds=round(time.time() - t0, 2))
        diag = {}
        for key in ("spill_count", "faults", "slashed", "empty_eligible_fallbacks", "audit_fail_rate",
                    "target_n", "fit_rows", "fit_groups", "audit_count", "migration_candidates",
                    "subnet_relaxations", "disqualified", "fit_groups", "sigma_node", "sigma_cell",
                    "fit_count", "fit_seconds", "divergences", "max_rhat", "min_ess_bulk",
                    "rhat_beta", "rhat_scales", "rhat_node", "ess_node_bulk"):
            if hasattr(policy, key):
                value = getattr(policy, key)
                if isinstance(value, np.ndarray):
                    diag[key] = float(value.sum())
                else:
                    diag[key] = float(value) if isinstance(value, (np.floating, float)) else int(value)
        if hasattr(policy, "copysets"):
            diag.update(copyset_count=len(policy.copysets), pool_size=policy.pool_size,
                        spill_count=policy.spill_count)
        if hasattr(policy, "exhausted_owners"):
            diag["exhausted_owners"] = len(policy.exhausted_owners)
        if hasattr(policy, "audit_history"):
            history = policy.audit_history
            diag["audit_observations"] = (sum(len(x) for x in history) if history and isinstance(history[0], list)
                                          else len(history))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"metrics": metrics, "losses": result.losses, "repairs": result.repairs,
                         "daily": result.daily, "objects": result.objects,
                         "diagnostics": diag}, fh)
        return (policy_name, scn, seed, metrics, diag, time.time() - t0, None)
    except Exception:
        return (policy_name, scn, seed, None, None, time.time() - t0, traceback.format_exc())


def _parse_seeds(value):
    if "-" in value:
        a, b = map(int, value.split("-", 1))
        return range(a, b + 1)
    return [int(x) for x in value.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", default="1-3")
    ap.add_argument("--scenarios", default="S0,S2")
    ap.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    ap.add_argument("--out", default="runs/step4")
    ap.add_argument("--nuts-fit-interval", type=int, default=365)
    ap.add_argument("--nuts-draws", type=int, default=500)
    ap.add_argument("--nuts-tune", type=int, default=500)
    ap.add_argument("--nuts-chains", type=int, default=4)
    ap.add_argument("--nuts-target-accept", type=float, default=0.99)
    args = ap.parse_args()
    policies = [x for x in args.policies.split(",") if x]
    scenarios = [x for x in args.scenarios.split(",") if x]
    nuts = {"fit_interval": args.nuts_fit_interval, "draws": args.nuts_draws,
            "tune": args.nuts_tune, "chains": args.nuts_chains,
            "target_accept": args.nuts_target_accept}
    jobs = [(p, s, seed, args.out, nuts) for s in scenarios for seed in _parse_seeds(args.seeds) for p in policies]
    os.makedirs(args.out, exist_ok=True)
    failed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            name, scn, seed, metrics, diag, elapsed, err = future.result()
            if err:
                failed += 1
                print(f"ERROR {done}/{len(jobs)} {scn} {name} seed={seed} ({elapsed:.1f}s)\n{err}", flush=True)
            elif elapsed < 0:
                print(f"SKIP {done}/{len(jobs)} {scn} {name} seed={seed} (saved result exists)", flush=True)
            else:
                print(f"DONE {done}/{len(jobs)} {scn} {name} seed={seed} "
                      f"loss={metrics.get('Потеряно объектов', 'n/a')} "
                      f"traffic={metrics.get('Трафик восстановления в год', 'n/a')} diag={diag} ({elapsed:.1f}s)", flush=True)
    print(f"Finished {len(jobs)} runs; failures={failed}; output={args.out}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
