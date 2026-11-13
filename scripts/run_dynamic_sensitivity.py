"""Чувствительность Dynamic-EC к двум порогам за последние 30 суток.

Пример: python scripts/run_dynamic_sensitivity.py --workers 4 --seeds 1-10
Сценарии и миры совпадают с runs/step4; параметры мира не переопределяются.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import DynamicECPolicy


THRESHOLDS = {
    "low": (0.01, 0.03),
    "base": (0.02, 0.05),
    "high": (0.04, 0.08),
}


def _seeds(value):
    if "-" in value:
        first, last = map(int, value.split("-", 1))
        return range(first, last + 1)
    return [int(x) for x in value.split(",") if x]


def _one(job):
    scenario_name, seed, variant, outdir = job
    path = os.path.join(outdir, scenario_name, f"{variant}_s{seed}.pkl")
    if os.path.exists(path):
        return scenario_name, seed, variant, -1.0, None
    started = time.time()
    try:
        cfg = scenario(scenario_name, world_mode="matched", seed=seed)
        world = World(cfg)
        chain = Chain(world)
        data = DataStream(cfg, len(world.nodes))
        ledger = Ledger(world, chain, data)
        mid, high = THRESHOLDS[variant]
        policy = DynamicECPolicy(world, medium_threshold=mid, high_threshold=high)
        result = ledger.run(policy)
        metrics = result.metrics()
        metrics.update(seed=seed, scenario=scenario_name, variant=variant,
                       medium_threshold=mid, high_threshold=high,
                       runtime_seconds=round(time.time() - started, 2))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({"metrics": metrics}, fh)
        return scenario_name, seed, variant, time.time() - started, None
    except Exception:
        return scenario_name, seed, variant, time.time() - started, traceback.format_exc()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", default="1-10")
    parser.add_argument("--scenarios", default="S0,S2")
    parser.add_argument("--out", default="runs/step4_dynamic_sensitivity")
    args = parser.parse_args()
    jobs = [(s, seed, variant, args.out)
            for s in args.scenarios.split(",") for seed in _seeds(args.seeds)
            for variant in THRESHOLDS]
    failed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(_one, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            scn, seed, variant, elapsed, error = future.result()
            if error:
                failed += 1
                print(f"ERROR {i}/{len(jobs)} {scn}/{variant}/s{seed} ({elapsed:.1f}s)\n{error}", flush=True)
            elif elapsed < 0:
                print(f"SKIP {i}/{len(jobs)} {scn}/{variant}/s{seed}", flush=True)
            else:
                print(f"DONE {i}/{len(jobs)} {scn}/{variant}/s{seed} ({elapsed:.1f}s)", flush=True)
    print(f"Finished {len(jobs)} runs; failures={failed}; output={args.out}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
