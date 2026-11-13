"""Paired simulator sensitivity for a hypothetical disk-to-node outage bridge.

All hardware outages are injected into a temporary in-memory World copy after
the ordinary world has been generated. Config and base world streams are not
changed. This is an illustrative sensitivity, not a Backblaze calibration.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from taris_sim.config import scenario
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import OraclePolicy, UniformPolicy
from taris_sim.rng import stream
from taris_sim.world import World
from taris_sim.chain import Chain

AFR = 0.01115521
HW_STREAM = 17
VARIANTS = (("base", 0.0, 0.0), ("q10_r6h", 0.10, 6.0),
            ("q10_r48h", 0.10, 48.0), ("q100_r6h", 1.0, 6.0),
            ("q100_r48h", 1.0, 48.0))


def add_hardware_outages(world: World, bridge: float, repair_hours: float, epochs: int) -> int:
    if bridge == 0:
        return 0
    nd = world.nodes
    rows = []
    for i, node in nd.iterrows():
        if bool(node.get("is_device", False)):
            continue
        start, end = float(node.alive_from), min(float(node.alive_to_base), float(epochs))
        if end <= start:
            continue
        rng = stream(world.cfg.seed, HW_STREAM, int(i))
        count = int(rng.poisson(AFR * bridge * (end - start) / 365.25))
        for t in np.sort(rng.uniform(start, end, count)):
            rows.append((int(i), float(t), min(float(t + repair_hours / 24.0), end),
                         "own", -1, False))
    if not rows:
        return 0
    add = pd.DataFrame(rows, columns=world.outages.columns[:6])
    out = World._merge_overlaps(pd.concat([world.outages, add], ignore_index=True))
    duration = out["end"].to_numpy() - out["start"].to_numpy()
    out["cls"] = np.where(out["destroyed"].to_numpy() | (duration > world.cfg.T_rep), "total", "partial")
    world.outages = out.reset_index(drop=True)
    return len(rows)


def run_one(seed: int, variant: str, bridge: float, repair_hours: float, policy_name: str, epochs: int) -> dict:
    cfg = scenario("S0", seed=seed)
    world = World(cfg)
    added = add_hardware_outages(world, bridge, repair_hours, epochs)
    chain = Chain(world)
    data = DataStream(cfg, len(world.nodes))
    ledger = Ledger(world, chain, data)
    policy = OraclePolicy(world, 13) if policy_name == "oracle13" else UniformPolicy(13)
    result = ledger.run(policy, epochs=epochs)
    traffic = float(result.repairs["traffic"].sum()) if len(result.repairs) else 0.0
    return {"seed": seed, "variant": variant, "policy": policy_name,
            "bridge_probability": bridge, "repair_hours_assumed": repair_hours,
            "injected_hardware_events": added,
            "simulated_days": epochs,
            "lost_objects": int(len(result.losses)),
            "repair_traffic_per_365_days": traffic * 365.0 / epochs,
            "short_placements": int(len(result.short_placements)),
            "base_config_hash": cfg.hash()}


def parse_seeds(raw: str) -> list[int]:
    if "-" in raw:
        a, b = map(int, raw.split("-", 1))
        if a < 0 or b < a:
            raise ValueError("seed range must be nonnegative and ascending")
        return list(range(a, b + 1))
    return [int(x) for x in raw.split(",")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seeds", nargs="?", default="1", help="e.g. 1-10 or 1,3,5")
    parser.add_argument("--out", type=Path, default=Path("runs/real_data/backblaze_q4_2025/simulator_sensitivity.csv"))
    parser.add_argument("--epochs", type=int, default=100,
                        help="number of days to simulate on the unchanged full world (default: 100)")
    parser.add_argument("--variants", nargs="+", choices=tuple(x[0] for x in VARIANTS),
                        default=("base", "q100_r6h", "q100_r48h"))
    parser.add_argument("--policies", nargs="+", choices=("uniform13", "oracle13"), default=("uniform13",))
    args = parser.parse_args()
    seeds = parse_seeds(args.seeds)
    if not 1 <= args.epochs <= scenario("S0").horizon_days:
        parser.error("epochs must be within the configured world horizon")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "variant", "policy", "bridge_probability", "repair_hours_assumed",
              "simulated_days", "injected_hardware_events", "lost_objects", "repair_traffic_per_365_days",
              "short_placements", "base_config_hash"]
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()
        for seed in seeds:
            for variant, bridge, hours in VARIANTS:
                if variant not in args.variants:
                    continue
                for policy in args.policies:
                    print(f"starting seed={seed} {variant} {policy} ({args.epochs} days)", flush=True)
                    row = run_one(seed, variant, bridge, hours, policy, args.epochs)
                    writer.writerow(row); f.flush()
                    print(f"seed={seed} {variant} {policy}: injected={row['injected_hardware_events']} "
                          f"losses={row['lost_objects']} traffic/365d={row['repair_traffic_per_365_days']:.3f}",
                          flush=True)
    print(f"rows={len(seeds) * len(args.variants) * len(args.policies)} output={args.out}")


if __name__ == "__main__":
    main()
