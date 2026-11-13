"""Чувствительность cap-размещения к пределу числа фрагментов (S2, короткий пилот).

    python scripts/agent_cap_sensitivity.py [days=90] [seeds=1-3]
"""
from __future__ import annotations

import os
import pickle
import sys
import threading
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def run(seed: int, days: int, max_n: int, cluster_opt: bool = False, release: bool = False,
        release_min_credibility: float = 0.5, release_cap_fraction: float = 1.0):
    import numpy as np
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    from taris_sim.policy import ReservePolicy

    cfg = scenario("S2", seed=seed, horizon_days=days, **MAIN)
    world = World(cfg)
    ledger = Ledger(world, Chain(world), DataStream(cfg, len(world.nodes)))
    policy = ReservePolicy(world, 13, mode="fixed", R_fixed=2, rates="agent",
                           placement="agent_capped", placement_max_n=max_n,
                           placement_cluster_opt=cluster_opt, release_surplus=release,
                           release_min_credibility=release_min_credibility,
                           release_cap_fraction=release_cap_fraction)
    started = time.time()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            print(f"S2 seed={seed} max_n={max_n} elapsed={int(time.time() - started)}s", flush=True)

    pulse = threading.Thread(target=heartbeat, daemon=True)
    pulse.start()
    try:
        result = ledger.run(policy)
    finally:
        stop.set()
        pulse.join(timeout=1)
    row = {"seed": seed, "days": days, "max_n": max_n,
           "losses": int(result.metrics()["Потеряно объектов"]),
           "traffic_year": float(result.metrics()["Трафик восстановления в год"]),
           "max_placed_n": int(result.objects["n"].max()),
           "placement_risk": dict(policy.placement_risk),
           "release_stats": dict(policy.release_stats),
           "credibility": {"min": float(np.min(policy.model.Z)),
                           "median": float(np.median(policy.model.Z)),
                           "mean": float(np.mean(policy.model.Z))},
           "released": int(len(result.releases)),
           "released_fragment_days": float(result.metrics()["Высвобождено фрагменто-дней"]),
           "seconds": round(time.time() - started)}
    variant = (f"release_cred_{release_min_credibility:g}_cap_{release_cap_fraction:g}" if release else
               "cluster" if cluster_opt else "base")
    out_dir = os.path.join(ROOT, "runs", "agent_cap_sensitivity", str(days), variant)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"S2_s{seed}_max{max_n}.pkl"), "wb") as f:
        pickle.dump(row, f)
    return row


if __name__ == "__main__":
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    days = int(next((x for x in sys.argv[1:] if x.isdigit()), "90"))
    seed_range = next((x for x in sys.argv[1:] if "-" in x and x.split("-")[0].isdigit()), "1-3")
    cluster_opt = "--cluster-opt" in sys.argv
    release = "--release" in sys.argv
    cred_arg = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--release-cred=")), "0.5")
    release_min_credibility = float(cred_arg)
    cap_arg = next((x.split("=", 1)[1] for x in sys.argv if x.startswith("--release-cap=")), "1.0")
    release_cap_fraction = float(cap_arg)
    first, last = map(int, seed_range.split("-"))
    for seed in range(first, last + 1):
        caps = (18,) if cluster_opt or release else (13, 14, 16, 18)
        for max_n in caps:
            print(run(seed, days, max_n, cluster_opt, release, release_min_credibility,
                      release_cap_fraction), flush=True)
