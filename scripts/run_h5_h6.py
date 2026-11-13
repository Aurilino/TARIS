"""Парная серия по протоколу docs/63_h5_h6_protocol.md; возобновляемые файлы по миру."""
import os
for _thread_setting in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_thread_setting] = "1"
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import time
import numpy as np
import pandas as pd
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import ReservePolicy
from taris_sim.protection import ProtectedReservePolicy
from taris_sim.graph import LinkGraph

REVISION = "h5h6-20261104-v1"
POLICIES = ("uniform", "agent", "protected", "novice_only", "no_novice", "no_penalty")


def code_hash():
    digest = hashlib.sha256()
    for path in ["scripts/run_h5_h6.py", *[f"taris_sim/{name}.py" for name in
                 ("config", "world", "chain", "data", "ledger", "policy", "risk", "experience", "rng", "graph", "adversary", "protection")]]:
        digest.update(path.encode())
        digest.update((ROOT / path).read_bytes())
    return digest.hexdigest()


def frame_hash(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def clone_leak(world, p):
    clone = copy.copy(world)
    clone.cfg = replace(world.cfg, adv_pair_leak=p)
    clone.adversary = copy.copy(world.adversary)
    clone.adversary.w = clone
    clone.adversary.cfg = clone.cfg
    clone.adversary.public_funding_pairs = []
    clone.adversary.funding_signals()
    return clone


def diagnostic(world, chain, t):
    pair = world.adversary.ownership_report(chain, t)
    graph = LinkGraph(world, chain)
    scores = graph.ownership(t)
    labels = graph.clusters(scores, graph.join <= t)
    nd = world.nodes
    selected = (graph.join <= t) & nd.owner.isin(world.adversary.coalition_owners).to_numpy()
    i, j = np.triu_indices(len(nd), 1)
    keep = selected[i] & selected[j]
    i, j = i[keep], j[keep]
    truth = nd.owner.to_numpy()[i] == nd.owner.to_numpy()[j]
    predicted = (labels[i] >= 0) & (labels[i] == labels[j])
    sp = nd.sybil.to_numpy()[i] & nd.sybil.to_numpy()[j]
    tp, fp = int((truth & predicted).sum()), int((~truth & predicted).sum())
    return {"pair": pair, "ownership_cluster": {
        "true_positive": tp, "false_positive": fp,
        "precision": tp/(tp+fp) if tp+fp else None,
        "recall": tp/int(truth.sum()) if truth.any() else None,
        "sybil_recall": float(predicted[sp].mean()) if sp.any() else None}}


def policy_of(name, world):
    common = dict(n=13, mode="fixed", R_fixed=2, rates="agent", placement="agent_greedy")
    if name == "uniform":
        return ReservePolicy(world, **dict(common, placement="uniform"))
    if name == "agent":
        return ReservePolicy(world, **common)
    options = {"protected": {}, "novice_only": dict(graph_cap=False, link_penalty=0.),
               "no_novice": dict(novice_cap=False), "no_penalty": dict(link_penalty=0.)}
    return ProtectedReservePolicy(world, **common, **options[name])


def metrics(result, policy, horizon):
    fr, ob = result.fragments, result.objects
    initial = np.zeros(len(ob), dtype=int)
    if len(fr):
        selected = fr.t_from.to_numpy() == ob.created_epoch.to_numpy()[fr.obj.to_numpy(dtype=int)]
        initial = np.bincount(fr.loc[selected, "obj"].to_numpy(dtype=int), minlength=len(ob))
    first_failed = int((ob.lost_t.to_numpy() == ob.created_epoch.to_numpy()).sum())
    allocated = float((np.minimum(fr.t_to.to_numpy(), horizon)-fr.t_from.to_numpy()).clip(min=0).sum())
    threat = result.adversary_report()
    return {"requested_objects": len(ob), "full_initial_sets": int((initial == 13).sum()),
            "initial_failures": first_failed, "total_losses": len(result.losses),
            "later_losses": len(result.losses)-first_failed, "short_placements": len(result.short_placements),
            "repair_traffic": float(result.repairs.traffic.sum()), "initial_write_traffic": float(initial.sum()/result.cfg.k),
            "allocated_fragment_days": allocated,
            "ciphertext_objects": len(threat["ciphertext_exposures"]),
            "plaintext_objects": threat["plaintext_disclosed_objects"],
            "losses_at_attack_time": threat["losses_at_attack_time"],
            "guard": getattr(policy, "guard_stats", None),
            "accepted_guard_sets": len(getattr(policy, "guard_log", ())),
            "fits": len(policy.fits)}


def run_case(task):
    group, seed, output, source_hash = task
    scen, share = {"main": ("S3", .3), "s4": ("S4", .3), "small": ("S3", .05)}[group]
    specification = dict(revision=REVISION, group=group, seed=seed, source_hash=source_hash)
    signature = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    target = Path(output) / f"{group}_{seed:02d}.json"
    if target.exists():
        saved = json.loads(target.read_text(encoding="utf-8"))
        if saved.get("signature") == signature:
            return str(target), "cached"
        raise ValueError(f"Результат с другим протоколом/кодом: {target}; выберите новый --out")
    cfg = scenario(scen, seed=seed, horizon_days=120, train_days=30, n_initial_nodes=120,
                   node_join_rate=.08, blocks_per_epoch=1, n_dc=30, n_providers=6, n_regions=5,
                   adv_sybil_share=share, adv_pair_leak=.3, adv_key_compromise_day=30.)
    world = World(cfg)
    chain = Chain(world)
    data = DataStream(cfg, len(world.nodes))
    data.objects = data.objects[data.objects.created_epoch >= 30].reset_index(drop=True)
    data.objects.index.name = "obj"
    table_fingerprint = {"nodes": frame_hash(world.nodes), "outages": frame_hash(world.outages),
                         "observed": frame_hash(chain.observed)}
    variants = [(name, .3) for name in (POLICIES if group == "main" else POLICIES[:3])]
    if group == "main":
        variants += [("protected", p) for p in (0., .1, .6)]
    runs = []
    for name, p in variants:
        w = clone_leak(world, p) if p != .3 else world
        assert frame_hash(w.nodes) == table_fingerprint["nodes"] and frame_hash(w.outages) == table_fingerprint["outages"]
        begin = time.perf_counter()
        policy = policy_of(name, w)
        result = Ledger(w, chain, data).run(policy)
        value = metrics(result, policy, cfg.horizon_days)
        value.update(policy=name, p_leak=p, seconds=time.perf_counter()-begin)
        runs.append(value)
        print(f"{group}/{seed}: {name} p={p:g}; потерь {value['total_losses']}, отказов при старте {value['initial_failures']}", flush=True)
    diagnostics = []
    for p in ((0., .1, .3, .6) if group == "main" else (.3,)):
        w = clone_leak(world, p) if p != .3 else world
        diagnostics.append(dict(p_leak=p, **diagnostic(w, chain, np.nextafter(world.adversary.attack_t, 0.))))
    assert frame_hash(world.nodes) == table_fingerprint["nodes"]
    record = dict(schema="taris-h5h6-case/1", signature=signature, specification=specification,
                  config=json.loads(cfg.to_json()), world_tables=table_fingerprint,
                  runs=runs, h6=diagnostics)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(target)
    return str(target), "computed"


def seeds(text):
    if "-" in text:
        start, end = map(int, text.split("-"))
        return list(range(start, end+1))
    return list(map(int, text.split(",")))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", default="main,s4,small")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("runs/h5_h6"))
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("workers должен быть >= 1")
    groups = args.groups.split(",")
    if any(g not in ("main", "s4", "small") for g in groups):
        ap.error("groups: main,s4,small")
    tasks = []
    source_hash = code_hash()
    for group in groups:
        for seed in (seeds(args.seeds) if args.seeds else range(1, 21 if group == "main" else 11)):
            tasks.append((group, seed, str(args.out), source_hash))
    if args.workers == 1:
        for task in tasks:
            print(run_case(task), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(run_case, task) for task in tasks]
            for future in as_completed(pending):
                print(future.result(), flush=True)


if __name__ == "__main__":
    main()
