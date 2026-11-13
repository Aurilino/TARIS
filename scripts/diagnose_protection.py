"""Исследовательская диагностика неудачи защиты после просмотра результатов серии."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import numpy as np
from taris_sim.config import Config
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.graph import LinkGraph
from scripts.run_h5_h6 import clone_leak, policy_of, code_hash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=6)
    ap.add_argument("--p-leak", type=float, default=.6)
    ap.add_argument("--data", type=Path, default=ROOT / "runs/h5_h6")
    args = ap.parse_args()
    record = json.loads((args.data / f"main_{args.seed:02d}.json").read_text(encoding="utf-8"))
    if record["specification"]["source_hash"] != code_hash():
        raise ValueError("Код отличается от исходной серии")
    w = clone_leak(World(Config(**record["config"])), args.p_leak)
    ch = Chain(w)
    data = DataStream(w.cfg, len(w.nodes))
    data.objects = data.objects[data.objects.created_epoch >= 30].reset_index(drop=True)
    data.objects.index.name = "obj"
    policy = policy_of("protected", w)
    result = Ledger(w, ch, data).run(policy)
    graph = LinkGraph(w, ch)
    snapshots = {t: graph.snapshot(float(t)) for t in range(0, w.cfg.horizon_days, 30)}
    rows = []
    for obj, group in result.fragments.groupby("obj"):
        created = float(result.objects.at[obj, "created_epoch"])
        initial = group[group.t_from == created]
        nodes = initial.node.to_numpy(dtype=int)
        labels = snapshots[int(created//30)*30].clusters[nodes]
        valid = labels[labels >= 0]
        lost = float(result.objects.at[obj, "lost_t"])
        rows.append(dict(obj=int(obj), t=created,
                         sybil_fragments=int(w.nodes.sybil.to_numpy()[nodes].sum()),
                         max_group=int(np.unique(valid, return_counts=True)[1].max()) if len(valid) else 0,
                         initial_fragments=len(nodes), lost_t=lost if np.isfinite(lost) else None))
    scores = []
    for t, snapshot in snapshots.items():
        ids = np.flatnonzero(w.nodes.sybil.to_numpy() & (w.nodes.join_t.to_numpy() <= t))
        labels = snapshot.clusters[ids]
        pairs = len(ids)*(len(ids)-1)//2
        grouped = labels[labels >= 0]
        sizes = np.unique(grouped, return_counts=True)[1]
        found = int((sizes*(sizes-1)//2).sum())
        scores.append(dict(t=t, registered_sybil=len(ids), combined_cluster_recall=found/pairs if pairs else None))
    output = dict(schema="taris-protection-diagnostic/1", seed=args.seed, p_leak=args.p_leak,
                  source_hash=code_hash(), total_losses=len(result.losses),
                  losses_at_attack=int((result.losses.t == w.adversary.attack_t).sum()),
                  snapshots=scores, objects=rows, scope="post-hoc example; no parameter tuning")
    out = args.data / "diagnostics"
    out.mkdir(exist_ok=True)
    target = out / f"seed{args.seed:02d}_p{int(args.p_leak*10):02d}.json"
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(target)
    print(scores)


if __name__ == "__main__":
    main()
