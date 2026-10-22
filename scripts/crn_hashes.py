"""Хэши таблиц мира и наблюдений — проверка общих случайных чисел при правках кода.

Хэш конфигурации сюда не входит: он меняется при добавлении новых параметров,
а сами события мира меняться не должны.

    python scripts/crn_hashes.py --save runs/crn_base.json      # снять эталон
    python scripts/crn_hashes.py --compare runs/crn_base.json   # сверить с эталоном
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain

NODE_COLS = ["owner", "join_t", "leave_t", "alive_from", "alive_to", "dc", "hw_true", "theta", "lam_avg", "lam_micro"]


def h(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def table_hashes(name: str, mode: str, seed: int) -> dict:
    w = World(scenario(name, world_mode=mode, seed=seed))
    ch = Chain(w)
    return {
        "nodes": h(w.nodes[NODE_COLS]),
        "beta0": repr(w.beta0),
        "outages": h(w.outages),
        "raw_own": h(w.raw_own),
        "shocks": h(w.shocks),
        "silent": h(w.silent),
        "micro": h(ch.micro),
        "observed": h(ch.observed),
        "audits": h(ch.audits),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save")
    ap.add_argument("--compare")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    res = {}
    for name in ("S0", "S2"):
        for mode in ("matched", "mismatched"):
            key = f"{name}/{mode}"
            res[key] = table_hashes(name, mode, a.seed)
            print(key, "готово", flush=True)
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print("Эталон сохранён:", a.save)
    if a.compare:
        base = json.loads(Path(a.compare).read_text(encoding="utf-8"))
        bad = [(k, t) for k in base for t in base[k] if base[k][t] != res[k].get(t)]
        for k, t in bad:
            print(f"РАСХОЖДЕНИЕ {k}: {t}  {base[k][t]} -> {res[k].get(t)}")
        print("Все таблицы совпадают побитово: OK" if not bad else f"Расхождений: {len(bad)}")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
