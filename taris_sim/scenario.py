"""Файл сценария и его исполнение (визуализация, этап 6; шаг 3, пункт 3.8 — вмешательства).

Формат taris-scenario/1 (JSON, время — в сутках от начала горизонта):

{
  "format": "taris-scenario/1",
  "name": "Пожар в ДЦ 4",
  "description": "...",
  "base": {"scenario": "S6", "world_mode": "matched", "seed": 1, "world": "main",
           "policy": "crisis_oracle13", "overrides": {}},
  "interventions": [
    {"kind": "crisis", "type": "fire", "cluster": 3, "t": 400},
    {"kind": "crisis", "type": "storm", "cluster": 5, "t": 250, "predictable": true, "W_days": 4},
    {"kind": "shock", "ctype": "dc", "cluster": 7, "t": 300, "hours": 30, "destroyed": false},
    {"kind": "node_down", "node": 12, "t": 500, "hours": 48}
  ]
}

Кризис-вмешательство проходит через тот же механизм, что случайные кризисы (предвестник, исход, каскады);
шок и выключение узла — простои заданной длительности. Случайные числа вмешательств — из отдельного потока,
поэтому остальные события мира не меняются (общие случайные числа).
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import scenario as base_scenario
from .crisis import MANUAL_TYPES

FORMAT = "taris-scenario/1"
# Основной мир просмотрщика и пилотных прогонов: 1 000 стартовых узлов. Вместе с сетью уменьшены частота регистраций
# и поток блоков (ёмкость узлов). Дата-центров не меньше n = 13 (оракул кладёт в дата-центр один фрагмент,
# а второй — только в крайнем случае), провайдеров и регионов — столько, чтобы при лимите r = 3 набиралось 13 узлов.
# Больше дата-центров — не нейтрально: частота бурь и пожаров задана на один дата-центр.
# WORLDS: мир -> (правки параметров, интервал агрегации журнала в сутках)
MAIN =dict(n_initial_nodes=1000, node_join_rate=0.12, blocks_per_epoch=4, n_dc=30, n_providers=6, n_regions=5)
WORLDS = {"main": (MAIN, 1), "full": ({}, 7)}
POLICIES = ("uniform15", "uniform13", "crush13", "copysets13", "filecoin13", "sia13", "storj13", "dynamic_ec13", "aec13", "gbm13", "k1_pg13", "k2_laplace13", "k2_nuts13", "oracle13", "crisis_oracle13")


def load(path) -> dict:
    sc = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate(sc)
    if errors:
        raise ValueError("Файл сценария с ошибками:\n- " + "\n- ".join(errors))
    return sc


def config_of(sc: dict):
    b = sc["base"]
    over, _ = WORLDS[b.get("world", "main")]
    return base_scenario(b.get("scenario", "S6"), world_mode=b.get("world_mode", "matched"), seed=int(b.get("seed", 1)),
                         **{**over, **b.get("overrides", {})})


def validate(sc: dict) -> list[str]:
    """Список ошибок файла сценария (пустой — файл годен)."""
    e = []
    if sc.get("format") != FORMAT:
        e.append(f"format должен быть {FORMAT}")
    b = sc.get("base", {})
    if b.get("world", "main") not in WORLDS:
        e.append(f"base.world: {b.get('world')} — нужно main или full")
    if b.get("policy", "crisis_oracle13") not in POLICIES:
        e.append(f"base.policy: {b.get('policy')} — нужно одно из {', '.join(POLICIES)}")
    if b.get("world_mode", "matched") not in ("matched", "mismatched"):
        e.append("base.world_mode: нужно matched или mismatched")
    try:
        c = config_of(sc)
    except Exception as err:                      # неизвестный сценарий или параметр
        return e + [f"base: {err}"]
    H = c.horizon_days
    for k, it in enumerate(sc.get("interventions", [])):
        where = f"вмешательство {k + 1}"
        kind, t = it.get("kind"), it.get("t")
        if not isinstance(t, (int, float)) or not 0 <= t < H:
            e.append(f"{where}: t — момент в сутках от 0 до {H}")
        if kind == "crisis":
            typ, cl = it.get("type"), it.get("cluster")
            if typ not in MANUAL_TYPES:
                e.append(f"{where}: type — одно из {', '.join(MANUAL_TYPES)}")
            elif typ in ("storm", "fire") and not (isinstance(cl, int) and 0 <= cl < c.n_dc):
                e.append(f"{where}: cluster — номер дата-центра от 0 до {c.n_dc - 1}")
            elif typ == "exodus" and not (isinstance(cl, int) and 0 <= cl < c.n_regions):
                e.append(f"{where}: cluster — номер региона от 0 до {c.n_regions - 1}")
            elif typ == "wear" and not (isinstance(cl, str) and "/" in cl and cl.split("/")[1] in c.hw_classes):
                e.append(f"{where}: cluster — партия «ДЦ/класс», класс из {', '.join(c.hw_classes)}")
        elif kind == "shock":
            n = {"dc": c.n_dc, "provider": c.n_providers, "region": c.n_regions}.get(it.get("ctype"))
            if n is None:
                e.append(f"{where}: ctype — dc, provider или region")
            elif not (isinstance(it.get("cluster"), int) and 0 <= it["cluster"] < n):
                e.append(f"{where}: cluster — номер от 0 до {n - 1}")
            if not isinstance(it.get("hours"), (int, float)) or it["hours"] <= 0:
                e.append(f"{where}: hours — длительность в часах больше 0")
        elif kind == "node_down":
            if not isinstance(it.get("node"), int) or it["node"] < 0:
                e.append(f"{where}: node — номер узла")
            if not isinstance(it.get("hours"), (int, float)) or it["hours"] <= 0:
                e.append(f"{where}: hours — длительность в часах больше 0")
        else:
            e.append(f"{where}: kind — crisis, shock или node_down")
    return e


def make_policy(name: str, world):
    from .policy import (UniformPolicy, CrushLikePolicy, CopysetsPolicy,
                         FilecoinInspiredPolicy, SiaInspiredPolicy, StorjInspiredPolicy,
                         DynamicECPolicy, AECInspiredPolicy, GradientBoostingPolicy,
                         K1PoissonGammaPolicy, K2LaplacePolicy, K2NUTSPolicy,
                         OraclePolicy, CrisisOraclePolicy)
    return {"uniform15": lambda: UniformPolicy(15), "uniform13": lambda: UniformPolicy(13),
            "crush13": lambda: CrushLikePolicy(world, 13),
            "copysets13": lambda: CopysetsPolicy(world, 13),
            "filecoin13": lambda: FilecoinInspiredPolicy(world, 13),
            "sia13": lambda: SiaInspiredPolicy(world, 13),
            "storj13": lambda: StorjInspiredPolicy(world, 13),
            "dynamic_ec13": lambda: DynamicECPolicy(world),
            "aec13": lambda: AECInspiredPolicy(world, 13),
            "gbm13": lambda: GradientBoostingPolicy(world, 13),
            "k1_pg13": lambda: K1PoissonGammaPolicy(world, 13),
            "k2_laplace13": lambda: K2LaplacePolicy(world, 13),
            "k2_nuts13": lambda: K2NUTSPolicy(world, 13),
            "oracle13": lambda: OraclePolicy(world, 13), "crisis_oracle13": lambda: CrisisOraclePolicy(world, 13)}[name]()


def empty_targets(w) -> list[str]:
    """Предупреждения о вмешательствах, которые ничего не задели: файл годен, но цель в момент вмешательства
    пуста (кризис по пустой партии или кластеру, шок кластера без живых узлов, выключение мёртвого узла).
    Проверка файла (validate) мир не строит, поэтому это видно только после построения мира."""
    out, manual = [], []
    if w.crisis is not None:
        manual = [e for e in w.crisis.episodes if e.get("manual")]
    k_cr = 0
    for k, it in enumerate(w.interventions):
        if it["kind"] == "crisis":
            n = len(manual[k_cr]["targets"]) if k_cr < len(manual) else 0
            k_cr += 1
            what = f"кризис {it['type']}, кластер {it.get('cluster')}"
        else:
            n = int(it.get("n_hit", 0))
            what = f"шок {it['ctype']} {it['cluster']}" if it["kind"] == "shock" else f"выключение узла {it['node']}"
        if n == 0:
            out.append(f"вмешательство {k + 1} ({what}, t={it['t']}): в цели 0 живых узлов — оно ничего не изменит")
    return out


def run(sc: dict):
    """Мир со вмешательствами, цепочка и прогон политики."""
    from .world import World
    from .chain import Chain
    from .data import DataStream
    from .ledger import Ledger
    cfg = config_of(sc)
    w = World(cfg, interventions=sc.get("interventions", []))
    ch = Chain(w)
    L = Ledger(w, ch, DataStream(cfg, len(w.nodes)))
    res = L.run(make_policy(sc["base"].get("policy", "crisis_oracle13"), w))
    return w, ch, res
