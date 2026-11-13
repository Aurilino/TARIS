"""Р6а: малые теневые прогоны S0/S2 без изменения основной политики."""
import sys
sys.dont_write_bytecode = True

from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain

from graph_search import graph_search
from learning import NodeSpec, fit, select_strength
from run_examples import encode
from shadow_adapter import ChainObservationAdapter, public_slice
from uncertainty import build_ensembles


def digest_tables(world, chain):
    return {name: hashlib.sha256(table.to_json(orient="split", double_precision=15).encode()).hexdigest()
            for name, table in (("nodes", world.nodes), ("outages", world.outages),
                                ("silent", world.silent), ("observed", chain.observed), ("audits", chain.audits))}


def run_case(scenario_name, seed):
    cfg = scenario(scenario_name, seed=seed, horizon_days=24, train_days=16,
                   n_initial_nodes=24, node_join_rate=0.1, n_dc=4,
                   n_providers=2, n_regions=2, blocks_per_epoch=1, k=2)
    world, chain = World(cfg), None
    chain = Chain(world)
    before = digest_tables(world, chain)
    # Реестр: только уже зарегистрированные в начале ноды, публичное время и заявленная площадка.
    selected = world.nodes.index[world.nodes["join_t"] <= 0].tolist()[:6]
    if len(selected) != 6:
        raise ValueError("Для малой проверки нужны шесть начальных нод")
    names = {int(node): f"N{int(node):03d}" for node in selected}
    joins = {int(node): float(world.nodes.at[node, "join_t"]) for node in selected}
    dc_column = "dc_claimed" if "dc_claimed" in world.nodes else "dc"
    registry = tuple(NodeSpec(names[node], f"claimed_dc_{int(world.nodes.at[node, dc_column])}", 0) for node in selected)
    adapter = ChainObservationAdapter(registry)
    payloads = []
    for hour in range(24 * cfg.horizon_days):
        payload = public_slice(chain, names, joins, hour)
        adapter.ingest(payload)
        payloads.append(asdict(payload))
    cutoff = 16 * 24
    strength, validation = select_strength(adapter.store, registry, train_cutoff=8 * 24, validation_cutoff=cutoff)
    ensembles, metadata = build_ensembles(adapter.store, registry, cutoff,
        strength=strength, seed=12000 + seed, replicas=4, block_size=24)
    # Физическая потеря не наблюдается этим интерфейсом: не выдаём базовый слой за обучение.
    request = {"k": 2, "availability_limit": "0.08", "loss_limit": "1"}
    future = [row for row in adapter.store.known(24 * cfg.horizon_days) if row.time >= cutoff]
    results = {}
    for name in ("point", "bootstrap", "combined"):
        model = ensembles[name]
        result = graph_search(model, **request)
        exact = model.search(**request)
        if result.best:
            selected_names = set(result.best.nodes)
            unavailable = sum(sum(value is False for node, value in row.values if node in selected_names) < 2 for row in future)
            future_unavailability = Fraction(unavailable, len(future))
        else:
            future_unavailability = None
        results[name] = {"graph": asdict(result), "exact": asdict(exact),
                         "future_observed_unavailability": future_unavailability,
                         "future_integrity_loss_risk": None}
    after = digest_tables(world, chain)
    # Проверяется не только отсутствие модификаций объектов, но и повторяемость генерации мира.
    repeated_world = World(cfg)
    repeated_chain = Chain(repeated_world)
    return {"scenario": scenario_name, "seed": seed, "config_hash": cfg.hash(),
            "registry": [asdict(spec) for spec in registry], "observed_hours": adapter.hours,
            "fit_cutoff_hours": cutoff, "loss_labels": 0, "loss_constraint_enabled": False,
            "request": request,
            "selected_strength": strength, "validation": validation, "ensemble": metadata,
            "results": results, "observations": payloads,
            "world_tables_unchanged": before == after,
            "world_generation_reproducible": before == digest_tables(repeated_world, repeated_chain),
            "hashes": before}


def main():
    sources_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "taris_sim").glob("*.py")}
    cases = []
    for name, seed in (("S0", 3), ("S2", 4), ("S2", 5)):
        cases.append(run_case(name, seed))
        print(name, seed, "завершён", flush=True)
    sources_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / "taris_sim").glob("*.py")}
    report = {"schema": "taris-shadow-adapter/1", "main_source_unchanged": sources_before == sources_after,
              "cases": cases,
              "limitation": "Теневое чтение доступности. Меток физической потери нет; политика в Ledger не заменялась."}
    output = Path(__file__).with_name("shadow_adapter_results.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=encode) + "\n", encoding="utf-8")
    print("Исходники основной среды неизменны:", report["main_source_unchanged"])


if __name__ == "__main__":
    main()
