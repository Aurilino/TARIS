"""Проверки противника и общих случайных чисел; --save-base снимает эталон."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from dataclasses import replace
import warnings
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import UniformPolicy
from taris_sim.graph import LinkGraph
from taris_sim.experience import Registry
from taris_sim.security import SecurityReplay
from taris_sim.export import export_run


def h(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def small(name, mode="matched", seed=19, **kw):
    args = dict(seed=seed, world_mode=mode, horizon_days=210 if name in ("S6", "S7", "S8") else 45,
                train_days=20, n_initial_nodes=45, node_join_rate=.04, n_dc=10,
                n_providers=4, n_regions=3, blocks_per_epoch=1, k=3, dev_ratio=.25)
    args.update(kw)
    return scenario(name, **args)


def regression():
    output = {}
    for name in ("S0", "S2", "S6", "S8"):
        for mode in ("matched", "mismatched"):
            w = World(small(name, mode))
            ch = Chain(w)
            ds = DataStream(w.cfg, len(w.nodes))
            result = Ledger(w, ch, ds).run(UniformPolicy(5))
            output[name + "/" + mode] = {
                "nodes": h(w.nodes), "beta0": repr(w.beta0),
                **{key: h(getattr(w, key)) for key in ("outages", "raw_own", "shocks", "silent")},
                **{key: h(getattr(ch, key)) for key in ("micro", "observed", "audits")},
                **{key: h(getattr(result, key)) for key in ("fragments", "daily", "losses", "repairs", "disclosures", "migrations")},
                "capacity": hashlib.sha256(ds.capacity.tobytes()).hexdigest()}
            print(name, mode, "готово", flush=True)
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-base", type=Path)
    ap.add_argument("--compare-base", type=Path,
                    default=Path(__file__).resolve().parent / "fixtures" / "adversary_base_before.json")
    ap.add_argument("--skip-regression", action="store_true", help="Только при уже пройденной регрессии без новых правок мира")
    args = ap.parse_args()
    if args.save_base:
        output = regression()
        args.save_base.parent.mkdir(parents=True, exist_ok=True)
        args.save_base.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print("Эталон:", args.save_base)
        return
    checks = []
    def ok(text):
        checks.append(text)
        print("OK:", text, flush=True)
    if args.skip_regression:
        print("Регрессия пропущена по явному флагу", flush=True)
    elif args.compare_base.exists():
        base = json.loads(args.compare_base.read_text(encoding="utf-8"))
        assert regression() == base, "Изменились сценарии без противника"
        ok("S0/S2/S6/S8, два режима: таблицы мира, наблюдения, ёмкости и результаты совпали с эталоном до правок")
    else:
        raise AssertionError("Нужен эталон, снятый до изменений: --compare-base <файл>")

    cfg = small("S4", seed=23, adv_sybil_share=.3, adv_pair_leak=.3,
                adv_whitewash_fraction=1., adv_rate_multiplier=2., adv_attack_day=38.,
                mean_event_rate=.12, weibull_scale_hours=30., adv_false_claim_fraction=1.,
                adv_bad_fraction=1.)
    w = World(cfg)
    assert w.adversary and w.adversary.whitewash and w.adversary.attack_nodes
    nd = w.nodes
    assert nd.loc[nd.sybil, "owner"].nunique() == 1
    assert nd.loc[nd.sybil, "wallet"].nunique() == int(nd.sybil.sum())
    assert nd.loc[nd.sybil, "dc"].nunique() > 1 and nd.loc[nd.sybil, "provider"].nunique() > 1
    assert 0 < w.adversary.actual_share < 1
    ok("Сибилла: один скрытый владелец, разные кошельки и инфраструктура, отдельная доля личностей")

    baseline = World(replace(cfg, adversary_enabled=False, cr_sybil_enabled=False))
    base_n = w.adversary.base_n
    assert w.beta0 == baseline.beta0
    shared = list(baseline.nodes.columns)
    assert nd.iloc[:base_n][shared].reset_index(drop=True).equals(baseline.nodes.reset_index(drop=True))
    assert w.raw_own[w.raw_own.node < base_n].reset_index(drop=True).equals(baseline.raw_own)
    pd.testing.assert_frame_equal(w.silent[w.silent.node < base_n].reset_index(drop=True),
                                  baseline.silent, check_dtype=False)
    ok("честные исходные узлы, калибровка и собственные потоки сохранены при добавлении Сибиллы")

    reg = Registry(w)
    assert np.array_equal(reg.dc, nd.dc_claimed.to_numpy())
    assert np.array_equal(reg.hw, nd.hw_claimed.to_numpy())
    assert (nd.loc[nd.sybil, "hw_claimed"] == int(np.argmin(cfg.beta_hw))).all()
    assert (nd.loc[nd.sybil, "lam_silent"] == cfg.silent_rate * cfg.adv_silent_multiplier).all()
    truth_columns = ["theta", "hw_true", "dc", "lin_pred", "risk_join_t", "alive_to_base"]
    honest_claims = World(replace(cfg, adv_false_claim_fraction=0.))
    assert nd[truth_columns].equals(honest_claims.nodes[truth_columns])
    ok("ложные заявления видны Registry; истинные риски и события от заявлений не меняются")

    for row in w.adversary.whitewash:
        parent, child = nd.loc[row["parent"]], nd.loc[row["node"]]
        assert child.owner == parent.owner and child.wallet != parent.wallet
        for key in ("theta", "hw_true", "lin_pred", "sev_u", "risk_join_t", "lam_silent"):
            assert child[key] == parent[key], key
        assert child.join_t >= parent.alive_to
        assert child.join_t == row["join_t"]
        assert parent.leave_reason == "whitewash"
    attack = nd.loc[w.adversary.attack_nodes]
    assert (attack.alive_to == 38.).all() and (attack.leave_reason == "sybil").all()
    assert not ((nd.sybil) & (nd.alive_from <= 38.) & (nd.alive_to > 38.)).any()
    ok("смена имени после полного случая сохраняет истинный риск; вся живая Сибилла уходит в заданный момент")

    ch = Chain(w)
    L = Ledger(w, ch, DataStream(cfg, len(nd)))
    result = L.run(UniformPolicy(5))
    report = result.adversary_report()
    observed = ch.observe(0., 39.)
    assert all("sybil" not in str(frame.columns) and "owner" not in str(frame.columns)
               for frame in observed.values() if isinstance(frame, pd.DataFrame))
    # Новая личность не наследует имя и регистрацию в истории агента.
    for row in w.adversary.whitewash:
        child = row["node"]
        assert not ((ch.observed.node == child) & (ch.observed.start < row["join_t"])).any()
    coalition = SecurityReplay(result, w.adversary.security_scenario(45.))
    counted = {r["obj"]: r["t"] for r in report["ciphertext_exposures"]}
    independent = {}
    for obj in range(len(result.objects)):
        query = coalition.query(obj, np.nextafter(45., 0.))
        states = query["attackers"]
        if states and states[0]["ciphertext_collected"]:
            independent[obj] = states[0]["first_ciphertext_t"]
        assert not any(a["plaintext_disclosed"] for a in states)
    assert counted == independent
    assert report["plaintext_disclosed_objects"] == 0
    # Утечка ключа независима от физического мира и превращает накопленный шифротекст в раскрытие.
    old_key_time = w.adversary.cfg
    w.adversary.cfg = replace(cfg, adv_key_compromise_day=10.)
    key_report = w.adversary.report(result)
    key_replay = SecurityReplay(result, w.adversary.security_scenario(45.))
    independently_disclosed = {obj for obj in range(len(result.objects))
                              if key_replay.query(obj, np.nextafter(45., 0.))["attackers"][0]["plaintext_disclosed"]}
    assert {r["obj"] for r in key_report["plaintext_exposures"]} == independently_disclosed
    assert key_report["plaintext_disclosed_objects"] == len(counted) > 0
    w.adversary.cfg = old_key_time
    first_frags = h(result.fragments)
    repeat = L.run(UniformPolicy(5))
    assert h(repeat.fragments) == first_frags and repeat.adversary_report() == report
    ok("коалиция накапливает разные индексы; независимый SecurityReplay совпал; повтор политики детерминирован")

    hidden = World(replace(cfg, adv_pair_leak=0.))
    exposed = World(replace(cfg, adv_pair_leak=1.))
    assert not hidden.adversary.public_funding_pairs
    members = list(exposed.nodes.index[exposed.nodes.sybil])
    assert len(exposed.adversary.public_funding_pairs) == len(members)*(len(members)-1)//2
    assert h(hidden.nodes) == h(exposed.nodes) and h(hidden.outages) == h(exposed.outages)
    graph0 = LinkGraph(hidden, Chain(hidden)).ownership(37.)
    graph1 = LinkGraph(exposed, Chain(exposed)).ownership(37.)
    valid = [i for i in members if exposed.nodes.at[i, "join_t"] <= 37.]
    for i in valid:
        for j in valid:
            if i != j:
                assert graph1[i, j] >= cfg.graph_b_wallet and graph0[i, j] < cfg.graph_b_wallet
    ok("p_leak=0/1: сигналы независимы от мира, граф использует только публичную связь")
    diagnostics = exposed.adversary.ownership_report(Chain(exposed), 37.)
    assert diagnostics["sybil_recall"] == 1.

    out = Path("runs/adversary")
    out.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        exported = export_run(w, ch, result, out / "check_log.json.gz", scenario="S4", graph=False)
    assert "adversary" in exported["truth"]
    assert "sybil" not in exported["nodes"]
    assert "whitewash_parent" not in exported["nodes"]
    assert exported["nodes"]["dc"] == list(map(int, nd.dc_claimed))
    assert "funding_pairs" in exported["net"]
    assert not any(k.startswith("adv_") for k in exported["meta"]["config"])
    assert "adv_attack_day" in exported["truth"]["adversary"]["parameters"]
    ok("экспорт сохраняет план и роли только в truth; сеть видит заявления и сигналы")

    w7 = World(small("S7", seed=19, adv_sybil_share=.2))
    c7 = Chain(w7)
    r7 = Ledger(w7, c7, DataStream(w7.cfg, len(w7.nodes))).run(UniformPolicy(5))
    assert w7.crisis is not None and w7.adversary.attack_nodes and r7.adversary_report()
    assert all(w7.nodes.at[i, "alive_to"] == w7.adversary.attack_t for i in w7.adversary.attack_nodes)
    ok("S7: библиотека кризисов и противник работают совместно")
    # Флаги проверяются до генерации противника.
    for bad in (dict(adv_sybil_share=1.), dict(adv_pair_leak=-.1), dict(adv_attack_day=1000.),
                dict(adv_whitewash_gap_days=float("nan"))):
        try:
            World(replace(cfg, **bad))
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    ok("неверные параметры противника отклоняются")
    summary = {"schema": "taris-adversary-check/1", "passed_groups": len(checks), "checks": checks,
               "fixture_config": json.loads(cfg.to_json()), "sample": report,
               "scope": "functional checks; no calibrated H5/H6 claim"}
    (out / "check_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Отчёт:", out / "check_report.json")


if __name__ == "__main__":
    main()
