"""Безопасность: события, накопление копий, реальные байты, CRN и старые метрики."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
from types import SimpleNamespace
import pandas as pd
from taris_sim.security import SecurityReplay, SecurityError, SCHEMA, validate
from crypto_demo.replay import verify_snapshot
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import UniformPolicy


def event(kind, t, **kw):
    return dict(kind=kind, t=t, **kw)


def doc(events=()):
    return {"schema": SCHEMA, "events": list(events)}


def fixture():
    columns = ["obj", "idx", "node", "t_from", "t_to", "t_dead"]
    fragments = pd.DataFrame([(0, 0, 0, 0., 2., 2.), (0, 0, 3, 2., float("inf"), float("inf")),
                              (0, 1, 1, 0., float("inf"), float("inf")),
                              (0, 2, 2, 0., float("inf"), float("inf"))], columns=columns)
    ledger = SimpleNamespace(w=SimpleNamespace(nodes=pd.DataFrame({"owner": [10, 11, 12, 10]})),
                             is_up=lambda node, t: not (node == 2 and 4. <= t < 5.))
    return SimpleNamespace(cfg=SimpleNamespace(k=2), fragments=fragments,
                           objects=pd.DataFrame({"created_epoch": [0]}),
                           daily=pd.DataFrame({"epoch": range(10)}), _L=ledger)


def expect_error(operation):
    try:
        operation()
    except SecurityError:
        return
    raise AssertionError("Ожидалась SecurityError")


def fingerprint(result):
    sha = hashlib.sha256()
    for name in ("daily", "fragments", "objects", "losses", "repairs", "disclosures", "migrations", "alarms"):
        sha.update(pd.util.hash_pandas_object(getattr(result, name), index=True).values.tobytes())
    return sha.hexdigest()


def main():
    checks = []
    def ok(text):
        checks.append(text)
        print("OK:", text, flush=True)
    result = fixture()
    events = [event("compromise_node", .5, attacker="A", node=0),
              event("compromise_node", 2.5, attacker="A", node=3),
              event("compromise_node", 3., attacker="A", node=1),
              event("compromise_manifest_key", 6., attacker="A", obj=0)]
    replay = SecurityReplay(result, doc(events))
    assert replay.query(0, 2.6)["attackers"][0]["copied_indices"] == [0]
    a = replay.query(0, 3.)["attackers"][0]
    assert a["ciphertext_collected"] and not a["plaintext_disclosed"]
    assert a["first_ciphertext_t"] == 3.
    a = replay.query(0, 6.)["attackers"][0]
    assert a["plaintext_disclosed"] and a["first_disclosure_t"] == 6.
    ok("копии переживают перенос; одинаковые индексы не удваиваются; ключ нужен для раскрытия")
    reverse = SecurityReplay(result, doc([event("compromise_object_key", .1, attacker="B", obj=0),
                                          event("compromise_owner", .5, attacker="B", owner=10),
                                          event("compromise_node", 3., attacker="B", node=1)]))
    assert not reverse.query(0, 2.)["attackers"][0]["plaintext_disclosed"]
    assert reverse.query(0, 3.)["attackers"][0]["first_disclosure_t"] == 3.
    ok("сначала ключ, затем фрагменты; компрометация владельца охватывает его узлы")
    late = SecurityReplay(result, doc([event("compromise_node", 2., attacker="C", node=0)]))
    assert late.query(0, 3.)["attackers"][0]["copied_indices"] == []
    ok("нет заднего копирования после конца жизни фрагмента")
    a, b = replay.query(0, 5.), replay.query(0, 1.)
    assert not b["attackers"][0]["ciphertext_collected"]
    assert replay.query(0, 5.) == a
    ok("перемотка запросов не переносит будущее в прошлое")

    client_events = [event("manifest_unavailable", 1., obj=0, until=2.),
                     event("client_key_unavailable", 2., obj=0, until=3.),
                     event("envelope_unavailable", 3., obj=0, until=4.),
                     event("deny_node", 4., node=0, until=5.),
                     event("corrupt_node", 4., node=1, until=5.),
                     event("deny_node", 4., node=3, until=5.)]
    outages = SecurityReplay(result, doc(client_events))
    for t, reason in [(1., "manifest_unavailable"), (2., "client_key_unavailable"),
                      (3., "envelope_unavailable"), (4., "insufficient_fragments")]:
        q = outages.query(0, t)
        assert not q["client_readable"] and reason in q["unavailability_reasons"]
    assert outages.query(0, 5.)["client_readable"]
    ok("раздельные причины недоступности; границы интервалов и сетевые простои")

    expect_error(lambda: validate(doc([event("deny_node", True, node=0, until=2)])))
    expect_error(lambda: validate(doc([event("deny_node", 1., node=0, until=1.)])))
    expect_error(lambda: SecurityReplay(result, doc([event("compromise_node", 1., attacker="x", node=999)])))
    expect_error(lambda: SecurityReplay(result, doc([event("compromise_object_key", 10., attacker="x", obj=0)])))
    expect_error(lambda: replay.query(0, 10.))
    expect_error(lambda: replay.query(999, 1.))
    expect_error(lambda: SecurityReplay(result, {}))
    changed = fixture()
    changed.fragments.loc[len(changed.fragments)] = [0, 3, 1, 5., float("inf"), float("inf")]
    expect_error(lambda: SecurityReplay(changed).query(0, 6.))
    ok("валидация событий и запросов; новый индекс требует нового манифеста")

    workspace = Path(__file__).resolve().parents[1] / "runs" / "security"
    proofs = []
    for t in (2.6, 6.):
        proofs.append(verify_snapshot(result, doc(events), 0, t, workspace))
    for t in (1., 2., 3., 4., 5.):
        proofs.append(verify_snapshot(result, doc(client_events), 0, t, workspace))
    ok("7 снимков проверены на настоящем AES-GCM/RS: клиент, противник, потеря ключа/манифеста, подмена")

    cfg = scenario("S0", seed=17, horizon_days=12, train_days=6, n_initial_nodes=50,
                   node_join_rate=.1, n_dc=10, n_providers=5, n_regions=3,
                   blocks_per_epoch=1, k=3)
    w = World(cfg)
    chain = Chain(w)
    ds = DataStream(cfg, len(w.nodes))
    ledger = Ledger(w, chain, ds)
    actual = ledger.run(UniformPolicy(5))
    before = fingerprint(actual)
    metric_before = actual.metrics()
    original_world = pd.util.hash_pandas_object(w.nodes, index=True).values.tobytes()
    selected = int(actual.objects.index[0])
    replay = SecurityReplay(actual)
    initial = actual.fragments[(actual.fragments.obj == selected) & (actual.fragments.t_from == 0)]
    nodes = [int(n) for n in initial.node[:cfg.k]]
    attack_events = [event("compromise_node", 0., attacker="sample", node=n) for n in nodes]
    attack_events.append(event("compromise_access_key", 2., attacker="sample"))
    sample_doc = doc(attack_events)
    q = [(selected, 1.), (selected, 3.)]
    report = actual.security_report(sample_doc, q)
    assert report["queries"][0]["attackers"][0]["ciphertext_collected"]
    assert not report["queries"][0]["attackers"][0]["plaintext_disclosed"]
    assert report["queries"][1]["attackers"][0]["plaintext_disclosed"]
    # Global compromise applies to later-created objects under the same unrotated access key.
    later = int(actual.objects.index[4])
    assert SecurityReplay(actual, sample_doc).query(later, 4.)["attackers"][0]["object_key_exposed"]
    assert fingerprint(actual) == before
    assert actual.metrics() == metric_before or canonical_metrics(actual.metrics()) == canonical_metrics(metric_before)
    assert pd.util.hash_pandas_object(w.nodes, index=True).values.tobytes() == original_world
    again = ledger.run(UniformPolicy(5))
    assert fingerprint(again) == before
    ok("реальный S0: старые таблицы/метрики и повторный прогон побитово совпали")
    proofs.append(verify_snapshot(actual, sample_doc, selected, 3., workspace))
    ok("снимок настоящего прогона TARIS согласован с байтовой демонстрацией")
    result_report = {"schema": "taris-security-check/1", "passed_groups": len(checks),
                     "checks": checks, "world": {"scenario": "S0", "seed": 17, "days": 12},
                     "table_fingerprint": before, "byte_proofs": proofs, "sample_report": report,
                     "scope": "read-only post-run; synthetic adversary interventions; no H5/H6 validation"}
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "check_report.json").write_text(json.dumps(result_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Отчёт:", workspace / "check_report.json")


def canonical_metrics(metrics):
    return json.dumps(metrics, sort_keys=True, ensure_ascii=False, default=str)


if __name__ == "__main__":
    main()
