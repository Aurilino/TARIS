"""Проверки журнала прогона (визуализация, этап 1.3; схема — docs/42_log_schema.md).

1. Соответствие схеме: разделы и столбцы на месте, длины столбцов таблицы совпадают, номера перечислений
   в пределах словарей.
2. Сверка с прогоном: потери, раскрытия, тревоги, кризисы, наблюдаемые и истинные простои, уходы и
   регистрации совпадают с таблицами симулятора; суммы сводок совпадают с числом событий.
3. «Взгляд сети»: в разделах кроме truth нет полей истины.
4. Воспроизводимость: два экспорта одного прогона побитово равны.

    python scripts/check_export.py            # основной мир (около 200 узлов), оракул кризисов, 1 год
"""
import sys, os, filecmp, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import CrisisOraclePolicy
from taris_sim.export import export_run, load, SCHEMA

MAIN = dict(n_initial_nodes=200, node_join_rate=0.12, blocks_per_epoch=4, n_dc=16, n_providers=6, n_regions=5)

# столбцы каждой таблицы по схеме (раздел -> таблица -> столбцы)
TABLES = {
    "world": {"regions": ["id"], "providers": ["id"], "dcs": ["id", "provider", "region"]},
    "net": {"outages": ["node", "start", "end", "cls"], "micro": ["node", "bin0", "counts"],
            "audits": ["node", "bin0", "fails"], "versions": ["version", "t"], "updates": ["node", "version", "t"]},
    "policy": {"alarms": ["t", "type", "ctype", "cluster", "prob", "horizon"],
               "alarm_episodes": ["type", "ctype", "cluster", "t0", "until", "n"],
               "repairs": ["obj", "t", "r", "traffic"], "migrations": ["obj", "idx", "from", "to", "t", "t_done", "ok"]},
    "objects": {"sample": ["obj", "created", "n", "lost"], "fragments": ["obj", "idx", "node", "from", "to", "end"]},
    "outcomes": {"losses": ["obj", "t"]},
    "truth": {"nodes": ["id", "owner", "hw_true", "lam_year", "phi", "leave", "leave_reason", "cancelled"],
              "outages": ["node", "start", "end", "kind", "destroyed"], "micro": ["node", "bin0", "counts"],
              "pre_micro": ["node", "bin0", "counts"], "shocks": ["ctype", "cluster", "start", "end", "destroyed", "n_nodes"],
              "crises": ["id", "type", "ctype", "cluster", "tau", "W", "w0", "end", "predictable", "false_alarm", "parent",
                         "n_targets", "n_hit", "destroyed"],
              "crisis_targets": ["crisis", "node"], "precursors": ["crisis", "node", "start", "end"],
              "audit_windows": ["node", "start", "end", "w0", "tau", "rho", "crisis"], "silent": ["node", "t"],
              "disclosures": ["obj", "owner", "t"],
              "crisis_report": ["id", "lost", "stored", "loss_share", "lead_h", "overlap"], "alarm_check": ["correct"]},
    "graph": {"pairs": ["snap", "i", "l", "rho_hat", "n_inf", "rho", "own", "conf"],
              "nodes": ["suspicion", "cluster", "cluster_infra"]},
}
GRAPH_CHECK = {"pairs": ["same_dc", "same_provider", "same_owner"],
               "snap": ["t", "infra_edges", "infra_same_dc", "own_edges", "own_same_owner", "own_recall",
                        "ari_owner", "ari_infra_dc"]}
FLAT = {"nodes": ["id", "dc", "provider", "region", "subnet", "hw_claimed", "wallet", "join"],
        "daily": ["epoch", "fragments", "objects_alive", "lost_cum", "repair_traffic", "migration_traffic",
                  "repairs", "migrations", "alarms_active"]}
ENUMS = {("net", "outages", "cls"): "outage_cls", ("truth", "outages", "kind"): "outage_kind",
         ("objects", "fragments", "end"): "frag_end", ("truth", "nodes", "leave_reason"): "leave_reason",
         ("truth", "crises", "type"): "crisis_type", ("truth", "crises", "ctype"): "ctype",
         ("policy", "alarms", "type"): "crisis_type", ("policy", "alarms", "ctype"): "ctype",
         ("truth", "shocks", "ctype"): "ctype"}
# поля, которых не должно быть вне раздела truth
TRUTH_FIELDS = {"owner", "hw_true", "theta", "lam_year", "phi", "kind", "source", "destroyed", "predictable",
                "false_alarm", "leave_reason", "cancelled", "correct"}

ok_all = True


def report(name, ok, extra=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{name}: {'OK' if ok else 'ОШИБКА'}{(' — ' + extra) if extra else ''}")


def check_schema(log):
    problems = []
    if log.get("schema") != SCHEMA:
        problems.append("версия схемы")
    for sec in ["meta", "dict"]:
        if sec not in log:
            problems.append(f"нет раздела {sec}")
    for sec, cols in FLAT.items():
        tab = log.get(sec, {})
        if set(tab) != set(cols):
            problems.append(f"{sec}: столбцы {sorted(set(tab) ^ set(cols))}")
        elif len({len(v) for v in tab.values()}) > 1:
            problems.append(f"{sec}: разная длина столбцов")
    for sec, tables in TABLES.items():
        for name, cols in tables.items():
            tab = log.get(sec, {}).get(name)
            if tab is None:
                problems.append(f"нет таблицы {sec}.{name}")
                continue
            if not tab and sec == "truth":
                continue                                     # таблицы кризисов пусты в мире без кризисов
            if set(tab) != set(cols):
                problems.append(f"{sec}.{name}: столбцы {sorted(set(tab) ^ set(cols))}")
            elif len({len(v) for v in tab.values()}) > 1:
                problems.append(f"{sec}.{name}: разная длина столбцов")
    # граф связей (v2): снимки, пары, по снимку на каждый узел; сверка с истиной — построчно к парам
    g, gc = log.get("graph", {}), log["truth"].get("graph_check", {})
    n_snap, N = len(g.get("t", [])), len(log["nodes"]["id"])
    if any(len(v) != n_snap or any(len(x) != N for x in v) for v in g.get("nodes", {}).values()):
        problems.append("graph.nodes: не по снимку на каждый узел")
    for name, cols in GRAPH_CHECK.items():
        tab = gc.get(name, {})
        if set(tab) != set(cols) or len({len(v) for v in tab.values()}) > 1:
            problems.append(f"truth.graph_check.{name}: столбцы или длины")
    if len(gc.get("pairs", {}).get("same_dc", [])) != len(g.get("pairs", {}).get("i", [])):
        problems.append("truth.graph_check.pairs не совпадает по длине с graph.pairs")
    if len(gc.get("snap", {}).get("t", [])) != n_snap:
        problems.append("truth.graph_check.snap не совпадает по числу снимков")
    for (sec, name, col), voc in ENUMS.items():
        vals = log[sec][name].get(col, []) if log[sec][name] else []
        if vals and (min(vals) < 0 or max(vals) >= len(log["dict"][voc])):
            problems.append(f"{sec}.{name}.{col}: номер вне словаря {voc}")
    report("1. Соответствие схеме", not problems, "; ".join(problems))


def check_counts(log, w, ch, res):
    c = w.cfg
    m = lambda x: np.asarray(x)
    pairs = []
    ob = ch.observed
    pairs.append(("наблюдаемые простои длиннее d", len(log["net"]["outages"]["node"]), int((ob["dur"] > c.d).sum())))
    pairs.append(("сумма сводки коротких наблюдаемых событий", sum(map(sum, log["net"]["micro"]["counts"])),
                  int((ob["dur"] <= c.d).sum())))
    pairs.append(("сумма сводки проваленных аудитов", sum(map(sum, log["net"]["audits"]["fails"])),
                  int((~ch.audits["passed_avail"]).sum())))
    pairs.append(("истинные простои", len(log["truth"]["outages"]["node"]), len(w.outages)))
    pairs.append(("истинные микропрерывания (сумма сводки)", sum(map(sum, log["truth"]["micro"]["counts"])), len(ch.micro)))
    pairs.append(("потери", len(log["outcomes"]["losses"]["obj"]), len(res.losses)))
    pairs.append(("раскрытия", len(log["truth"]["disclosures"]["obj"]), len(res.disclosures)))
    pairs.append(("тревоги", len(log["policy"]["alarms"]["t"]), len(res.alarms)))
    pairs.append(("кризисы и ложные тревоги", len(log["truth"]["crises"].get("id", [])), len(w.crises) if w.crisis else 0))
    pairs.append(("тихие потери", len(log["truth"]["silent"]["node"]), len(w.silent)))
    nd = w.nodes
    pairs.append(("уходы", int((m(log["truth"]["nodes"]["leave"]) >= 0).sum()), int(np.isfinite(nd["leave_t"]).sum())))
    pairs.append(("регистрации (не отменённые)", sum(j is not None for j in log["nodes"]["join"]),
                  int((~nd["cancelled"]).sum()) if "cancelled" in nd else len(nd)))
    pairs.append(("последняя строка daily: потеряно нарастающим итогом", log["daily"]["lost_cum"][-1], len(res.losses)))
    sample = set(log["objects"]["sample"]["obj"])
    pairs.append(("фрагменты выборки", len(log["objects"]["fragments"]["obj"]), int(res.fragments["obj"].isin(sample).sum())))
    from taris_sim.graph import LinkGraph
    snaps = LinkGraph(w, ch).snapshots()
    pairs.append(("снимки графа", len(log["graph"]["t"]), len(snaps)))
    pairs.append(("пары графа во всех снимках", len(log["graph"]["pairs"]["i"]), sum(len(s.pairs) for s in snaps)))
    bad = [f"{n}: {a} против {b}" for n, a, b in pairs if a != b]
    report("2. Сверка с прогоном", not bad, "; ".join(bad) if bad else f"{len(pairs)} сверок")


def check_net_view(log):
    leaks = []

    def walk(obj, where):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in TRUTH_FIELDS:
                    leaks.append(f"{where}.{k}")
                walk(v, f"{where}.{k}")

    for sec in ["world", "nodes", "net", "policy", "objects", "outcomes", "daily", "graph"]:
        walk(log[sec], sec)
    report("3. Взгляд сети без полей истины", not leaks, ", ".join(leaks))


if __name__ == "__main__":
    E = 365
    cfg = scenario("S6", **MAIN)
    w = World(cfg); ch = Chain(w); ds = DataStream(cfg, len(w.nodes)); L = Ledger(w, ch, ds)
    res = L.run(CrisisOraclePolicy(w, 13), epochs=E)
    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = os.path.join(tmp, "a.taris.json.gz"), os.path.join(tmp, "b.taris.json.gz")
        export_run(w, ch, res, p1, scenario="S6")
        log = load(p1)
        print(f"Журнал основного мира за {E} суток: {os.path.getsize(p1) / 2**20:.2f} МБ, узлов {len(log['nodes']['id'])}")
        check_schema(log)
        check_counts(log, w, ch, res)
        check_net_view(log)
        export_run(w, ch, res, p2, scenario="S6")
        report("4. Воспроизводимость (два экспорта побитово равны)", filecmp.cmp(p1, p2, shallow=False))
    print("\nИтог:", "все проверки пройдены" if ok_all else "ЕСТЬ ОШИБКИ")
    sys.exit(0 if ok_all else 1)
