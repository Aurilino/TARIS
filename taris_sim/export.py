"""Журнал прогона для просмотрщика (шаг 3, пункт 3.8; визуализация, этап 1).

Схема — docs/42_log_schema.md, версия taris-log/1. Раздел truth — «взгляд истины»; все остальные
разделы — то, что видит сеть. Таблицы — столбцами, время — целыми минутами (-1 — нет значения),
перечисления — номерами из раздела dict. Файл — JSON, сжатый gzip без метки времени, поэтому
один и тот же прогон даёт побитово один и тот же журнал.
"""
from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = "taris-log/2"          # v2 = v1 + раздел graph (граф связей, пункт 2.4) и truth.graph_check
OUTAGE_KIND = ["own", "shock", "storm", "crisis", "pre_short", "leave"]
OUTAGE_CLS = ["micro", "partial", "total"]
FRAG_END = ["", "простой > T_rep", "тихая потеря (аудит)", "перенос"]
LEAVE_REASON = ["", "churn", "exodus", "wear_death", "pre_churn"]
CRISIS_TYPES = ["storm", "cascade", "wear", "exodus", "software", "fire"]
CTYPES = ["dc", "provider", "region", "batch", "version"]


def minutes(x) -> list:
    """Время в сутках -> целые минуты; бесконечность и NaN -> -1."""
    x = np.asarray(x, dtype=float)
    return np.where(np.isfinite(x), np.round(x * 1440.0), -1).astype(np.int64).tolist()


def ints(x) -> list:
    return np.asarray(x).astype(np.int64).tolist()


def rounded(x, nd=4) -> list:
    return [None if not np.isfinite(v) else round(float(v), nd) for v in np.asarray(x, dtype=float)]


def code(values, vocab) -> list:
    idx = {v: i for i, v in enumerate(vocab)}
    return [idx[v] for v in values]


def plain(v):
    """Значение метрики -> JSON: numpy -> python, inf и NaN -> null."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if np.isfinite(v) else None
    return v


def git_commit(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def binned(times: dict, nodes: pd.DataFrame, agg: float) -> dict:
    """Сводка событий по узлам и интервалам агрегации: для каждого узла — первый интервал его жизни
    и числа событий по интервалам до конца жизни (у отменённых узлов — пустой массив)."""
    bin0, counts = [], []
    a0, a1 = nodes["alive_from"].to_numpy(), nodes["alive_to"].to_numpy()
    for i in range(len(nodes)):
        if a1[i] <= a0[i]:
            bin0.append(-1); counts.append([])
            continue
        b0, b1 = int(a0[i] // agg), int(np.ceil(a1[i] / agg))
        t = times.get(i, np.empty(0))
        b = (t // agg).astype(int) - b0
        b = b[(b >= 0) & (b < b1 - b0)]
        bin0.append(b0); counts.append(np.bincount(b, minlength=b1 - b0).tolist())
    return {"node": list(range(len(nodes))), "bin0": bin0, "counts": counts}


def by_node(df: pd.DataFrame, col: str) -> dict:
    return {int(i): np.sort(g[col].to_numpy()) for i, g in df.groupby("node")}


def graph_sections(world, chain, show_min: float = 0.3):
    """Раздел graph (снимки графа связей — «взгляд сети») и truth.graph_check (сверка с истиной)."""
    from .graph import LinkGraph, ari, partition
    nd = world.nodes
    dc, prov, own = nd["dc"].to_numpy(), nd["provider"].to_numpy(), nd["owner"].to_numpy()
    snaps = LinkGraph(world, chain).snapshots()
    g = {"t": [], "pairs": {k: [] for k in ["snap", "i", "l", "rho_hat", "n_inf", "rho", "own", "conf"]},
         "nodes": {"suspicion": [], "cluster": [], "cluster_infra": []}}
    chk = {"pairs": {"same_dc": [], "same_provider": [], "same_owner": []},
           "snap": {k: [] for k in ["t", "infra_edges", "infra_same_dc", "own_edges", "own_same_owner",
                                    "own_recall", "ari_owner", "ari_infra_dc"]}}
    for k, s in enumerate(snaps):
        p = s.pairs
        g["t"] += minutes([s.t])
        P = g["pairs"]
        P["snap"] += [k] * len(p); P["i"] += ints(p["i"]); P["l"] += ints(p["l"])
        P["rho_hat"] += rounded(p["rho_hat"], 3); P["n_inf"] += ints(p["n_inf"])
        P["rho"] += rounded(p["rho"], 3); P["own"] += rounded(p["own"], 3); P["conf"] += rounded(p["conf"], 3)
        g["nodes"]["suspicion"].append(rounded(s.suspicion, 3))
        g["nodes"]["cluster"].append(ints(s.clusters)); g["nodes"]["cluster_infra"].append(ints(s.clusters_infra))
        i, l = p["i"].to_numpy(), p["l"].to_numpy()
        sd, sp, so = dc[i] == dc[l], prov[i] == prov[l], own[i] == own[l]
        chk["pairs"]["same_dc"] += ints(sd); chk["pairs"]["same_provider"] += ints(sp); chk["pairs"]["same_owner"] += ints(so)
        reg = np.flatnonzero(~np.isnan(s.suspicion))
        inf_e, own_e = p["rho"].to_numpy() >= show_min, p["own"].to_numpy() >= show_min
        same_owner_pairs = sum(n * (n - 1) // 2 for n in np.bincount(own[reg]))
        S = chk["snap"]
        S["t"] += minutes([s.t]); S["infra_edges"].append(int(inf_e.sum()))
        S["infra_same_dc"].append(plain(float(sd[inf_e].mean())) if inf_e.any() else None)
        S["own_edges"].append(int(own_e.sum()))
        S["own_same_owner"].append(plain(float(so[own_e].mean())) if own_e.any() else None)
        S["own_recall"].append(plain(float((so & own_e).sum() / same_owner_pairs)) if same_owner_pairs else None)
        S["ari_owner"].append(round(ari(partition(s.clusters, reg), own[reg]), 4))
        S["ari_infra_dc"].append(round(ari(partition(s.clusters_infra, reg), dc[reg]), 4))
    return g, chk


def export_run(world, chain, res, path, scenario: str | None = None, agg_days: int = 1, n_sample: int = 50,
               root: Path | None = None, graph: bool = True) -> dict:
    """Пишет журнал прогона по схеме taris-log/1 и возвращает его как словарь."""
    w, c = world, world.cfg
    nd = w.nodes
    H = float(c.horizon_days)
    crisis = w.crisis is not None
    root = root or Path(__file__).resolve().parents[1]
    subnets = sorted(nd["subnet"].unique())

    # ---------------------------------------------------------------- сеть
    cancelled = nd["cancelled"].to_numpy() if "cancelled" in nd else np.zeros(len(nd), bool)
    nodes = {"id": list(range(len(nd))), "dc": ints(nd["dc"]), "provider": ints(nd["provider"]),
             "region": ints(nd["region"]), "subnet": code(nd["subnet"], subnets),
             "hw_claimed": ints(nd["hw_claimed"]), "wallet": ints(nd["wallet"]),
             # заявленный класс узла: 1 — устройство пользователя (пункт 3.10, dc = -1), 0 — узел в дата-центре
             "device": ints(nd["is_device"]) if "is_device" in nd else [0] * len(nd),
             # регистрация может быть до начала горизонта (отрицательные минуты); отменённая кризисом — null
             "join": [None if cancelled[i] else m for i, m in enumerate(minutes(nd["join_t"]))]}
    ob = chain.observed
    long_ = ob[ob["dur"] > c.d]
    short = ob[ob["dur"] <= c.d]
    fails = chain.audits[~chain.audits["passed_avail"]]
    net = {
        "outages": {"node": ints(long_["node"]), "start": minutes(long_["start"]), "end": minutes(long_["end"].clip(upper=H)),
                    "cls": [2 if d > c.T_rep else 1 for d in long_["dur"]]},
        "micro": binned(by_node(short, "start"), nd, agg_days),
        "audits": binned(by_node(fails, "t"), nd, agg_days),
        "versions": {"version": [], "t": []}, "updates": {"node": [], "version": [], "t": []},
    }
    net["audits"]["fails"] = net["audits"].pop("counts")
    if crisis:
        net["versions"] = {"version": ints(w.versions["version"]), "t": minutes(w.versions["t"])}
        net["updates"] = {"node": ints(w.node_updates["node"]), "version": ints(w.node_updates["version"]),
                          "t": minutes(w.node_updates["t"])}

    # ---------------------------------------------------------------- политика и объекты
    ob_all = res.objects
    step = max(1, len(ob_all) // n_sample)
    sample = np.arange(0, len(ob_all), step)[:n_sample]
    in_s = lambda s: s.isin(sample)
    al = res.alarms
    ae = res.alarm_episodes() if crisis else pd.DataFrame(columns=["type", "ctype", "cluster", "t0", "until", "n", "correct"])
    rp, mg, fr = res.repairs, res.migrations, res.fragments
    rp_s, mg_s, fr_s = rp[in_s(rp["obj"])], mg[in_s(mg["obj"])], fr[in_s(fr["obj"])]
    policy = {
        "alarms": {"t": minutes(al["t"]), "type": code(al["type"], CRISIS_TYPES), "ctype": code(al["ctype"], CTYPES),
                   "cluster": [str(x) for x in al["cluster"]], "prob": rounded(al["prob"]), "horizon": rounded(al["horizon"])},
        "alarm_episodes": {"type": code(ae["type"], CRISIS_TYPES), "ctype": code(ae["ctype"], CTYPES),
                           "cluster": [str(x) for x in ae["cluster"]], "t0": minutes(ae["t0"]),
                           "until": minutes(ae["until"]), "n": ints(ae["n"])},
        "repairs": {"obj": ints(rp_s["obj"]), "t": minutes(rp_s["t"]), "r": ints(rp_s["r"]), "traffic": rounded(rp_s["traffic"])},
        "migrations": {"obj": ints(mg_s["obj"]), "idx": ints(mg_s["idx"]), "from": ints(mg_s["from"]), "to": ints(mg_s["to"]),
                       "t": minutes(mg_s["t"]), "t_done": minutes(mg_s["t_done"]), "ok": ints(mg_s["ok"].astype(bool))},
    }
    so = ob_all.iloc[sample]
    objects = {
        "sample": {"obj": ints(sample), "created": minutes(so["created_epoch"]), "n": ints(so["n"]), "lost": minutes(so["lost_t"])},
        "fragments": {"obj": ints(fr_s["obj"]), "idx": ints(fr_s["idx"]), "node": ints(fr_s["node"]),
                      "from": minutes(fr_s["t_from"]), "to": minutes(fr_s["t_to"]), "end": code(fr_s["end"], FRAG_END)},
    }
    # раскрытие сеть не видит (коалиция собирает фрагменты тайно), поэтому оно — в truth
    outcomes = {"losses": {"obj": ints(res.losses["obj"]), "t": minutes(res.losses["t"])}}

    # ---------------------------------------------------------------- метрики по эпохам
    d = res.daily
    E = len(d)
    ep = lambda t: np.clip(np.floor(np.asarray(t, dtype=float)).astype(int), 0, E - 1)
    lost_cum = np.cumsum(np.bincount(ep(res.losses["t"]), minlength=E)) if len(res.losses) else np.zeros(E, int)
    rep_tr = np.bincount(ep(rp["t"]), weights=rp["traffic"], minlength=E) if len(rp) else np.zeros(E)
    mig_tr = np.bincount(ep(mg["t"]), weights=mg["traffic"], minlength=E) if len(mg) else np.zeros(E)
    active = np.zeros(E, int)
    for a in ae.itertuples():
        active[int(np.ceil(a.t0)):min(E, int(np.ceil(a.until)))] += 1
    daily = {"epoch": ints(d["epoch"]), "fragments": ints(d["fragments"]), "objects_alive": ints(d["objects_alive"]),
             "lost_cum": ints(lost_cum), "repair_traffic": rounded(rep_tr, 3), "migration_traffic": rounded(mig_tr, 3),
             "repairs": ints(np.bincount(ep(rp["t"]), minlength=E)) if len(rp) else [0] * E,
             "migrations": ints(np.bincount(ep(mg["t"]), minlength=E)) if len(mg) else [0] * E,
             "alarms_active": ints(active)}

    # ---------------------------------------------------------------- истина
    to = w.outages
    phi = [w.true_phi(i) for i in range(len(nd))]
    leave_reason = nd["leave_reason"] if "leave_reason" in nd else pd.Series(np.where(np.isfinite(nd["leave_t"]), "churn", ""))
    truth = {
        "nodes": {"id": list(range(len(nd))), "owner": ints(nd["owner"]), "hw_true": ints(nd["hw_true"]),
                  "lam_year": rounded(nd["lam_avg"] * 365.0), "phi": rounded(phi, 5), "leave": minutes(nd["leave_t"]),
                  "leave_reason": code(leave_reason, LEAVE_REASON), "cancelled": ints(cancelled)},
        "outages": {"node": ints(to["node"]), "start": minutes(to["start"]), "end": minutes(to["end"].clip(upper=H)),
                    "kind": code(to["kind"], OUTAGE_KIND), "destroyed": ints(to["destroyed"].astype(bool))},
        "micro": binned(by_node(chain.micro, "start"), nd, agg_days),
        "pre_micro": binned({}, nd, agg_days),
        "shocks": {"ctype": code(w.shocks["ctype"], CTYPES), "cluster": ints(w.shocks["cluster"]),
                   "start": minutes(w.shocks["start"]), "end": minutes(w.shocks["end"]),
                   "destroyed": ints(w.shocks["destroyed"].astype(bool)), "n_nodes": ints(w.shocks["n_nodes"])},
        "crises": {}, "crisis_targets": {"crisis": [], "node": []}, "precursors": {"crisis": [], "node": [], "start": [], "end": []},
        "audit_windows": {}, "silent": {"node": ints(w.silent["node"]), "t": minutes(w.silent["t"])},
        "disclosures": {"obj": ints(res.disclosures["obj"]), "owner": ints(res.disclosures["owner"]),
                        "t": minutes(res.disclosures["t"])},
        "crisis_report": {}, "alarm_check": {"correct": ints(ae["correct"].astype(bool))},
    }
    if crisis:
        cr, k = w.crises, w.crisis
        pm = w.pre_events[w.pre_events["channel"] == "micro"]
        truth["pre_micro"] = binned(by_node(pm, "start"), nd, agg_days)
        truth["crises"] = {"id": ints(cr["id"]), "type": code(cr["type"], CRISIS_TYPES), "ctype": code(cr["ctype"], CTYPES),
                           "cluster": [str(x) for x in cr["cluster"]], "tau": minutes(cr["tau"]), "W": rounded(cr["W"]),
                           "w0": minutes(cr["w0"]), "end": minutes(cr["end"]), "predictable": ints(cr["predictable"]),
                           "false_alarm": ints(cr["false_alarm"]), "parent": ints(cr["parent"]),
                           "n_targets": ints(cr["n_targets"]), "n_hit": ints(cr["n_hit"]), "destroyed": ints(cr["destroyed"]),
                           "manual": ints(cr["manual"])}
        alive_to = nd["alive_to"].to_numpy()
        for e in k.episodes:
            truth["crisis_targets"]["crisis"] += [e["id"]] * len(e["targets"])
            truth["crisis_targets"]["node"] += ints(e["targets"])
            if e["predictable"] or e["false_alarm"]:
                for i in k.precursor_nodes(e):
                    lo, hi = k.precursor_span(e, i)
                    hi = min(hi, alive_to[i])
                    if hi > lo:
                        p = truth["precursors"]
                        p["crisis"].append(e["id"]); p["node"].append(int(i))
                        p["start"] += minutes([lo]); p["end"] += minutes([hi])
        pa = w.pre_audit
        truth["audit_windows"] = {"node": ints(pa["node"]), "start": minutes(pa["start"]), "end": minutes(pa["end"]),
                                  "w0": minutes(pa["w0"]), "tau": minutes(pa["tau"]), "rho": rounded(pa["rho"]),
                                  "crisis": ints(pa["episode"])}
        summary, tab = res.crisis_report()
        truth["crisis_report"] = {"id": ints(tab["id"]), "lost": rounded(tab["lost"]), "stored": rounded(tab["stored"]),
                                  "loss_share": rounded(tab["loss_share"], 7), "lead_h": rounded(tab["lead_h"], 2),
                                  "overlap": ints(tab["overlap"])}
    else:
        summary = {}

    metrics = {k_: plain(v) for k_, v in {**res.metrics(), **summary}.items()}
    log = {
        "schema": SCHEMA,
        "meta": {"scenario": scenario, "world_mode": c.world_mode, "seed": c.seed,
                 "config": json.loads(c.to_json()), "config_hash": c.hash(), "code_commit": git_commit(root),
                 "policy": res.policy_name, "horizon_days": c.horizon_days, "agg_days": agg_days,
                 "n_sample": len(sample), "metrics": metrics,
                 "interventions": [{k_: plain(v) for k_, v in it.items()} for it in w.interventions]},
        "dict": {"outage_kind": OUTAGE_KIND, "outage_cls": OUTAGE_CLS, "frag_end": FRAG_END, "leave_reason": LEAVE_REASON,
                 "crisis_type": CRISIS_TYPES, "ctype": CTYPES, "hw": list(c.hw_classes), "subnet": [str(s) for s in subnets]},
        "world": {"regions": {"id": list(range(c.n_regions))},
                  "providers": {"id": list(range(int(nd["provider"].max()) + 1)),        # с домашними провайдерами (3.10)
                                "home": [int(p >= c.n_providers) for p in range(int(nd["provider"].max()) + 1)]},
                  "dcs": {"id": list(range(c.n_dc)), "provider": ints(w.dc_provider), "region": ints(w.dc_region)}},
        "nodes": nodes, "net": net, "policy": policy, "objects": objects, "outcomes": outcomes, "daily": daily,
        "truth": truth,
    }
    if graph:
        log["graph"], truth["graph_check"] = graph_sections(w, chain)
        log["meta"]["graph_show_min"] = 0.3
    raw = json.dumps(log, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh, gzip.GzipFile(fileobj=fh, mode="wb", mtime=0, filename="") as gz:
        gz.write(raw)
    return log


def load(path) -> dict:
    with gzip.open(path, "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))
