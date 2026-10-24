"""Проверки файла сценария и вмешательств (визуализация, этап 6.5; шаг 3, пункт 3.8).

1. Сквозной путь: сценарий без вмешательств даёт тот же мир (хэш, простои, кризисы, микропрерывания)
   и тот же прогон учёта, что обычный запуск, побитово.
2. Каждое вмешательство действует: шок и выключение — простои заданной длительности у всех живых узлов
   цели; кризис — эпизод с заданными типом, кластером и моментом, затронутыми узлами, предвестником
   (если предсказуем) или без него (внезапный, пожар).
3. Остальные события мира не меняются: у узлов вне целей вмешательств (и вне их каскадов) жизнь, простои,
   тихие потери и микропрерывания совпадают с миром без вмешательств; случайная шкала кризисов та же.
4. Файлы с ошибками: каждая ошибка даёт понятное сообщение с номером вмешательства и полем.

Горизонт сокращён до 365 суток (основной мир S6), чтобы проверка шла несколько минут.

    python scripts/check_scenario.py
"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from taris_sim import scenario as SC
from taris_sim.world import World, MANUAL_TAG
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.config import HOUR

ok_all = True
H = 365
BASE = {"scenario": "S6", "world_mode": "matched", "seed": 1, "world": "main", "policy": "crisis_oracle13",
        "overrides": {"horizon_days": H}}
INTERVENTIONS = [
    {"kind": "crisis", "type": "fire", "cluster": 3, "t": 200},
    {"kind": "crisis", "type": "storm", "cluster": 5, "t": 150, "predictable": True, "W_days": 4},
    {"kind": "crisis", "type": "storm", "cluster": 8, "t": 160, "predictable": False},
    {"kind": "crisis", "type": "wear", "cluster": None, "t": 250, "predictable": True},     # партия — крупнейшая живая
    {"kind": "crisis", "type": "exodus", "cluster": 1, "t": 280, "predictable": True},
    {"kind": "crisis", "type": "software", "cluster": None, "t": 300, "predictable": False},
    {"kind": "shock", "ctype": "provider", "cluster": 2, "t": 100, "hours": 30, "destroyed": False},
    {"kind": "shock", "ctype": "dc", "cluster": 7, "t": 120, "hours": 10, "destroyed": True},
    {"kind": "node_down", "node": 12, "t": 90, "hours": 48},
]


def check(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"  [{'ok' if ok else 'ОШИБКА'}] {name}" + (f": {detail}" if detail else ""))


def sc_of(interventions):
    return {"format": SC.FORMAT, "name": "проверка", "base": copy.deepcopy(BASE), "interventions": interventions}


def same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    return a.shape == b.shape and bool((pd.util.hash_pandas_object(a, index=False).values
                                        == pd.util.hash_pandas_object(b, index=False).values).all())


def sorted_out(o: pd.DataFrame) -> pd.DataFrame:
    # метка cluster у простоев кризисов — номер эпизода; вмешательства сдвигают номера каскадов, поэтому без неё
    return o[["node", "start", "end", "kind", "destroyed"]].sort_values(["node", "start", "end"], kind="mergesort")


def biggest_batch(w, t):
    """Партия «ДЦ/класс» с наибольшим числом живых узлов в момент t (у износа нужна непустая цель)."""
    nd = w.nodes
    a = nd.loc[w.alive_at(np.arange(len(nd)), t)]
    dc, hw = a.groupby(["dc", "hw_true"]).size().idxmax()
    return f"{int(dc)}/{w.cfg.hw_classes[int(hw)]}"


print("1. Сквозной путь: сценарий без вмешательств = обычный запуск")
sc0 = sc_of([])
check("файл без вмешательств проходит проверку", SC.validate(sc0) == [])
cfg = SC.config_of(sc0)
w_plain = World(cfg)
INTERVENTIONS[3]["cluster"] = biggest_batch(w_plain, INTERVENTIONS[3]["t"])
w0 = World(cfg, interventions=[])
check("хэш мира совпадает", w_plain.fingerprint() == w0.fingerprint(), w0.fingerprint())
check("простои совпадают", same(w_plain.outages, w0.outages), f"{len(w0.outages)} простоев")
check("кризисы совпадают", same(w_plain.crises, w0.crises), f"{len(w0.crises)} эпизодов")
ch_plain, ch0 = Chain(w_plain), Chain(w0)
check("микропрерывания совпадают", same(ch_plain.micro, ch0.micro))
res_plain = Ledger(w_plain, ch_plain, DataStream(cfg, len(w_plain.nodes))).run(SC.make_policy(BASE["policy"], w_plain))
_, _, res0 = SC.run(sc0)
for k in ("fragments", "losses", "repairs", "migrations", "alarms"):
    check(f"прогон учёта: {k} совпадают", same(getattr(res_plain, k), getattr(res0, k)), f"{len(getattr(res0, k))} строк")

print("\n2. Вмешательства действуют")
sc1 = sc_of(INTERVENTIONS)
check("файл со всеми видами вмешательств проходит проверку", SC.validate(sc1) == [], "; ".join(SC.validate(sc1)))
w1 = World(SC.config_of(sc1), interventions=sc1["interventions"])
nd, out = w1.nodes, w1.outages
manual = w1.crises[w1.crises["manual"]].reset_index(drop=True)
k_cr = 0
affected = set()
for k, it in enumerate(INTERVENTIONS):
    t = float(it["t"])
    if it["kind"] in ("shock", "node_down"):
        idx = (np.flatnonzero(nd[it["ctype"]].to_numpy() == it["cluster"]) if it["kind"] == "shock"
               else np.array([it["node"]]))
        alive = w1.alive_at(idx, t)
        affected |= set(idx.tolist())
        # простой вмешательства мог слиться с пересекающимся простоем узла — тогда он только длиннее
        L = it["hours"] * HOUR
        cover = [((out["node"] == i) & (out["start"] <= t + 1e-9) & (out["end"] >= min(t + L, nd["alive_to"].iat[i]) - 1e-9)).any()
                 for i in alive]
        cls_ok = True
        if it.get("destroyed") or L > w1.cfg.T_rep:
            cls_ok = all((out[(out["node"] == i) & (out["start"] <= t + 1e-9) & (out["end"] > t)]["cls"] == "total").all() for i in alive)
        where = f"{it['ctype']} {it['cluster']}" if it["kind"] == "shock" else f"узел {it['node']}"
        check(f"{k + 1}. {it['kind']} ({where}, t={t:g}, {it['hours']} ч)",
              len(alive) > 0 and all(cover) and cls_ok and it.get("n_hit", w1.interventions[k]["n_hit"]) == len(alive),
              f"затронуто {len(alive)} из {len(idx)} узлов, все в простое {it['hours']} ч"
              + (", полная гибель" if it.get("destroyed") else ""))
    else:
        e = manual.iloc[k_cr]; k_cr += 1
        ep = next(x for x in w1.crisis.episodes if x["id"] == e["id"])
        affected |= set(ep["targets"].tolist())
        pred = bool(it.get("predictable", True)) and it["type"] != "fire"
        # предвестник по своим каналам: пинги (pre_events), аудиты и тихие потери (износ), уходы и регистрации (уход)
        fr = w1.crisis.frames()
        n_pre = (int((w1.pre_events["episode"] == e["id"]).sum()) + int((w1.pre_audit["episode"] == e["id"]).sum())
                 + int((fr["pre_silent"]["episode"] == e["id"]).sum())
                 + int(((w1.forced["episode"] == e["id"]) & (w1.forced["t"] < e["tau"])).sum())
                 + int(((w1.cancelled["episode"] == e["id"]) & (w1.cancelled["join_t"] < e["tau"])).sum()))
        ok = (e["type"] == it["type"] and abs(e["tau"] - t) < 1e-9 and e["n_targets"] > 0
              and bool(e["predictable"]) == pred and (n_pre > 0) == pred)
        if it.get("W_days") is not None:
            ok &= abs(e["W"] - it["W_days"]) < 1e-9
        if it["type"] != "software":
            ok &= str(e["cluster"]) == str(it["cluster"])
        if it["type"] == "fire":
            ok &= bool(e["destroyed"])
        hit = e["n_hit"] if it["type"] not in ("wear", "exodus") else int(np.isin(w1.forced["node"], ep["targets"]).sum())
        ok &= hit > 0
        check(f"{k + 1}. кризис {it['type']} (кластер {e['cluster']}, t={t:g})", ok,
              f"целей {e['n_targets']}, задето {hit}, " + (f"предвестник {e['W']:.1f} сут, событий {n_pre}" if pred else "без предвестника")
              + (", гибель" if e["destroyed"] else ""))
        if it["type"] == "exodus":       # массовый уход подавляет и регистрации в регионе
            affected |= set(np.flatnonzero(nd["region"].to_numpy() == it["cluster"]).tolist())
# каскады от кризисов-вмешательств тоже задевают узлы
for x in w1.crisis.episodes:
    if x["parent"] in set(manual["id"]):
        affected |= set(x["targets"].tolist())
check("в журнале мира вмешательства помечены", len(manual) == sum(i["kind"] == "crisis" for i in INTERVENTIONS)
      and (out["cluster"] >= MANUAL_TAG).sum() > 0, f"кризисов manual={len(manual)}, простоев с меткой вмешательства "
      f"{int((out['cluster'] >= MANUAL_TAG).sum())}")
check("у вмешательств с живой целью нет предупреждений", SC.empty_targets(w1) == [], "; ".join(SC.empty_targets(w1)))
# пустые цели: партия без узлов, узел, ушедший до вмешательства
batches = {(int(d), int(h)) for d, h in w0.nodes[["dc", "hw_true"]].to_numpy()}
empty = next(f"{d}/{h}" for d in range(w0.cfg.n_dc) for j, h in enumerate(w0.cfg.hw_classes) if (d, j) not in batches)
gone = int(np.flatnonzero(w0.nodes["alive_to"].to_numpy() < 50)[0])
sc_e = sc_of([{"kind": "crisis", "type": "wear", "cluster": empty, "t": 200, "predictable": True},
              {"kind": "node_down", "node": gone, "t": 100, "hours": 5}])
w_e = World(SC.config_of(sc_e), interventions=sc_e["interventions"])
warn = SC.empty_targets(w_e)
check("пустая цель даёт предупреждение", len(warn) == 2 and "вмешательство 1" in warn[0] and "вмешательство 2" in warn[1],
      " | ".join(warn))

print("\n3. Остальные события мира не меняются")
rest = np.setdiff1d(np.arange(len(nd)), np.fromiter(affected, int))
check("число узлов то же", len(nd) == len(w0.nodes), f"{len(nd)}; вне целей вмешательств {len(rest)}")
cols = ["owner", "join_t", "leave_t", "alive_from", "alive_to", "dc", "provider", "region", "theta"]
check("жизнь и признаки узлов вне целей совпадают", same(nd.loc[rest, cols], w0.nodes.loc[rest, cols]))
o1, o0 = w1.outages, w0.outages
check("простои узлов вне целей совпадают", same(sorted_out(o1[np.isin(o1["node"], rest)]), sorted_out(o0[np.isin(o0["node"], rest)])),
      f"{int(np.isin(o0['node'], rest).sum())} простоев")
s1, s0 = w1.silent, w0.silent
check("тихие потери узлов вне целей совпадают", same(s1[np.isin(s1["node"], rest)].sort_values(["node", "t"]),
                                                     s0[np.isin(s0["node"], rest)].sort_values(["node", "t"])))
ch1 = Chain(w1)
m1, m0 = ch1.micro, ch0.micro
check("микропрерывания узлов вне целей совпадают", same(m1[np.isin(m1["node"], rest)], m0[np.isin(m0["node"], rest)]))
key = ["type", "ctype", "cluster", "tau", "W", "predictable", "false_alarm"]
lib1 = w1.crises[~w1.crises["manual"] & ~w1.crises["parent"].isin(manual["id"])][key]
check("случайная шкала кризисов та же", same(lib1, w0.crises[key]), f"{len(w0.crises)} эпизодов")

print("\n4. Файлы с ошибками")
bad = [
    ({"format": "taris-scenario/0"}, None, "format"),
    ({"base": {**BASE, "world": "huge"}}, None, "base.world"),
    ({"base": {**BASE, "policy": "agent"}}, None, "base.policy"),
    ({"base": {**BASE, "world_mode": "random"}}, None, "base.world_mode"),
    ({"base": {**BASE, "overrides": {"no_such_param": 1}}}, None, "base:"),
    (None, {"kind": "crisis", "type": "flood", "cluster": 1, "t": 10}, "вмешательство 1: type"),
    (None, {"kind": "crisis", "type": "fire", "cluster": 99, "t": 10}, "вмешательство 1: cluster"),
    (None, {"kind": "crisis", "type": "exodus", "cluster": 9, "t": 10}, "вмешательство 1: cluster"),
    (None, {"kind": "crisis", "type": "wear", "cluster": "2/floppy", "t": 10}, "вмешательство 1: cluster"),
    (None, {"kind": "crisis", "type": "fire", "cluster": 1, "t": H + 5}, "вмешательство 1: t"),
    (None, {"kind": "shock", "ctype": "planet", "cluster": 1, "t": 10, "hours": 5}, "вмешательство 1: ctype"),
    (None, {"kind": "shock", "ctype": "dc", "cluster": 1, "t": 10, "hours": 0}, "вмешательство 1: hours"),
    (None, {"kind": "node_down", "node": -3, "t": 10, "hours": 5}, "вмешательство 1: node"),
    (None, {"kind": "meteor", "t": 10}, "вмешательство 1: kind"),
]
for patch, it, expect in bad:
    sc = sc_of([it] if it else [])
    if patch:
        sc.update(patch)
    err = SC.validate(sc)
    check(f"ошибка «{expect}»", any(expect in x for x in err), "; ".join(err))
try:
    import tempfile, json
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(sc_of([bad[5][1]]), f, ensure_ascii=False)
    SC.load(f.name)
    check("load отказывается читать файл с ошибками", False)
except ValueError as err:
    check("load отказывается читать файл с ошибками", "вмешательство 1" in str(err), str(err).splitlines()[1])
finally:
    os.unlink(f.name)

print("\nИтог:", "все проверки пройдены" if ok_all else "есть ошибки")
sys.exit(0 if ok_all else 1)
