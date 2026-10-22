"""Проверки модуля ledger: учёт потерь сверяется с независимым пересчётом по журналу фрагментов.

Часть 1 — S2, равномерная политика. Часть 2 — S6 (кризисы), оракул кризисов: переносы фрагментов и тревоги.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import UniformPolicy, CrisisOraclePolicy

E = 365


def recount(res, k):
    """Независимый пересчёт: объект потерян <=> число существующих фрагментов когда-либо < k.
    Удачный перенос — непрерывное продолжение фрагмента: убыль старой копии и появление новой
    в один момент в пересчёт не входят."""
    fr, ob, mg = res.fragments, res.objects, res.migrations
    cont = set()
    if len(mg):
        ok = mg[mg.ok]
        cont = set(zip(ok.obj, ok.idx, ok.to, ok.t_done))
    bad_flag, bad_time = 0, 0
    for obj, g in fr.groupby("obj"):
        ev = []
        for r in g.itertuples(index=False):
            if (r.obj, r.idx, r.node, r.t_from) not in cont:
                ev.append((r.t_from, 1))
            b = min(r.t_dead, r.t_to)
            if np.isfinite(b) and r.end != "перенос":
                ev.append((b, -1))
        ev.sort(key=lambda e: (e[0], e[1]))
        cnt, first_below = 0, np.inf
        t0 = ob.at[obj, "created_epoch"]
        for t, dlt in ev:
            cnt += dlt
            if t > t0 and cnt < k and first_below == np.inf:
                first_below = t
        lost_t = ob.at[obj, "lost_t"]
        if np.isfinite(first_below) != np.isfinite(lost_t):
            bad_flag += 1
        elif np.isfinite(lost_t) and abs(first_below - lost_t) > 1e-9:
            bad_time += 1
    print(f"Объектов проверено: {fr.obj.nunique()}, потеряно: {len(res.losses)}; расхождений в факте потери: {bad_flag}, во времени: {bad_time}")


def conservation(res, L):
    """Размещено - признано утраченными - перенесено (старые копии) = хранится сейчас."""
    fr, mg = res.fragments, res.migrations
    placed = len(fr)
    moved = int((fr.end == "перенос").sum())
    declared = int(np.isfinite(fr.t_to).sum()) - moved
    ok = placed - declared - moved == L.used.sum() and moved == int(mg.ok.sum() if len(mg) else 0)
    print(f"Фрагменты: размещено {placed}, признано утраченными {declared}, перенесено {moved}, хранится {int(L.used.sum())} -> {'OK' if ok else 'ОШИБКА'}")


def traffic(res, ds, k):
    rp, mg = res.repairs, res.migrations
    ok = np.allclose(rp.traffic, (k + rp.r) * ds.objects.loc[rp.obj, "size"].to_numpy() / k)
    print(f"Формула трафика восстановления: {'OK' if ok else 'ОШИБКА'}")
    if len(mg):
        ok = np.allclose(mg.traffic, ds.objects.loc[mg.obj, "size"].to_numpy() / k)
        print(f"Формула трафика переноса: {'OK' if ok else 'ОШИБКА'}")


# ---------------------------------------------------------------- часть 1: S2
print("=== S2, равномерное размещение, год")
cfg = scenario("S2")
w = World(cfg); ch = Chain(w); ds = DataStream(cfg, len(w.nodes)); L = Ledger(w, ch, ds)
res = L.run(UniformPolicy(12), epochs=E)
recount(res, cfg.k)
conservation(res, L)
traffic(res, ds, cfg.k)
res2 = L.run(UniformPolicy(12), epochs=E)
same = res.losses.equals(res2.losses) and res.repairs.equals(res2.repairs)
print(f"Воспроизводимость прогона: {'OK' if same else 'ОШИБКА'}")

# ---------------------------------------------------------------- часть 2: S6
print("\n=== S6, оракул кризисов, год")
cfg = scenario("S6")
w = World(cfg); ch = Chain(w); ds = DataStream(cfg, len(w.nodes)); L = Ledger(w, ch, ds)
res = L.run(CrisisOraclePolicy(w, 13), epochs=E)
recount(res, cfg.k)
conservation(res, L)
traffic(res, ds, cfg.k)
# у живого объекта нет двух фрагментов с одним номером и двух фрагментов на одном узле
dup = 0
for obj in range(len(res.objects)):
    fs = L.obj_frags[obj]
    idx = [L.f_idx[f] for f in fs]; nodes = [L.f_node[f] for f in fs]
    dup += (len(idx) != len(set(idx))) + (len(nodes) != len(set(nodes)))
print(f"Дубли номеров и узлов у объектов: {'OK' if dup == 0 else f'ОШИБКА ({dup})'}")
# тревоги оракула кризисов верны всегда, а полнота равна доле окон, захватывающих границу эпохи
s, tab = res.crisis_report()
cr = w.crises
pred = cr[cr.predictable & ~cr.false_alarm & (cr.tau < E)]
exp_det = int((np.ceil(pred.w0.clip(lower=0)) < pred.tau).sum())
ok = s["Точность тревог"] == 1.0 and s["Замечено заранее"] == exp_det
print(f"Тревоги оракула: точность {s['Точность тревог']}, замечено {s['Замечено заранее']} из ожидаемых {exp_det} -> {'OK' if ok else 'ОШИБКА'}")
m = res.metrics()
ok = m["Размещено не полностью"] == 0 and m["Объекто-суток без запроса восстановления"] == 0
print(f"Недоразмещение и простаивающие восстановления: {'OK' if ok else 'ОШИБКА'}")
