"""Проверки модуля ledger: учёт потерь сверяется с независимым пересчётом по журналу фрагментов."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import UniformPolicy

cfg = scenario("S2")
w = World(cfg); ch = Chain(w); ds = DataStream(cfg, len(w.nodes)); L = Ledger(w, ch, ds)
E = 365
res = L.run(UniformPolicy(12), epochs=E)
fr, ob, k = res.fragments, res.objects, cfg.k

# 1. Независимый пересчёт: объект потерян <=> число существующих фрагментов когда-либо < k
bad_flag, bad_time, checked = 0, 0, 0
for obj, g in fr.groupby("obj"):
    a = g.t_from.to_numpy(); b = np.minimum(g.t_dead.to_numpy(), g.t_to.to_numpy())
    ev = sorted([(x, 1) for x in a] + [(x, -1) for x in b if np.isfinite(x)], key=lambda e: (e[0], e[1]))
    cnt, first_below = 0, np.inf
    t0 = ob.at[obj, "created_epoch"]
    for t, dlt in ev:
        cnt += dlt
        if t > t0 and cnt < k and first_below == np.inf:
            first_below = t
    lost_t = ob.at[obj, "lost_t"]
    checked += 1
    if np.isfinite(first_below) != np.isfinite(lost_t):
        bad_flag += 1
    elif np.isfinite(lost_t) and abs(first_below - lost_t) > 1e-9:
        bad_time += 1
print(f"Объектов проверено: {checked}, потеряно: {len(res.losses)}; расхождений в факте потери: {bad_flag}, во времени: {bad_time}")

# 2. Сохранение фрагментов: размещено - признано утраченными = хранится сейчас
placed = len(fr); declared = int(np.isfinite(fr.t_to).sum())
print(f"Фрагменты: размещено {placed}, признано утраченными {declared}, хранится {int(L.used.sum())} -> {'OK' if placed - declared == L.used.sum() else 'ОШИБКА'}")

# 3. Трафик восстановления = (k + r) * size / k
rp = res.repairs
ok = np.allclose(rp.traffic, (k + rp.r) * ds.objects.loc[rp.obj, "size"].to_numpy() / k)
print(f"Формула трафика восстановления: {'OK' if ok else 'ОШИБКА'}")

# 4. Воспроизводимость прогона
res2 = L.run(UniformPolicy(12), epochs=E)
same = res.losses.equals(res2.losses) and res.repairs.equals(res2.repairs)
print(f"Воспроизводимость прогона: {'OK' if same else 'ОШИБКА'}")
