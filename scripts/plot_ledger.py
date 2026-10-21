"""Графики первого сквозного прогона: жизнь объекта (G) и сравнение границ (H)."""
import sys, os, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RUNS, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
R = {n: pickle.load(open(f"{RUNS}/{n}.pkl", "rb")) for n in ("uniform15", "uniform13", "oracle13")}
LABEL = {"uniform15": "Равномерное, n = 15", "uniform13": "Равномерное, n = 13", "oracle13": "Оракул, n = 13"}
COL = {"uniform15": "#7FA7D9", "uniform13": "#2F6DB5", "oracle13": "#3A9E5C"}
K = 10

# ------------------------------------------------- G. Жизнь объекта: резерв информации
d = R["uniform13"]
fr, ob = d["fragments_sample"], d["objects"]
life = {o: ob.at[o, "lost_t"] - ob.at[o, "created_epoch"] for o in fr.obj.unique() if np.isfinite(ob.at[o, "lost_t"])}
lost_ids = sorted(life, key=life.get)
alive_ids = [o for o in fr.obj.unique() if not np.isfinite(ob.at[o, "lost_t"]) and ob.at[o, "created_epoch"] < 400]
pick = alive_ids[:2] + lost_ids[:1]
fig, axes = plt.subplots(len(pick), 1, figsize=(11, 7.2))
for ax, obj in zip(axes, pick):
    g = fr[fr.obj == obj]
    t0 = ob.at[obj, "created_epoch"]
    lost = np.isfinite(ob.at[obj, "lost_t"])
    end_t = ob.at[obj, "lost_t"] if lost else t0 + 365
    a = g.t_from.to_numpy(); b = np.minimum(g.t_dead.to_numpy(), g.t_to.to_numpy())
    ev = sorted([(x, 1) for x in a] + [(x, -1) for x in b if x <= end_t])
    ts, cnt, c = [], [], 0
    for t, dl in ev:
        c += dl
        if ts and abs(t - ts[-1]) < 1e-12:
            cnt[-1] = c
        else:
            ts.append(t); cnt.append(c)
    ts = np.array(ts) - t0; cnt = np.array(cnt)
    x1 = (end_t - t0) + (10 if lost else 0)
    ax.step(np.r_[ts, x1], np.r_[cnt, cnt[-1]], where="post", color="#2F6DB5", lw=1.4)
    ax.axhline(13, color="#9E9E9E", ls=":", lw=1); ax.axhline(K, color="#C8423B", ls="--", lw=1)
    ax.fill_between([0, x1], K, 13, color="#3A9E5C", alpha=0.07)
    ax.set_ylim(K - 4, 14.5); ax.set_xlim(0, x1); ax.set_ylabel("Живых\nфрагментов")
    if lost:
        ax.axvline(end_t - t0, color="#C8423B", lw=1.5)
        ax.text(end_t - t0 - 2, K - 3.4, f"объект потерян на {end_t - t0:.0f}-е сутки жизни", color="#C8423B", fontsize=8, ha="right")
    ax.set_title(f"Объект {obj}" + (" — потерян" if lost else ""), fontsize=9, loc="left")
    ax.set_xlabel("Сутки жизни объекта", fontsize=8)
axes[0].legend(handles=[Line2D([], [], color="#9E9E9E", ls=":", label="n = 13 (при создании)"),
                        Line2D([], [], color="#C8423B", ls="--", label="k = 10 (минимум для восстановления)"),
                        plt.Rectangle((0, 0), 1, 1, color="#3A9E5C", alpha=0.15, label="резерв информации: n − k = 3")],
               loc="lower left", fontsize=8, frameon=False, ncol=3)
fig.suptitle("Резерв информации объекта во времени (равномерное размещение, n = 13)", fontsize=11)
fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.07, hspace=0.55)
fig.savefig(f"{OUT}/G_object_reserve.png", dpi=150); plt.close(fig)

# ------------------------------------------------- H. Сравнение границ
fig = plt.figure(figsize=(11, 6.6))
ax = fig.add_axes([0.07, 0.47, 0.9, 0.46])
for n in ("uniform13", "uniform15", "oracle13"):
    L = d_ = R[n]["losses"]
    t = np.sort(L.t.to_numpy())
    ax.step(np.r_[0, t, 1095], np.r_[0, np.arange(1, len(t) + 1), len(t)], where="post", color=COL[n], lw=1.8, label=LABEL[n])
sh = R["uniform15"]["shocks"]
sh = sh.assign(dur_h=(sh.end - sh.start) * 24)
last_x, lvl = -999, 0
for r in sh[sh.destroyed | (sh.dur_h > 24)].sort_values("start").itertuples():
    col = "#C8423B" if r.destroyed else "#E8A33D"
    ax.axvline(r.start, color=col, lw=0.9, ls=":")
    lab = f"{'провайдер' if r.ctype == 'provider' else 'ДЦ'} {r.n_nodes} узл." + ("" if r.destroyed else f", {r.dur_h:.0f} ч")
    lvl = (lvl + 1) % 3 if r.start - last_x < 60 else 0
    y = 2.2e3 if r.ctype == "provider" else (0.35, 1.2, 4.5)[lvl]
    ax.text(r.start + 4, y, lab, fontsize=7, color=col)
    last_x = r.start
ax.set_yscale("symlog", linthresh=10); ax.set_ylim(0, 4e3); ax.set_xlim(0, 1095)
ax.set_xlabel("Время, сутки"); ax.set_ylabel("Потеряно объектов (накопленно)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Потери данных за 3 года (S2): красный пунктир — шок с уничтожением данных, оранжевый — простой кластера дольше T_rep", fontsize=9.5)
names = ["uniform13", "uniform15", "oracle13"]
mets = [("Потеряно объектов", "Потеряно объектов", "{:.0f}"),
        ("Средняя избыточность n/k", "Избыточность n/k", "{:.2f}"),
        ("Трафик восстановления в год", "Трафик восстановления,\nтыс. ед. объёма в год", "{:.0f}")]
for j, (key, title, fmt) in enumerate(mets):
    a = fig.add_axes([0.07 + j * 0.32, 0.07, 0.25, 0.27])
    vals = [R[n]["metrics"][key] / (1000 if "Трафик" in key else 1) for n in names]
    a.bar(range(3), vals, color=[COL[n] for n in names])
    for i, v in enumerate(vals):
        a.text(i, v * 1.02, fmt.format(v), ha="center", fontsize=8)
    a.set_xticks(range(3)); a.set_xticklabels(["Равн.\nn=13", "Равн.\nn=15", "Оракул\nn=13"], fontsize=8)
    a.set_title(title, fontsize=9); a.set_ylim(0, max(vals) * 1.18)
fig.savefig(f"{OUT}/H_bounds.png", dpi=150); plt.close(fig)

tab = pd.DataFrame({LABEL[n]: R[n]["metrics"] for n in names}).drop(index=["Политика", "Объектов", "Время прогона, с"])
print(tab.to_string())
