"""Графики модуля chain: как сеть видит простой (E) и зачем нужна франшиза (F)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats
from taris_sim.config import scenario, MINUTE, HOUR
from taris_sim.world import World
from taris_sim.chain import Chain

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C_PART, C_TOT, C_MICRO, C_NOISE, C_OK = "#E8A33D", "#C8423B", "#7FA7D9", "#9E9E9E", "#D9D9D9"

cfg = scenario("S2")
w = World(cfg); ch = Chain(w)
nd, td = w.nodes, ch.true_down
obs = ch.classify()

# ------------------------------------------------- E. Как сеть видит простой
WIN = 12 * HOUR
own = td[(td.source == "own") & ((td.end - td.start) > 1 * HOUR) & ((td.end - td.start) < 4 * HOUR)]
best = None
for r in own.itertuples():
    t0 = r.start - 4 * HOUR
    near = td[(td.node == r.node) & (td.start > t0) & (td.end < t0 + WIN)]
    noise = obs[(obs.node == r.node) & (obs.source == "noise") & (obs.start > t0) & (obs.end < t0 + WIN)]
    mic = near[near.source == "micro"]
    obs_mic = obs[(obs.node == r.node) & (obs.source == "micro") & (obs.start > t0) & (obs.end < t0 + WIN)]
    score = min(len(mic), 4) + 2 * min(len(noise), 1) + 2 * min(len(obs_mic), 1)
    if (best is None or score > best[0]) and len(near[near.source == "own"]) == 1:
        best = (score, r.node, t0)
    if score >= 8:
        break
_, node, t0 = best
t1 = t0 + WIN
P = cfg.ping
fig, ax = plt.subplots(figsize=(11, 4.2))
H = lambda t: (t - t0) / HOUR
# истина
for r in td[(td.node == node) & (td.end > t0) & (td.start < t1)].itertuples():
    col = C_MICRO if r.source == "micro" else (C_TOT if (r.end - r.start) > cfg.T_rep else C_PART)
    ax.barh(2, max(H(r.end) - H(r.start), 0.02), left=H(r.start), height=0.5, color=col)
# пинги
pts = ch.phase[node] + P * np.arange(np.ceil((t0 - ch.phase[node]) / P), np.floor((t1 - ch.phase[node]) / P) + 1)
o_n = obs[(obs.node == node) & (obs.end > t0) & (obs.start < t1)]
failed = np.zeros(len(pts), bool)
for r in o_n.itertuples():
    failed |= (pts >= r.start - 1e-12) & (pts < r.end - 1e-12)
ax.vlines(H(pts[~failed]), 0.8, 1.2, color=C_OK, lw=1)
ax.vlines(H(pts[failed]), 0.75, 1.25, color="#222222", lw=1.6)
# что видит сеть
last_x, lvl = -9, 0
for r in o_n.sort_values("start").itertuples():
    col = {"micro": C_NOISE, "partial": C_PART, "total": C_TOT}[r.cls]
    ax.barh(0, H(r.end) - H(r.start), left=H(r.start), height=0.5, color=col)
    lab = f"{r.dur / MINUTE:.0f} мин" if r.dur < HOUR else f"{r.dur / HOUR:.1f} ч"
    xm = H(r.start) + (H(r.end) - H(r.start)) / 2
    lvl = (lvl + 1) % 2 if xm - last_x < 0.6 else 0
    ax.text(xm, -0.42 - 0.18 * lvl, lab, ha="center", fontsize=7)
    last_x = xm
    if r.source == "noise":   # во «взгляде истины» узел в это время работал
        ax.plot(xm, 2, marker="x", color=C_NOISE, ms=7)
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Истина:\nсостояние узла", "Пинги\nкаждые 5 мин", "Видит сеть\n(франшиза 15 мин)"])
ax.set_ylim(-0.85, 2.6); ax.set_xlim(0, 12)
ax.set_xlabel("Часы")
ax.set_title(f"Как сеть видит простой: узел {node}, окно 12 часов")
ax.legend(handles=[Patch(color=C_PART, label="простой длиннее франшизы"), Patch(color=C_MICRO, label="микропрерывание (истина)"),
                   Line2D([], [], color="#222222", lw=1.6, label="проваленный пинг"), Line2D([], [], color=C_OK, lw=1, label="успешный пинг"),
                   Patch(color=C_NOISE, label="отсечено франшизой"),
                   Line2D([], [], color=C_NOISE, marker="x", ls="", label="шум сети: узел работал")],
          loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/E_observation.png", dpi=150); plt.close(fig)

# ------------------------------------------------- F. Зачем нужна франшиза
expo = (nd.alive_to - nd.alive_from).clip(lower=0).to_numpy()
keep = expo >= 180
true_rate = nd.lam_avg.to_numpy()
ds_min = [0, 10, 15, 20, 30, 60]
rows = []
o_all = ch.observed
src_grp = np.where(o_all.source.isin(["own", "shock", "storm", "leave"]), "real", o_all.source)
for dm in ds_min:
    sel = (o_all.dur > dm * MINUTE).to_numpy()
    cnt = pd.Series(1, index=o_all.node[sel]).groupby(level=0).sum().reindex(range(len(nd)), fill_value=0).to_numpy()
    rate = cnt / np.maximum(expo, 1e-9)
    rho = stats.spearmanr(rate[keep], true_rate[keep]).statistic
    by = pd.Series(src_grp[sel]).value_counts()
    years = expo.sum() / 365
    rows.append(dict(d=dm, real=by.get("real", 0) / years / len(nd) * len(nd) / (expo.sum() / 365) * years if False else by.get("real", 0) / (expo.sum() / 365),
                     micro=by.get("micro", 0) / (expo.sum() / 365), noise=by.get("noise", 0) / (expo.sum() / 365),
                     rho=rho, precision=by.get("real", 0) / max(sel.sum(), 1)))
F = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = np.arange(len(ds_min))
ax = axes[0]
ax.bar(x, F.real, color=C_PART, label="реальные простои")
ax.bar(x, F.micro, bottom=F.real, color=C_MICRO, label="микропрерывания")
ax.bar(x, F.noise, bottom=F.real + F.micro, color=C_NOISE, label="шум сети")
ax.set_yscale("log"); ax.set_ylim(5, 600)
ax.set_xticks(x); ax.set_xticklabels([f"{d}" for d in ds_min]); ax.set_xlabel("Франшиза d, минуты")
ax.set_ylabel("Наблюдаемых «страховых случаев»\nна узел в год (лог. шкала)")
for i, r in F.iterrows():
    ax.text(i, (r.real + r.micro + r.noise) * 1.12, f"{r.real + r.micro + r.noise:.0f}", ha="center", fontsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.set_title("Из чего состоят наблюдаемые случаи", fontsize=10)
ax = axes[1]
ax.plot(ds_min, F.rho, "o-", color="#2F6DB5", label="ранговая корреляция с истинным риском")
ax.plot(ds_min, F.precision, "s--", color=C_TOT, label="доля реальных простоев среди случаев")
ax.axvline(cfg.franchise_d_hours * 60, color="#555555", lw=1, ls=":"); ax.text(16, 0.55, "d = 15 мин\n(по умолчанию)", fontsize=8)
ax.set_ylim(0, 1.02); ax.set_xlabel("Франшиза d, минуты"); ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.12))
ax.set_title("Качество сигнала о риске узла", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/F_franchise.png", dpi=150); plt.close(fig)

# сопутствующие цифры
micro_share = ch.micro.assign(dur=ch.micro.end - ch.micro.start).groupby("node").dur.sum().reindex(range(len(nd)), fill_value=0).to_numpy() / np.maximum(expo, 1e-9)
rho_micro = stats.spearmanr(micro_share[keep], true_rate[keep]).statistic
print(F.round(3).to_string(index=False))
print(f"\nРанговая корреляция доли времени в микропрерываниях с истинным риском: {rho_micro:.3f}")
a = ch.audits
print(f"Аудиты: провалено {1 - a.passed_avail.mean():.2%} (узел недоступен {1 - a.available.mean():.2%}, шум {a.noise_fail.mean():.2%})")
