"""Графики модуля world (шаг 3): шкала простоев, риск по возрасту, длительности, доля узлов в простое."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize
from taris_sim.config import scenario, HOUR
from taris_sim.world import World

OUT = sys.argv[1] if len(sys.argv) > 1 else "figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C_PART, C_TOT, C_SHOCK, C_LINE = "#E8A33D", "#C8423B", "#6C4FB3", "#BBBBBB"

worlds = {m: World(scenario("S2", world_mode=m)) for m in ("matched", "mismatched")}
TITLE = {"matched": "Согласованный мир", "mismatched": "Несогласованный мир"}

# ---------------------------------------------------------------- A. Шкала простоев
w = worlds["matched"]
nd, out, sh = w.nodes, w.outages, w.shocks
win = (0, 180)
sh_dc = sh[(sh.ctype == "dc") & (sh.start < win[1] - 10) & (sh.start > win[0] + 10)]
dc = int(sh_dc.cluster.iloc[0]) if len(sh_dc) else int(nd.dc.mode()[0])
alive_mid = (nd.alive_from < win[1]) & (nd.alive_to > win[0])
pick_dc = nd[(nd.dc == dc) & alive_mid].index[:12].tolist()
rng = np.random.default_rng(7)
others = nd[(nd.dc != dc) & alive_mid].index.to_numpy()
pick = pick_dc + rng.choice(others, 12, replace=False).tolist()

fig, ax = plt.subplots(figsize=(11, 6.5))
for y, i in enumerate(pick):
    a0, a1 = max(nd.alive_from[i], win[0]), min(nd.alive_to[i], win[1])
    ax.plot([a0, a1], [y, y], color=C_LINE, lw=1, zorder=1)
    if np.isfinite(nd.leave_t[i]) and nd.leave_t[i] < win[1]:
        ax.plot(nd.leave_t[i], y, marker="x", color="black", ms=7, zorder=4)
    o = out[(out.node == i) & (out.end > win[0]) & (out.start < win[1])]
    for r in o.itertuples():
        col = C_TOT if r.cls == "total" else C_PART
        ax.barh(y, max(r.end - r.start, 0.25), left=r.start, height=0.6, color=col,
                edgecolor=C_SHOCK if r.kind == "shock" else "none", lw=1.5, zorder=3)
prov, reg = int(w.dc_provider[dc]), int(w.dc_region[dc])
for r in sh[(sh.start > win[0]) & (sh.start < win[1])].itertuples():
    if (r.ctype, r.cluster) in (("dc", dc), ("provider", prov), ("region", reg)):
        ax.axvline(r.start, color=C_SHOCK, ls="--" if r.ctype == "dc" else ":", lw=1.2, zorder=0)
ax.axhline(len(pick_dc) - 0.5, color="black", lw=0.6)
ax.set_yticks(range(len(pick)))
ax.set_yticklabels([f"узел {i}" for i in pick], fontsize=7)
ax.text(win[0] + 1, len(pick_dc) / 2 - 0.5, f"дата-центр {dc}", rotation=90, va="center", fontsize=8, color=C_SHOCK)
ax.set_xlim(win); ax.invert_yaxis()
ax.set_xlabel("Время, сутки")
ax.set_title(f"Шкала простоев узлов (согласованный мир, S2): верхние {len(pick_dc)} узлов — из одного дата-центра")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color=C_PART, label="частичный ущерб (d < L ≤ T_rep)"),
                   Patch(color=C_TOT, label="полная гибель (L > T_rep)"),
                   Patch(facecolor="white", edgecolor=C_SHOCK, lw=1.5, label="простой от шока кластера"),
                   Line2D([], [], color="black", marker="x", ls="", label="уход из сети"),
                   Line2D([], [], color=C_SHOCK, ls="--", label="шок дата-центра"),
                   Line2D([], [], color=C_SHOCK, ls=":", label="шок провайдера или региона")],
          loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/A_timeline.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- B. Риск по возрасту
bands = np.array([0, 30, 60, 90, 180, 365, 730, 1095, 1460, 2190])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, (m, w) in zip(axes, worlds.items()):
    nd, raw = w.nodes, w.raw_own
    ages_ev = raw.start.to_numpy() - nd.join_t.to_numpy()[raw.node]
    n_ev = np.histogram(ages_ev, bands)[0]
    expo = np.zeros(len(bands) - 1); expn = np.zeros(len(bands) - 1)
    for i in range(len(nd)):
        a0, a1, j0 = nd.alive_from[i], nd.alive_to[i], nd.join_t[i]
        if a1 <= a0:
            continue
        t = np.arange(a0, a1, 1.0) + 0.5
        t = t[t < a1]
        if len(t) == 0:
            continue
        age = t - j0
        b = np.digitize(age, bands) - 1
        ok = (b >= 0) & (b < len(expo))
        np.add.at(expo, b[ok], 1.0)
        np.add.at(expn, b[ok], w.rate(np.full(ok.sum(), i), t[ok]))
    mid = np.sqrt(np.maximum(bands[:-1], 15) * bands[1:])
    rate_obs = n_ev / expo * 365; rate_exp = expn / expo * 365
    err = 1.96 * np.sqrt(n_ev) / expo * 365
    ax.errorbar(mid, rate_obs, yerr=err, fmt="o", color="#2F6DB5", ms=5, capsize=3, label="сгенерировано (95% интервал)")
    ax.plot(mid, rate_exp, "-", color="#222222", lw=1.5, label="заданная истинная интенсивность")
    ax.set_xscale("log"); ax.set_xlabel("Возраст узла в сети, сутки (лог. шкала)")
    ax.set_title(TITLE[m] + (": степенная зависимость (Вейбулл)" if m == "matched" else ": кривая-«ванна»"), fontsize=10)
axes[0].set_ylabel("Страховых случаев на узел в год")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/B_age_hazard.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- C. Длительности простоев
def cond_surv(w, i, l_h):
    c = w.cfg
    hw, u = int(w.nodes.hw_true.iat[i]), float(w.nodes.sev_u.iat[i])
    d = c.franchise_d_hours
    if c.world_mode == "matched":
        k, th = c.weibull_shape, c.weibull_scale_hours * np.exp(c.alpha_hw[hw] + u)
        return np.exp(-(l_h / th) ** k + (d / th) ** k)
    mu, s = np.log(c.lognorm_median_hours) + c.alpha_hw[hw] + u, c.lognorm_sigma
    return stats.norm.sf((np.log(l_h) - mu) / s) / stats.norm.sf((np.log(d) - mu) / s)

def weibull_trunc_fit(L, d):
    def nll(p):
        lth, lk = p; th, k = np.exp(lth), np.exp(lk)
        return -np.sum(np.log(k) - np.log(th) + (k - 1) * np.log(L / th) - (L / th) ** k + (d / th) ** k)
    r = optimize.minimize(nll, [np.log(np.median(L)), 0.0], method="Nelder-Mead", options=dict(maxiter=4000, xatol=1e-8, fatol=1e-8))
    return np.exp(r.x)

grid = np.logspace(np.log10(0.25), np.log10(24 * 30), 200)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
summary = []
for ax, (m, w) in zip(axes, worlds.items()):
    c = w.cfg
    L = w.raw_own.dur.to_numpy() / HOUR
    nodes_ev = w.raw_own.node.to_numpy()
    emp = np.array([(L > g).mean() for g in grid])
    uniq, cnt = np.unique(nodes_ev, return_counts=True)
    true = np.zeros_like(grid)
    for i, n_i in zip(uniq, cnt):
        true += n_i * cond_surv(w, i, grid)
    true /= cnt.sum()
    th, k = weibull_trunc_fit(L, c.franchise_d_hours)
    wfit = np.exp(-(grid / th) ** k + (c.franchise_d_hours / th) ** k)
    ax.plot(grid, emp, color="#2F6DB5", lw=2, label="сгенерировано")
    ax.plot(grid, true, color="#222222", lw=1.2, ls=":", label="заданное истинное распределение")
    ax.plot(grid, wfit, color=C_TOT, lw=1.2, ls="--", label=f"подгонка Вейбулла (форма {k:.2f})")
    for x, lab in ((c.franchise_d_hours, "d"), (c.T_rep_hours, "T_rep")):
        ax.axvline(x, color=C_LINE, lw=1); ax.text(x * 1.08, 0.9, lab, fontsize=9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.05)
    ax.set_xlabel("Длительность простоя l, часы"); ax.set_title(TITLE[m], fontsize=10)
    ax.text(30, 0.25, f"подгонка Вейбулла:\nформа {k:.2f}, масштаб {th:.1f} ч", fontsize=8, color=C_TOT)
    phi_emp = (L > c.T_rep_hours).mean()
    phi_w = np.exp(-(c.T_rep_hours / th) ** k + (c.franchise_d_hours / th) ** k)
    p7 = 24 * 7
    tail_emp = (L > p7).mean(); tail_w = np.exp(-(p7 / th) ** k + (c.franchise_d_hours / th) ** k)
    summary.append((TITLE[m], phi_emp, phi_w, tail_emp, tail_w))
axes[0].set_ylabel("P(L > l | L > d)")
axes[0].legend(frameon=False, fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/C_durations.png", dpi=150); plt.close(fig)

# ---------------------------------------------------------------- D. Доля узлов в простое
fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
tg = np.arange(0, worlds["matched"].cfg.horizon_days, HOUR)
for ax, (m, w) in zip(axes, worlds.items()):
    nd, out = w.nodes, w.outages
    down = np.zeros(len(tg))
    s_idx = np.searchsorted(tg, out.start.to_numpy()); e_idx = np.searchsorted(tg, out.end.to_numpy())
    delta = np.zeros(len(tg) + 1)
    np.add.at(delta, s_idx, 1); np.add.at(delta, e_idx, -1)
    down = np.cumsum(delta)[:-1]
    al = np.zeros(len(tg) + 1)
    np.add.at(al, np.searchsorted(tg, nd.alive_from.to_numpy()), 1)
    np.add.at(al, np.searchsorted(tg, nd.alive_to.to_numpy()), -1)
    alive = np.cumsum(al)[:-1]
    share = 100 * down / np.maximum(alive, 1)
    ax.plot(tg, share, color="#2F6DB5", lw=0.5)
    ax.set_ylabel("% узлов в простое")
    ax.set_title(TITLE[m] + (": мгновенные общие шоки" if m == "matched" else ": «бури» — простои пачкой"), fontsize=10)
    med = np.median(share); ax.axhline(med, color="#555555", lw=1, ls="--"); ax.text(tg[-1] * 1.005, med, f"медиана\n{med:.1f}%", fontsize=8, va="center")
axes[1].set_xlabel("Время, сутки")
fig.tight_layout(); fig.savefig(f"{OUT}/D_share_down.png", dpi=150); plt.close(fig)

print("Сводка по тяжести (доля простоев дольше T_rep и дольше недели):")
for t, pe, pw, te, tw in summary:
    print(f"  {t}: phi сгенерировано {pe:.3f}, по подгонке Вейбулла {pw:.3f}; хвост > 7 сут: сгенерировано {te:.4f}, Вейбулл {tw:.4f}")
