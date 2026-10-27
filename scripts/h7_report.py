"""Отчёт по прогонам H7 (scripts/run_h7.py): таблицы и график «трафик — потери».

По каждой группе (сценарий, интервал аудита) и политике, по зёрнам:
- потери объектов и EAFDL — среднее и 95% доверительный интервал (t-распределение);
- трафик восстановления в год — среднее и 95% ДИ; средняя избыточность n/k;
- CVaR годовой доли потерь на уровнях 95% и 99% по всем годам всех зёрен (при 20 зёрнах и 3 годах —
  60 значений: CVaR 99% — по сути худший год, CVaR 95% — среднее трёх худших) и доля лет с потерями;
- парные разности с полным TARIS-резервом по зёрнам (у политик на одном зерне общий мир) и критерий
  знаковых рангов Уилкоксона.
Пишет runs/h7/report.md и figures/h7_traffic_loss.png.

    python scripts/h7_report.py [eps]      # по умолчанию основной пакет (eps = 1e-3); eps — кривая Н6
"""
import sys, os, glob, pickle
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "runs", "h7")
NAMES = {"immediate": "немедленное", "lazy2": "порог R_min = 2", "markov": "марковский конкурент",
         "thiele": "TARIS-резерв", "thiele_nocorr": "TARIS без IBNR/RBNS", "thiele_const": "TARIS, постоянные интенсивности",
         "thiele_agent": "TARIS, оценки агента", "thiele_agent_ver": "TARIS, оценки агента (проверяемые)"}
ORDER = list(NAMES)


def load():
    rows, years = [], []
    for f in glob.glob(os.path.join(OUT, "*.pkl")):
        d = pickle.load(open(f, "rb"))
        scn, audit, which, seed, eps = d["job"]
        m = d["metrics"]
        rows.append({"scn": scn, "audit": audit, "policy": which, "seed": seed, "eps": eps,
                     "lost": m["Потеряно объектов"], "eafdl": m["EAFDL (среднее по годам)"],
                     "traffic": m["Трафик восстановления в год"], "ops": m["Операций восстановления"],
                     "nk": m["Средняя избыточность n/k"], "stalled": m["Объекто-суток без запроса восстановления"],
                     "sec": d["seconds"]})
        years += [{"scn": scn, "audit": audit, "policy": which, "seed": seed, "eps": eps, "year": y, "share": v}
                  for y, v in enumerate(d["year_loss_share"])]
    return pd.DataFrame(rows), pd.DataFrame(years)


def md(t: pd.DataFrame, index: bool = False) -> str:
    """Таблица в Markdown без внешних пакетов."""
    if index:
        t = t.reset_index()
    cols = [str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join(lines)


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else np.nan, np.nan
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def cvar(v, q):
    v = np.sort(np.asarray(v, float))[::-1]
    k = max(1, int(np.ceil(len(v) * (1 - q))))
    return float(v[:k].mean())


def fmt(m, h, p=1):
    return f"{m:.{p}f} ± {h:.{p}f}" if np.isfinite(h) else f"{m:.{p}f}"


def group_table(df, yr, scn, audit, eps):
    g = df[(df.scn == scn) & (df.audit == audit) & (df.eps == eps)]
    y = yr[(yr.scn == scn) & (yr.audit == audit) & (yr.eps == eps)]
    ref = g[g.policy == "thiele"].set_index("seed")
    rows = []
    for p in [x for x in ORDER if x in set(g.policy)]:
        a = g[g.policy == p].set_index("seed")
        ys = y[y.policy == p]["share"]
        row = {"политика": NAMES[p], "зёрен": len(a),
               "потеряно": fmt(*ci(a.lost)), "EAFDL, %": fmt(*(100 * np.array(ci(a.eafdl))), p=3),
               "трафик восст./год": fmt(*ci(a.traffic), p=0), "n/k": f"{a.nk.mean():.3f}",
               "CVaR95, %": f"{100 * cvar(ys, 0.95):.3f}", "CVaR99, %": f"{100 * cvar(ys, 0.99):.3f}",
               "лет с потерями": f"{(ys > 0).mean():.2f}"}
        if p != "thiele" and len(ref):
            s = a.index.intersection(ref.index)
            dt = a.loc[s, "traffic"] - ref.loc[s, "traffic"]
            dl = a.loc[s, "lost"] - ref.loc[s, "lost"]
            pt = stats.wilcoxon(dt).pvalue if (dt != 0).any() and len(s) > 1 else np.nan
            pl = stats.wilcoxon(dl).pvalue if (dl != 0).any() and len(s) > 1 else np.nan
            row["Δ трафика к TARIS"] = f"{fmt(*ci(dt), p=0)} (p = {pt:.3g})"
            row["Δ потерь к TARIS"] = f"{fmt(*ci(dl))} (p = {pl:.3g})"
        rows.append(row)
    return pd.DataFrame(rows)


def figure(df, eps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"immediate": "#6b7280", "lazy2": "#d97706", "markov": "#7c3aed", "thiele": "#0f766e",
              "thiele_nocorr": "#2563eb", "thiele_const": "#db2777",
              "thiele_agent": "#16a34a", "thiele_agent_ver": "#65a30d"}
    groups = [(s, a) for s in ("S0", "S2") for a in (1, 7) if len(df[(df.scn == s) & (df.audit == a) & (df.eps == eps)])]
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 3.8), squeeze=False)
    for ax, (s, a) in zip(axes[0], groups):
        g = df[(df.scn == s) & (df.audit == a) & (df.eps == eps)]
        for p in [x for x in ORDER if x in set(g.policy)]:
            x = g[g.policy == p]
            (mt, ht), (ml, hl) = ci(x.traffic), ci(x.lost)
            ax.errorbar(mt / 1000, ml, xerr=ht / 1000, yerr=hl, fmt="o", ms=7, lw=1.5, color=colors[p], label=NAMES[p])
        ax.set_title(f"{s}, аудит раз в {a} сут", fontsize=10)
        ax.set_xlabel("трафик восстановления, тыс. объёмов в год", fontsize=9)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("потеряно объектов за 3 года (из 4 380)", fontsize=9)
    axes[0][-1].legend(fontsize=8, frameon=False, loc="upper right")
    fig.suptitle(f"H7: трафик восстановления и потери, среднее и 95% ДИ по зёрнам (ε = {eps:g} за год)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    path = os.path.join(ROOT, "figures", "h7_traffic_loss.png" if eps == 1e-3 else f"h7_traffic_loss_e{eps:g}.png")
    fig.savefig(path, dpi=150)
    return path


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    df, yr = load()
    curve = len(sys.argv) > 1 and sys.argv[1] == "eps"
    out = []
    if curve:
        for eps in sorted(set(df.eps)):
            for p in ("thiele", "markov"):
                g = df[(df.scn == "S0") & (df.audit == 1) & (df.eps == eps) & (df.policy == p)]
                if len(g):
                    out.append({"eps": eps, "политика": NAMES[p], "зёрен": len(g), "потеряно": fmt(*ci(g.lost)),
                                "трафик восст./год": fmt(*ci(g.traffic), p=0)})
        text = md(pd.DataFrame(out))
    else:
        parts = []
        for s in ("S0", "S2"):
            for a in (1, 7):
                t = group_table(df, yr, s, a, 1e-3)
                if len(t):
                    parts.append(f"### {s}, аудит раз в {a} сут\n\n" + md(t))
        text = "\n\n".join(parts) + f"\n\nГрафик: {figure(df, 1e-3)}\n\nВремя прогонов, с (среднее по политике и сценарию):\n\n" + \
            md(df.groupby(["scn", "audit", "policy"]).sec.mean().round().astype(int).unstack(), index=True)
    print(text)
    with open(os.path.join(OUT, "report_eps.md" if curve else "report.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
