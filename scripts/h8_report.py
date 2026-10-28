"""Отчёт по прогонам H8 (scripts/run_h8.py): упреждение кризисов агентом.

По политикам, среднее ± 95% ДИ по зёрнам: полнота и точность тревог, упреждение, потери на предсказуемых
и внезапных кризисах, потерянные объекты, трафик переносов (всего и по ложным тревогам). Полнота — по
типам кризисов. Кривая обучения агента — полнота и точность по годам прогона. Пишет runs/h8/report.md.

    python scripts/h8_report.py
"""
import os, glob, pickle
import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "runs", "h8")
NAMES = {"none": "без прогноза", "agent_d": "агент, раз в сутки", "agent_h": "агент, раз в час",
         "agent_h5": "агент, раз в час, порог 5", "oracle_h": "оракул кризисов, раз в час",
         "agent_d_econ": "агент, раз в сутки, эконом. порог", "agent_h_econ": "агент, раз в час, эконом. порог"}


def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2:
        return (float(x.mean()) if len(x) else np.nan), np.nan
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def fmt(m, h, p=2):
    return "—" if not np.isfinite(m) else (f"{m:.{p}f} ± {h:.{p}f}" if np.isfinite(h) else f"{m:.{p}f}")


def md(t):
    cols = [str(c) for c in t.columns]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                     ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)])


if __name__ == "__main__":
    runs = [pickle.load(open(f, "rb")) for f in sorted(glob.glob(os.path.join(OUT, "*.pkl")))]
    rows, by_type, years = [], [], []
    for d in runs:
        pol, seed = d["job"]
        s, tab, ae = d["crisis_summary"], d["crisis_table"], d["alarm_episodes"]
        rows.append(dict(pol=pol, seed=seed, rec=s["Полнота тревог"], prec=s["Точность тревог"],
                         lead=s["Упреждение, ч (среднее по замеченным)"], ep=s["Эпизодов тревог"],
                         lp=s["Потери на предсказуемом кризисе, доля"], ls=s["Потери на внезапном кризисе, доля"],
                         lost=len(d["losses"]), mig=d["migration_traffic"], fa=s["Трафик переносов по ложным тревогам"],
                         sec=d["seconds"]))
        pred = tab[tab["predictable"]]
        for typ, g in pred.groupby("type"):
            by_type.append(dict(pol=pol, seed=seed, type=typ, n=len(g), det=int((g["lead_h"] > 0).sum())))
        for y in range(3):
            g = pred[(pred["tau"] >= 365 * y) & (pred["tau"] < 365 * (y + 1))]
            a = ae[(ae["t0"] >= 365 * y) & (ae["t0"] < 365 * (y + 1))] if len(ae) else ae
            years.append(dict(pol=pol, seed=seed, year=y + 1, n=len(g), det=int((g["lead_h"] > 0).sum()),
                              ae=len(a), ok=int(a["correct"].sum()) if len(a) else 0))
    R, T, Y = pd.DataFrame(rows), pd.DataFrame(by_type), pd.DataFrame(years)
    order = [p for p in NAMES if p in set(R.pol)]
    main = []
    for p in order:
        g = R[R.pol == p]
        main.append({"политика": NAMES[p], "зёрен": len(g), "полнота": fmt(*ci(g.rec)), "точность": fmt(*ci(g.prec)),
                     "упреждение, ч": fmt(*ci(g.lead), p=1), "эпизодов тревог": fmt(*ci(g.ep), p=0),
                     "потери на предсказуемом, %": fmt(*(100 * np.array(ci(g.lp))), p=3),
                     "потери на внезапном, %": fmt(*(100 * np.array(ci(g.ls))), p=3),
                     "потеряно объектов": fmt(*ci(g.lost), p=1), "трафик переносов": fmt(*ci(g.mig), p=0),
                     "из них по ложным тревогам": fmt(*ci(g.fa), p=0), "время, с": int(g.sec.mean())})
    typ = T.groupby(["pol", "type"]).agg(n=("n", "sum"), det=("det", "sum")).reset_index()
    typ["полнота"] = (typ.det / typ.n).round(2)
    tt = typ.pivot(index="type", columns="pol", values="полнота")[order].rename(columns=NAMES).reset_index()
    tn = typ.groupby("type").n.max().rename("кризисов (на политику)").reset_index()
    tt = tn.merge(tt, on="type").rename(columns={"type": "тип"})
    yy = Y.groupby(["pol", "year"]).agg(n=("n", "sum"), det=("det", "sum"), ae=("ae", "sum"), ok=("ok", "sum")).reset_index()
    yy["полнота"] = (yy.det / yy.n.clip(lower=1)).round(2)
    yy["точность"] = (yy.ok / yy.ae.clip(lower=1)).round(2)
    yy = yy[yy.pol.isin(["agent_d", "agent_h", "agent_h5", "agent_d_econ", "agent_h_econ"])].assign(pol=lambda x: x.pol.map(NAMES))
    yy = yy.rename(columns={"pol": "политика", "year": "год", "n": "предсказуемых", "det": "замечено",
                            "ae": "эпизодов тревог", "ok": "верных"})
    text = ("### Сводка\n\n" + md(pd.DataFrame(main)) + "\n\n### Полнота по типам кризисов\n\n" + md(tt) +
            "\n\n### Кривая обучения (по годам прогона)\n\n" + md(yy))
    thr = [(d["job"], d["thr"]) for d in runs if d.get("thr") and d["job"][0] in ("agent_d", "agent_h", "agent_d_econ", "agent_h_econ")]
    if thr:
        text += "\n\n### Выученные пороги к концу прогона\n\n" + md(pd.DataFrame(
            [{"политика": NAMES[j[0]], "зерно": j[1], **{k: v for k, v in t.items()}} for j, t in thr]).sort_values(["политика", "зерно"]))
    ek = []
    for d in runs:
        for lab, last in {r[1]: r for r in (d.get("econ_log") or [])}.items():
            ek.append(dict(pol=d["job"][0], seed=d["job"][1], lab=lab, thr=last[2], kap=last[3], B=last[4], C=last[5],
                           q=(d["q_hit"][lab][0] + 1) / (d["q_hit"][lab][1] + 2)))
    if ek:
        E = pd.DataFrame(ek)
        g = E.groupby(["pol", "lab"]).agg(порог=("thr", "median"), kappa_экв=("kap", "median"), B=("B", "mean"),
                                          C=("C", "mean"), q=("q", "mean"), зёрен=("seed", "count")).round(3).reset_index()
        g["pol"] = g.pol.map(NAMES)
        text += ("\n\n### Экономический порог: к концу прогона (медиана порога и kappa_экв, средние B, C, q)\n\n" +
                 md(g.rename(columns={"pol": "политика", "lab": "группа"})))
    print(text)
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write(text + "\n")
