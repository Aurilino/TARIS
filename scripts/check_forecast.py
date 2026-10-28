"""Проверка счёта аномалий агента (шаг 5, П5): различает ли он окна предвестника кризисов пункта 3.9.

Основной мир, сценарий S6, несколько зёрен; размещение — оракул (на счёт не влияет). Каждую эпоху агент считает
счёт S по всем группам (taris_sim/forecast.py) только по данным сети. Истина мира берётся лишь для разметки:
- «предвестник» — эпохи в последние сутки окна [w0, tau) предсказуемых кризисов и ложных тревог (сигнал в
  данных есть у обоих; для износа и ухода — последние 7 суток окна);
- «внезапный» — последние сутки перед внезапным кризисом;
- «фон» — эпохи группы вне любых окон и кризисов этой группы.
Для каждого типа: AUC «предвестник против фона» и «внезапный против фона» (должна быть около 0,5),
доля окон, где счёт хоть раз превысил порог, и доля фоновых эпох выше порога.

    python scripts/check_forecast.py [число зёрен, по умолчанию 3] [шаг тревог в часах, по умолчанию 24]
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from taris_sim.config import scenario
from taris_sim.scenario import MAIN
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import OraclePolicy
from taris_sim.experience import Experience
from taris_sim.forecast import Forecaster, WINDOW

THR = (2.0, 3.0, 5.0)


class Logger(OraclePolicy):
    """Оракул размещает; агент рядом считает счёт при каждом вызове тревог (раз в сутки или каждый подшаг)."""

    def __init__(self, w, hours=24):
        super().__init__(w, 13)
        self.fast = hours < 24
        self.exp = Experience(w)
        self.fc = Forecaster(w, self.exp, step_hours=hours)
        self.rows = []

    def alarm(self, t, view):
        self.fc.update(t, view)
        for typ, ct, cl, S, O, E, g in self.fc.scores():
            self.rows.append((float(t), typ, ct, str(cl), S, O, E))
        return []


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return np.nan
    allv = np.r_[pos, neg]
    ranks = pd.Series(allv).rank().to_numpy()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def one(seed, hours=24):
    cfg = scenario("S6", seed=seed, alarm_step_hours=hours, **MAIN)
    w = World(cfg); ch = Chain(w); L = Ledger(w, ch, DataStream(cfg, len(w.nodes)))
    p = Logger(w, hours)
    t0 = time.time()
    L.run(p)
    sc = pd.DataFrame(p.rows, columns=["t", "type", "ctype", "cluster", "S", "O", "E"])
    # буря и каскад — одна группа (дата-центр), тип в счёте зависит от соседей; для разметки — общий
    sc["grp"] = np.where(sc["type"].isin(["storm", "cascade"]), "dc", sc["type"])
    cr = w.crises.copy()
    cr["grp"] = np.where(cr["type"].isin(["storm", "cascade", "fire"]), "dc", cr["type"])
    cr["ckey"] = cr["cluster"].astype(str)
    print(f"зерно {seed}: прогон {time.time() - t0:.0f} с, счётов {len(sc)}, кризисов {len(cr)}")
    out = []
    for r in cr.itertuples():
        g = sc[(sc.grp == r.grp) & (sc.cluster == r.ckey)]
        h = WINDOW.get(r.type, 1)
        last = g[(g.t >= r.tau - h) & (g.t < r.tau)]
        win = g[(g.t >= r.w0) & (g.t < r.tau)] if r.predictable else last
        kind = "ложная" if r.false_alarm else ("предсказуемый" if r.predictable else "внезапный")
        above = win[win.S > 5.0]
        out.append(dict(seed=seed, type=r.type, kind=kind, S_last=last.S.max() if len(last) else np.nan,
                        S_win=win.S.max() if len(win) else np.nan, W=r.tau - r.w0,
                        lead5=(r.tau - above.t.min()) * 24 if len(above) else np.nan))
    # фон: эпохи группы вне окон [w0 - 1, end + 2] любых эпизодов этой группы
    bad = []
    for r in cr.itertuples():
        bad.append((r.grp, r.ckey, np.floor(r.w0) - 1 - WINDOW.get(r.type, 1), np.ceil(r.end) + 2))
    keep = np.ones(len(sc), bool)
    for grp, ck, a, b in bad:
        keep &= ~((sc.grp == grp).to_numpy() & (sc.cluster == ck).to_numpy() & (sc.t >= a).to_numpy() & (sc.t <= b).to_numpy())
    bgd = sc[keep].assign(seed=seed)
    return pd.DataFrame(out), bgd[["seed", "grp", "S"]]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    hours = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    res = [one(s, hours) for s in range(1, n + 1)]
    ep = pd.concat([r[0] for r in res], ignore_index=True)
    bg = pd.concat([r[1] for r in res], ignore_index=True)
    grp_of = {"storm": "dc", "cascade": "dc", "fire": "dc", "wear": "wear", "software": "software", "exodus": "exodus"}
    rows = []
    for (typ, kind), g in ep.groupby(["type", "kind"]):
        neg = bg[bg.grp == grp_of[typ]].S.to_numpy()
        pos = g.S_last.dropna().to_numpy()
        row = {"тип": typ, "эпизоды": kind, "n": len(g), "AUC (последние сутки/неделя)": round(auc(pos, neg), 3),
               "до tau, ч (порог 5)": round(float(np.nanmedian(g.lead5)), 1) if g.lead5.notna().any() else np.nan}
        for c in THR:
            row[f"окно выше {c:g}"] = round(float((g.S_win > c).mean()), 2)
            row[f"фон выше {c:g}"] = round(float((neg > c).mean()), 4)
        rows.append(row)
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    print("\nФоновых эпох по группам:", bg.groupby("grp").size().to_dict())
