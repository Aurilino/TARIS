"""Проверка подшагов тревог и переноса внутри эпохи (Ч3).

Основной мир, сценарий S6, зерно 1, оракул кризисов (n = 13).
1. Совместимость: хэши результатов прогона (потери, восстановления, переносы, тревоги, фрагменты) при
   alarm_step_hours = 24 совпадают с эталоном до появления подшагов (--save на старом коде, --compare на новом);
   так же совпадают прогоны «fast = False при шаге 1 ч» и «fast = True при шаге 24 ч».
2. Пустые подшаги: политика с fast = True и шагом 1 ч, которая в подшагах ничего не делает, даёт тот же прогон.
3. Без будущего: всё, что политика видит в подшагах (view.step, down_since, кандидаты), — это chain.observed,
   обрезанный на момент подшага; объединение view.step по всем вызовам — ровно наблюдения до конца прогона.
4. Оракул кризисов при шаге 1 ч против шага 24 ч: переносы по типам кризисов и момент первого переноса
   относительно начала окна предвестника; потерь не больше.

    python scripts/check_substeps.py --save runs/substep_base.json      # эталон (на коде до подшагов)
    python scripts/check_substeps.py --compare runs/substep_base.json   # все проверки
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from taris_sim.config import scenario
from taris_sim.scenario import MAIN
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import CrisisOraclePolicy

EPOCHS = 400
FAIL = []


def check(ok, msg):
    print(("  ок   " if ok else "  ОШИБКА ") + msg)
    if not ok:
        FAIL.append(msg)


def h(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df.reset_index(drop=True), index=False).values.tobytes()).hexdigest()[:16]


def run(hours=24, fast=None, policy_cls=CrisisOraclePolicy):
    cfg = scenario("S6", seed=1, alarm_step_hours=hours, **MAIN) if hours != 24 else scenario("S6", seed=1, **MAIN)
    w = World(cfg); ch = Chain(w); L = Ledger(w, ch, DataStream(cfg, len(w.nodes)))
    pol = policy_cls(w, 13) if fast is None else policy_cls(w, 13, fast=fast)
    t0 = time.time()
    r = L.run(pol, epochs=EPOCHS)
    return r, pol, ch, time.time() - t0


def hashes(r):
    fr = r.fragments.copy()
    fr["end"] = fr["end"].astype(str)
    return {"losses": h(r.losses), "repairs": h(r.repairs), "migrations": h(r.migrations),
            "alarms": h(r.alarms.astype({"cluster": str})), "fragments": h(fr), "daily": h(r.daily)}


class Quiet(CrisisOraclePolicy):
    """Быстрая политика, которая в подшагах ничего не делает (решения — только в целые моменты)."""

    def __init__(self, w, n, fast=True):
        super().__init__(w, n, fast=fast)

    def alarm(self, t, view):
        return super().alarm(t, view) if t == int(t) else []

    def migrate(self, t, view):
        return super().migrate(t, view) if t == int(t) else []


class Spy(CrisisOraclePolicy):
    """Записывает всё, что видит в подшагах, и ничего не делает в них."""

    def __init__(self, w, n, fast=True):
        super().__init__(w, n, fast=fast)
        self.seen = []

    def alarm(self, t, view):
        st = view.step
        if st is not None:
            self.seen.append((t, st, view.down_since(), view.candidates()))
        return super().alarm(t, view) if t == int(t) else []

    def migrate(self, t, view):
        return super().migrate(t, view) if t == int(t) else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--save"); ap.add_argument("--compare")
    a = ap.parse_args()
    r, pol, ch, sec = run(24, None) if a.save else run(24, False)
    base = hashes(r)
    if a.save:
        Path(a.save).write_text(json.dumps(base, indent=1), encoding="utf-8")
        print("эталон записан:", a.save, base)
        sys.exit(0)

    print("1. Совместимость с кодом до подшагов")
    ref = json.loads(Path(a.compare).read_text(encoding="utf-8"))
    check(base == ref, f"шаг 24 ч, fast = False: хэши прогона совпадают с эталоном ({sec:.0f} с)")
    r2, *_ = run(1, False)
    check(hashes(r2) == ref, "шаг 1 ч, fast = False: подшагов нет, прогон тот же")
    r3, *_ = run(24, True)
    check(hashes(r3) == ref, "шаг 24 ч, fast = True: подшагов нет, прогон тот же")

    print("2. Пустые подшаги")
    r4, _, _, sec4 = run(1, True, Quiet)
    check(hashes(r4) == ref, f"шаг 1 ч, fast = True, в подшагах ничего не делается: прогон тот же ({sec4:.0f} с)")

    print("3. Без будущего")
    r5, spy, ch5, _ = run(1, True, Spy)
    check(hashes(r5) == ref, "политика-наблюдатель не меняет прогон")
    o = ch5.observed
    bad_start = bad_end = bad_on = bad_down = bad_cand = 0
    n_start = 0
    starts = []
    L5 = r5._L
    for t, st, down, cand in spy.seen:
        s = st["outages_started"]; e = st["outages_ended"]; og = st["outages_ongoing"]
        bad_start += int(((s["start"] < st["t0"]) | (s["start"] >= t)).sum())
        bad_end += int(((e["end"] < st["t0"]) | (e["end"] >= t)).sum())
        ref_on = o[(o["start"] < t) & (o["end"] >= t)]
        bad_on += int(not np.array_equal(np.sort(og["node"].to_numpy()), np.sort(ref_on["node"].to_numpy())))
        dn = np.zeros(L5.N, bool); dn[ref_on["node"].to_numpy()] = True
        bad_down += int(not np.array_equal(~np.isnan(down), dn))
        bad_cand += int(np.isin(cand, ref_on["node"].to_numpy()).any())
        starts.append(s[["node", "start"]])
        n_start += len(s)
        if t == int(t):
            assert st["t1"] == t
    allst = pd.concat(starts, ignore_index=True).sort_values(["node", "start"]).reset_index(drop=True)
    ref_st = o[o["start"] < spy.seen[-1][0]][["node", "start"]].sort_values(["node", "start"]).reset_index(drop=True)
    print(f"  вызовов {len(spy.seen)}, начал простоев увидено {n_start}")
    check(bad_start == 0 and bad_end == 0, "начала и концы простоев в view.step — только в [прошлый вызов, t)")
    check(bad_on == 0, "идущие простои в view.step — ровно идущие в момент t по chain.observed")
    check(bad_down == 0, "down_since в подшаге — ровно идущие в момент t простои")
    check(bad_cand == 0, "кандидаты в подшаге не включают узлы в идущем простое")
    check(allst.equals(ref_st), "объединение view.step по всем вызовам = все начала простоев до последнего вызова, без повторов")

    print("4. Оракул кризисов: шаг 1 ч против 24 ч")
    r6, p6, *_ = run(1, True)
    w = r6._L.w
    cr = w.crises[(w.crises.tau < EPOCHS) & w.crises.predictable & ~w.crises.false_alarm]
    rows = []
    for lab, rr in (("24 ч", r), ("1 ч", r6)):
        mg = rr.migrations
        for c in cr.itertuples():
            tg = p6.targets[c.id]
            m = mg[mg["from"].isin(tg) & (mg.t >= c.w0) & (mg.t < c.tau)]
            rows.append(dict(шаг=lab, тип=c.type, W=c.tau - c.w0, переносов=len(m), удачных=int(m.ok.sum()),
                             первый=(m.t.min() - c.w0) if len(m) else np.nan))
    t = pd.DataFrame(rows)
    print(t.groupby(["тип", "шаг"]).agg(кризисов=("W", "size"), окно_сут=("W", "mean"), переносов=("переносов", "sum"),
                                        удачных=("удачных", "sum"), первый_после_w0_сут=("первый", "mean")).round(3).to_string())
    print(f"  всего переносов: 24 ч — {len(r.migrations)}, 1 ч — {len(r6.migrations)}; "
          f"потеряно объектов: 24 ч — {len(r.losses)}, 1 ч — {len(r6.losses)}")
    check(len(r6.losses) <= len(r.losses), "при шаге 1 ч потерь не больше")
    print("\nИТОГ:", "все проверки пройдены" if not FAIL else f"не пройдено {len(FAIL)}: " + "; ".join(FAIL))
