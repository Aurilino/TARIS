"""Проверка экономического порога тревог (шаг 5, Э4).

1. Критерий на заданных эпизодах: порог совпадает с перебором вручную; если перенос дороже выгоды — «не тревожить».
2. Выгода и цена по формулам Э1 на простом снимке совпадают с расчётом вручную.
3. Прогон S6 (основной мир, горизонт days суток): эквивалентный kappa по группам; цена переносов по ложным
   тревогам в модели против трафика переносов в журнале (порядок величины); утечки истины нет — агент
   считает только по view и своей карте (в коде policy._value и _snapshot нет обращений к world).
Побитовое совпадение режима с числом kappa с кодом до Э — отдельно (--save / --compare, как check_substeps.py).

    python scripts/check_econ.py [days, по умолчанию 365] [unit] [--save|--compare файл]
"""
import sys, os, json, hashlib, inspect
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from taris_sim.config import scenario
from taris_sim.scenario import MAIN
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import AgentCrisisPolicy

FAIL = []


def ok(cond, msg):
    print(("ок  " if cond else "НЕТ ") + msg)
    if not cond:
        FAIL.append(msg)


def world(days, hours=24, seed=1):
    cfg = scenario("S6", seed=seed, alarm_step_hours=hours, horizon_days=days, **MAIN)
    w = World(cfg); ch = Chain(w)
    return w, Ledger(w, ch, DataStream(cfg, len(w.nodes)))


def digest(res, pol):
    h = hashlib.sha256()
    for df in (res.migrations, res.alarm_episodes(), res.repairs):
        h.update(df.to_csv(index=False).encode())
    h.update(repr(pol.thr_log).encode())
    return h.hexdigest()


def bitwise(days, mode, path):
    out = {}
    for hours in (24, 1):
        w, L = world(200, hours)      # при горизонте < 200 суток мир S6 не строится (пустая выборка кризисов)
        pol = AgentCrisisPolicy(w, 13, fast=hours < 24)
        res = L.run(pol)
        out[str(hours)] = digest(res, pol)
        print(f"  шаг {hours} ч: {out[str(hours)][:16]}")
    if mode == "--save":
        json.dump(out, open(path, "w"))
        print("эталон записан:", path)
    else:
        base = json.load(open(path))
        for k in out:
            ok(out[k] == base[k], f"kappa = 0,5, шаг {k} ч: побитово как до Э")


def unit():
    w, L = world(200)
    pol = AgentCrisisPolicy(w, 13, kappa="econ")
    pol.fc.cum_day = 0                                  # фон задан вручную: 2 долгих простоя на 2000 узло-суток
    pol.fc.cum["longst"][:, 0] = 0.0; pol.fc.cum["longst"][0, 0] = 1.5
    pol.fc.cum["up"][:, 0] = 0.0; pol.fc.cum["up"][0, 0] = 2000.0 * pol.fc.cpd
    snap = {1: 100, 2: 10}
    B, C = pol._value("dc", snap)
    q, k, n = 0.5, pol.k, pol.n
    lam = 2.0 / 2000.0
    p = 1 - np.exp(-lam * pol.w_loss)
    from scipy.stats import binom
    pl = lambda m: binom.sf(m - k, m, p)
    R = 100 * ((1 - (1 - q)) + q / k) + 10 * ((1 - (1 - q) ** 2) + 2 * q / k)
    dL = 100 * (0.5 * pl(n) + 0.5 * pl(n - 1) - pl(n)) + 10 * (0.25 * pl(n) + 0.5 * pl(n - 1) + 0.25 * pl(n - 2) - pl(n))
    ok(abs(C - 0.1 * 120 / k) < 1e-12, f"цена переноса C = {C:.3f} = c_tr * F / k")
    ok(abs(B - (0.1 * R + 1e4 * dL)) < 1e-9 * max(B, 1), f"выгода B = {B:.3f} = c_tr * R + W * dL")
    # критерий: перебор вручную
    rng = np.random.default_rng(0)
    pol.labels["dc"] = [(float(rng.uniform(2, 25)), bool(rng.random() < 0.3), {1: int(rng.integers(5, 50))}) for _ in range(40)]
    pol.q_hit["dc"] = [30.0, 40.0]
    pol._learn(0.0)
    L_ = pol.labels["dc"]
    BC = [pol._value("dc", v[2]) for v in L_]
    grid = pol.GRID + (np.inf,)
    gain = [sum((v[1] * b - c) for v, (b, c) in zip(L_, BC) if v[0] >= g) for g in grid]
    best = max(i for i in range(len(grid)) if gain[i] >= max(gain) - 1e-12)
    ok(pol.thr["dc"] == grid[best], f"порог по перебору: {pol.thr['dc']}")
    # перенос дороже выгоды — не тревожить
    pol.labels["dc"] = [(10.0, True, {1: 10})] * 3 + [(10.0, False, {1: 10})] * 50
    pol.q_hit["dc"] = [0.0, 1000.0]
    pol._learn(0.0)
    ok(pol.thr["dc"] == np.inf, f"перенос дороже выгоды: порог {pol.thr['dc']}")
    src = inspect.getsource(AgentCrisisPolicy._value) + inspect.getsource(AgentCrisisPolicy._snapshot) + \
        inspect.getsource(AgentCrisisPolicy._hit_share)
    ok("world" not in src and ".L." not in src, "в расчёте цены нет обращений к миру и журналу")


def run(days):
    w, L = world(days)
    pol = AgentCrisisPolicy(w, 13, kappa="econ")
    res = L.run(pol)
    print("эквивалентный kappa (последние записи по группам):")
    last = {}
    for t, lab, c, kap, Bm, Cm in pol.econ_log:
        last[lab] = (t, c, kap, Bm, Cm)
    for lab, (t, c, kap, Bm, Cm) in last.items():
        print(f"  {lab}: t={t:.0f}, порог {c}, kappa_экв {kap:.3f}, B {Bm:.2f}, C {Cm:.2f}, q {pol.q_hit[lab]}")
    ok(len(last) > 0, "агент выучил хотя бы один экономический порог")
    for lab in last:
        ok(0 < last[lab][4] and np.isfinite(last[lab][2]) or last[lab][1] == np.inf, f"{lab}: C > 0, kappa_экв конечен")
    # модельная цена переноса против журнала: средний перенос за поднятую тревогу
    ae = res.alarm_episodes()
    mg = res.migrations
    per = []
    for a in ae.itertuples():
        m = mg[(mg["t"] >= a.t0) & (mg["t"] <= a.until)]
        if len(m):
            inside = res._in_cluster(m["from"].to_numpy().astype(int), a.ctype, a.cluster, a.t0)
            per.append(float(m["traffic"].to_numpy()[inside].sum()))
    snaps = [v[2] for L_ in pol.labels.values() for v in L_]
    model = [sum(f * c for f, c in s.items()) / pol.k for s in snaps]
    if per and model:
        print(f"  трафик переноса на эпизод: в журнале {np.mean(per):.1f} (по {len(per)} тревогам), "
              f"по снимку {np.mean(model):.1f} (по {len(model)} эпизодам)")
        ok(0.2 < np.mean(per) / max(np.mean(model), 1e-9) < 5, "перенос на эпизод: модель и журнал одного порядка")
    s, _ = res.crisis_report()
    print(f"  полнота {s['Полнота тревог']:.2f}, точность {s['Точность тревог']:.2f}, потерь {len(res.losses)}, "
          f"трафик переносов {res.migrations['traffic'].sum():.0f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    days = int(args[0]) if args and args[0].isdigit() else 365
    if "--save" in args or "--compare" in args:
        mode = "--save" if "--save" in args else "--compare"
        bitwise(days, mode, args[args.index(mode) + 1])
    else:
        unit()
        if "unit" not in args:
            run(days)
    print("ИТОГ:", "всё в порядке" if not FAIL else f"не прошло {len(FAIL)}: " + "; ".join(FAIL))
