"""Пакетный прогон H7 (шаг 3, пункт 3.11): резерв информации на многих зёрнах.

Основной мир, согласованный режим, n = 16, k = 10, eps = 1e-3 за 365 суток, горизонт 3 года, зёрна 1..20.
Обычные аудиты (раз в сутки), S0 и S2 — 6 политик: немедленное восстановление, эвристический порог R_min = 2,
марковский конкурент, TARIS-резерв, TARIS без IBNR/RBNS, TARIS с постоянными интенсивностями.
Аудит раз в 7 суток, S0 и S2 — 3 политики: немедленное, TARIS, TARIS без IBNR/RBNS.
На одном зерне мир у всех политик общий (общие случайные числа) — сравнение парное.

Результат каждого прогона — runs/h7/<сценарий>_a<интервал аудита>_<политика>_s<зерно>.pkl; готовые пропускаются,
поэтому прогон можно прервать и продолжить. Ход — в runs/h7/progress.log.

    python scripts/run_h7.py [число потоков, по умолчанию 4] [зёрна, например 1-20]
    python scripts/run_h7.py 4 1-20 eps        # Н6: кривая по eps (TARIS и марковский, S0, eps = 1e-2 и 1e-4)
    python scripts/run_h7.py 4 1-20 agent      # шаг 5, А9: TARIS-резерв на оценках агента (с заявленными признаками и без)
"""
import sys, os, time, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "runs", "h7")
N, EPS = 16, 1e-3
BASE = ("immediate", "lazy2", "markov", "thiele", "thiele_nocorr", "thiele_const")
RARE = ("immediate", "thiele", "thiele_nocorr")


def make(w, which, eps):
    from taris_sim.policy import ReservePolicy
    return {"immediate": lambda: ReservePolicy(w, N, "fixed"),
            "lazy2": lambda: ReservePolicy(w, N, "fixed", R_fixed=2),
            "markov": lambda: ReservePolicy(w, N, "markov", eps),
            "thiele": lambda: ReservePolicy(w, N, "thiele", eps),
            "thiele_nocorr": lambda: ReservePolicy(w, N, "thiele", eps, corr=False),
            "thiele_const": lambda: ReservePolicy(w, N, "thiele", eps, time_varying=False),
            "thiele_agent": lambda: ReservePolicy(w, N, "thiele", eps, rates="agent"),
            "thiele_agent_ver": lambda: ReservePolicy(w, N, "thiele", eps, rates="agent", claimed=False)}[which]()


def name(scn, audit, which, seed, eps):
    tag = "" if eps == EPS else f"_e{eps:g}"
    return f"{scn}_a{audit}_{which}{tag}_s{seed}"


def run(job):
    scn, audit, which, seed, eps = job
    import numpy as np
    from collections import Counter
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    cfg = scenario(scn, seed=seed, audit_interval_days=audit, **MAIN)
    w = World(cfg); ch = Chain(w); L = Ledger(w, ch, DataStream(cfg, len(w.nodes)))
    pol = make(w, which, eps)
    t0 = time.time()
    res = L.run(pol)
    m = res.metrics()
    d = res.daily
    # годовая доля потерь: потерянный объём за год / средний хранимый объём за год (для CVaR, пункт 3.6)
    size = res.objects["size"].to_numpy()
    lost_t = res.objects["lost_t"].to_numpy()
    years = []
    for y in range(int(np.ceil(len(d) / 365))):
        a, b = 365 * y, min(365 * (y + 1), len(d))
        lost = float(size[(lost_t >= a) & (lost_t < b)].sum())
        stored = float(d["objects_alive"].iloc[a:b].mean()) * float(size.mean())
        years.append(lost / stored if stored > 0 else 0.0)
    out = {"job": job, "metrics": m, "year_loss_share": years, "seconds": round(time.time() - t0),
           "thresholds": dict(Counter(v[1] for v in pol.thr.values())) if pol.thr else {pol.R_fixed: -1},
           "fits": getattr(pol, "fits", None),
           "losses": res.losses, "daily": d}
    with open(os.path.join(OUT, name(*job) + ".pkl"), "wb") as fh:
        pickle.dump(out, fh)
    return job, out["seconds"], m["Потеряно объектов"], m["Трафик восстановления в год"]


def jobs(seeds, curve=False, agent=False):
    out = []
    for seed in seeds:
        if agent:
            out += [(scn, 1, p, seed, EPS) for scn in ("S0", "S2") for p in ("thiele_agent", "thiele_agent_ver")]
            continue
        if curve:
            out += [("S0", 1, p, seed, e) for e in (1e-2, 1e-4) for p in ("thiele", "markov")]
            continue
        for scn in ("S0", "S2"):
            out += [(scn, 1, p, seed, EPS) for p in BASE]
            out += [(scn, 7, p, seed, EPS) for p in RARE]
    # долгие (TARIS на S2) — первыми, чтобы потоки не простаивали в конце
    slow = lambda j: (j[0] == "S2") * 2 + j[2].startswith("thiele")
    return sorted(out, key=lambda j: -slow(j))


def keep_awake():
    """Пока идёт пакет, просит Windows не засыпать от бездействия (как проигрыватель видео): настройки не меняются,
    запрос снимается вместе с процессом; закрытая крышка или ручной сон по-прежнему усыпляют компьютер."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


if __name__ == "__main__":
    keep_awake()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    a, b = (sys.argv[2] if len(sys.argv) > 2 else "1-20").split("-")
    seeds = range(int(a), int(b) + 1)
    curve = len(sys.argv) > 3 and sys.argv[3] == "eps"
    agent = len(sys.argv) > 3 and sys.argv[3] == "agent"
    os.makedirs(OUT, exist_ok=True)
    todo = [j for j in jobs(seeds, curve, agent) if not os.path.exists(os.path.join(OUT, name(*j) + ".pkl"))]
    log = open(os.path.join(OUT, "progress.log"), "a", encoding="utf-8")
    print(f"{time.strftime('%Y-%m-%d %H:%M')} к расчёту {len(todo)} прогонов, потоков {workers}", file=log, flush=True)
    print(f"к расчёту {len(todo)} прогонов", flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(run, j) for j in todo]
        for f in as_completed(futs):
            job, sec, lost, tr = f.result()
            done += 1
            msg = (f"{time.strftime('%H:%M')} [{done}/{len(todo)}] {name(*job)}: {sec} с, потеряно {lost}, "
                   f"трафик {tr:.0f} в год; прошло {(time.time() - t0) / 60:.0f} мин")
            print(msg, file=log, flush=True)
            print(msg, flush=True)
