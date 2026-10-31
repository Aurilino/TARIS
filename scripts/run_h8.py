"""Пакетный прогон H8 (шаг 5, Ч6): упреждение кризисов агентом.

Основной мир, сценарий S6 (библиотека кризисов), согласованный режим, n = 13, горизонт 3 года, зёрна 1..N.
Размещение у всех политик — оракул (сравнение изолирует прогноз); на одном зерне мир общий (общие случайные
числа), разности парные. Политики:
- none        — та же политика без прогноза (нижняя граница);
- agent_d     — агент, тревоги и перенос раз в сутки;
- agent_h     — агент, тревоги и перенос раз в час (alarm_step_hours = 1), порог учит сам;
- agent_h5    — то же, но пороги постоянные (дата-центр — 5, остальные — 10): что даёт сам прогноз;
- oracle_h    — оракул кризисов, раз в час (верхняя граница);
- agent_d_econ, agent_h_econ — агент раз в сутки и раз в час с экономическим порогом (kappa="econ", Э1–Э5).
Результат — runs/h8/<политика>_s<зерно>.pkl (готовые пропускаются), ход — runs/h8/progress.log.

    python scripts/run_h8.py [число потоков] [зёрна] [политики через запятую] [--tag=calibrated]
"""
import sys, os, time, pickle
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
TAG = next((x.split("=", 1)[1] for x in sys.argv[1:] if x.startswith("--tag=")), "")
OUT = os.path.join(ROOT, "runs", f"h8_{TAG}" if TAG else "h8")
N = 13
POLICIES = ("none", "agent_d", "agent_h", "agent_h5", "oracle_h", "agent_d_econ", "agent_h_econ")
HOURLY = ("agent_h", "agent_h5", "oracle_h", "agent_h_econ")
FIXED = {"dc": 5.0, "wear": 10.0, "exodus": 10.0, "software": 10.0}


def run(job):
    which, seed = job
    import numpy as np
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    from taris_sim.policy import AgentCrisisPolicy, CrisisOraclePolicy
    hourly = which in HOURLY
    cfg = scenario("S6", seed=seed, alarm_step_hours=1 if hourly else 24, **MAIN)
    w = World(cfg); ch = Chain(w); L = Ledger(w, ch, DataStream(cfg, len(w.nodes)))
    pol = {"none": lambda: AgentCrisisPolicy(w, N, forecast=False),
           "agent_d": lambda: AgentCrisisPolicy(w, N),
           "agent_h": lambda: AgentCrisisPolicy(w, N, fast=True),
           "agent_h5": lambda: AgentCrisisPolicy(w, N, fast=True, fixed_thr=FIXED),
           "oracle_h": lambda: CrisisOraclePolicy(w, N, fast=True),
           "agent_d_econ": lambda: AgentCrisisPolicy(w, N, kappa="econ"),
           "agent_h_econ": lambda: AgentCrisisPolicy(w, N, fast=True, kappa="econ")}[which]()
    t0 = time.time()
    res = L.run(pol)
    summ, tab = res.crisis_report()
    out = {"job": job, "metrics": res.metrics(), "crisis_summary": summ, "crisis_table": tab,
           "alarm_episodes": res.alarm_episodes(), "losses": res.losses,
           "migrations": len(res.migrations), "migration_traffic": float(res.migrations["traffic"].sum()) if len(res.migrations) else 0.0,
           "thr_log": getattr(pol, "thr_log", None), "econ_log": getattr(pol, "econ_log", None),
           "calibration_log": getattr(pol, "calibration_log", None),
           "q_hit": getattr(pol, "q_hit", None), "thr": getattr(pol, "thr", None),
           "labels": {k: [(float(x[0]), bool(x[1])) for x in v] for k, v in getattr(pol, "labels", {}).items()},
           "seconds": round(time.time() - t0)}
    with open(os.path.join(OUT, f"{which}_s{seed}.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    return job, out["seconds"], len(res.losses), summ.get("Полнота тревог"), summ.get("Точность тревог")


def keep_awake():
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


if __name__ == "__main__":
    keep_awake()
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    a, b = (sys.argv[2] if len(sys.argv) > 2 else "1-10").split("-")
    os.makedirs(OUT, exist_ok=True)
    # часовые (долгие) — первыми
    pol_arg = next((x for x in sys.argv[3:] if not x.startswith("--")), None)
    pols = pol_arg.split(",") if pol_arg else POLICIES
    jobs = sorted([(p, s) for s in range(int(a), int(b) + 1) for p in pols], key=lambda j: -(j[0] in HOURLY))
    todo = [j for j in jobs if not os.path.exists(os.path.join(OUT, f"{j[0]}_s{j[1]}.pkl"))]
    log = open(os.path.join(OUT, "progress.log"), "a", encoding="utf-8")
    print(f"{time.strftime('%Y-%m-%d %H:%M')} к расчёту {len(todo)} прогонов, потоков {workers}", file=log, flush=True)
    t0, done = time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(run, j) for j in todo]):
            job, sec, lost, rec, prec = f.result()
            done += 1
            msg = (f"{time.strftime('%H:%M')} [{done}/{len(todo)}] {job[0]}_s{job[1]}: {sec} с, потеряно {lost}, "
                   f"полнота {rec}, точность {prec}; прошло {(time.time() - t0) / 60:.0f} мин")
            print(msg, file=log, flush=True)
            print(msg, flush=True)
