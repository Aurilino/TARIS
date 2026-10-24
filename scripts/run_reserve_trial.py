"""Пробный прогон резерва информации (шаг 3, пункт 3.11; предварительный взгляд на H7).

Основной мир, согласованный режим, n = 16, k = 10, горизонт 3 года; сценарии S0 и S2. Политики:
- немедленное восстановление (оракул: порог R_min = n - k - 1);
- эвристический ленивый порог R_min = 2 (как в Total Recall и Carbonite);
- марковский конкурент: порог на объект по постоянной интенсивности, минимум трафика при P_loss <= eps;
- TARIS-резерв (Тиле): порог на объект по меняющимся интенсивностям, минимум стоимости при P_loss <= eps,
  запас с поправками IBNR и RBNS.
Интенсивности у всех — истинные (как у оракула). Результаты: runs/reserve/<сценарий>_<политика>.pkl и сводка.

    python scripts/run_reserve_trial.py [eps]
"""
import sys, os, time, pickle
from concurrent.futures import ProcessPoolExecutor
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

N = 16
EPS = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-3
POLS = ("immediate", "lazy2", "markov", "thiele")


def run(args):
    scn, which = args
    import numpy as np
    from collections import Counter
    from taris_sim.config import scenario
    from taris_sim.scenario import MAIN
    from taris_sim.world import World
    from taris_sim.chain import Chain
    from taris_sim.data import DataStream
    from taris_sim.ledger import Ledger
    from taris_sim.policy import ReservePolicy
    cfg = scenario(scn, **MAIN)
    w = World(cfg); ch = Chain(w); L = Ledger(w, ch, DataStream(cfg, len(w.nodes)))
    pol = {"immediate": lambda: ReservePolicy(w, N, "fixed"), "lazy2": lambda: ReservePolicy(w, N, "fixed", R_fixed=2),
           "markov": lambda: ReservePolicy(w, N, "markov", EPS), "thiele": lambda: ReservePolicy(w, N, "thiele", EPS)}[which]()
    t0 = time.time()
    res = L.run(pol)
    m = res.metrics()
    m["Время прогона, с"] = round(time.time() - t0)
    m["Пороги"] = dict(Counter(v[1] for v in pol.thr.values())) if pol.thr else {pol.R_fixed: "все"}
    os.makedirs(os.path.join(ROOT, "runs", "reserve"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "reserve", f"{scn}_{which}.pkl"), "wb") as fh:
        pickle.dump({"metrics": m, "losses": res.losses, "repairs": res.repairs, "daily": res.daily}, fh)
    return scn, which, m


if __name__ == "__main__":
    import pandas as pd
    pd.set_option("display.width", 220)
    jobs = [(s, p) for s in ("S0", "S2") for p in POLS]
    rows = []
    with ProcessPoolExecutor(max_workers=4) as ex:
        for scn, which, m in ex.map(run, jobs):
            print(scn, which, "готово", m["Время прогона, с"], "с", flush=True)
            rows.append({"сценарий": scn, "политика": m["Политика"], "потеряно": m["Потеряно объектов"],
                         "EAFDL": f"{m['EAFDL (среднее по годам)']:.2e}", "операций восст.": m["Операций восстановления"],
                         "трафик восст./год": round(m["Трафик восстановления в год"]),
                         "n/k": round(m["Средняя избыточность n/k"], 3), "пороги": m["Пороги"]})
    print(f"\nn = {N}, eps = {EPS:g} за 365 суток")
    print(pd.DataFrame(rows).to_string(index=False))
