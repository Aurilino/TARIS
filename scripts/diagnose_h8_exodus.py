"""Воспроизводимый разбор потерь агента на массовом уходе H8, зерно 3.

Повторяет тегированный пилот agent_h, сохраняет журнал переносов и связывает
переносы с объектами, потерянными в окне кризиса. Для точного повтора версия
счёта ухода принудительно возвращается к исходному 7-дневному окну.

    python scripts/diagnose_h8_exodus.py
"""
import os
import pickle
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from taris_sim.config import scenario
from taris_sim.scenario import MAIN
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import AgentCrisisPolicy
from taris_sim.forecast import WINDOW


def main():
    os.chdir(ROOT)
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    seed = 3
    expected = pickle.load(open("runs/h8_calibrated/agent_h_s3.pkl", "rb"))
    WINDOW["exodus"] = 7
    cfg = scenario("S6", seed=seed, alarm_step_hours=1, **MAIN)
    w = World(cfg)
    ch = Chain(w)
    ledger = Ledger(w, ch, DataStream(cfg, len(w.nodes)))
    policy = AgentCrisisPolicy(w, 13, fast=True)
    t0 = time.time()
    result = ledger.run(policy)
    summary, table = result.crisis_report()

    # Проверить, что диагностический повтор воспроизвёл исходный H8-прогон.
    assert result.losses.equals(expected["losses"]), "потери не совпали с сохранённым H8-прогоном"
    assert table.equals(expected["crisis_table"]), "таблица кризисов не совпала с сохранённым H8-прогоном"
    event = table[(table.type == "exodus") & (table.cluster.astype(str) == "1") & (table.lost > 0)].iloc[0]
    tau, end = float(event.tau), float(event.end + cfg.T_rep)
    alarms = result.alarm_episodes()
    same_alarm = alarms[(alarms.type == "exodus") & (alarms.cluster.astype(str) == "1") & (alarms.t0 >= tau)]
    alarm_t = float(same_alarm.t0.min()) if len(same_alarm) else float("nan")
    episode = next(e for e in w.crisis.episodes if e["id"] == int(event.id))
    target = episode["targets"]
    fr = result.fragments
    f_end = np.minimum(fr["t_to"].to_numpy(), fr["t_dead"].to_numpy())
    on_target = np.isin(fr["node"].to_numpy(), target) & (fr["t_from"].to_numpy() <= tau) & (f_end > tau)
    exposed = np.unique(fr.loc[on_target, "obj"].to_numpy()).astype(int)
    lost_t = result.objects["lost_t"].to_numpy()
    hit = exposed[(lost_t[exposed] >= tau) & (lost_t[exposed] <= end)]

    migrations = result.migrations
    window = migrations[(migrations.t >= tau - 21) & (migrations.t <= end)].copy()
    regions = w.nodes["region"].to_numpy()
    window["from_region"] = regions[window["from"].to_numpy().astype(int)]
    window["to_region"] = regions[window["to"].to_numpy().astype(int)]
    hit_moves = window[window.obj.isin(hit)]
    outdir = "runs/h8_diagnostics"
    os.makedirs(outdir, exist_ok=True)
    window.to_pickle(os.path.join(outdir, "agent_h_s3_exodus_migrations.pkl.gz"), compression="gzip")
    result.losses.to_pickle(os.path.join(outdir, "agent_h_s3_losses.pkl.gz"), compression="gzip")
    with open(os.path.join(outdir, "agent_h_s3_exodus_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"tau={tau:.6f}; первая тревога после tau={alarm_t:.6f}; конец окна учёта={end:.6f}; "
                f"потеряно по кризису={event.lost:.0f}; "
                f"экспонированных объектов={len(exposed)}; объекты потерь={len(hit)}\n")
        f.write(f"переносов в интервале={len(window)}; успешных={int(window.ok.sum())}; "
                f"трафик={window.traffic.sum():.1f}; переносов объектов потерь={len(hit_moves)}\n")
        f.write("переносы по направлениям регионов:\n")
        f.write(window.groupby(["from_region", "to_region"]).agg(n=("obj", "size"), traffic=("traffic", "sum"),
                ok=("ok", "sum")).to_string())
        f.write("\nпереносы объектов, потерянных в окне кризиса, по направлениям:\n")
        f.write(hit_moves.groupby(["from_region", "to_region"]).agg(n=("obj", "size"), traffic=("traffic", "sum"),
                ok=("ok", "sum")).to_string())
        f.write(f"\n\nпотери по кризису: {event.to_dict()}\nсводка: {summary}\nвремя прогона={time.time()-t0:.0f} с\n")
    print(open(os.path.join(outdir, "agent_h_s3_exodus_report.txt"), encoding="utf-8").read(), flush=True)


if __name__ == "__main__":
    main()
