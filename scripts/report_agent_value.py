"""Парная фактическая оценка экономической ценности тревог из пилота W.

    python scripts/report_agent_value.py

Для каждого зерна сравнивает прогон без прогноза с тем же миром при экономическом
пороге. Оценка не считает перекрывающиеся кризисы отдельно: эффект берётся по
итоговым потерям всего мира и фактическому трафику переносов.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNS = os.path.join(ROOT, "runs", "agent_econ_sensitivity")
OUT = os.path.join(RUNS, "paired_value_report.md")
VALUES = (100, 1000, 10000)
C_TR = 0.1


def load(name, seed):
    path = os.path.join(RUNS, f"{name}_s{seed}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Нет результата {path}; сначала выполните H8 W-пилот")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def ci95(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    half = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() - half), float(x.mean() + half)


def main():
    seeds = sorted(int(p.rsplit("_s", 1)[1][:-4]) for p in os.listdir(RUNS)
                   if p.startswith("none_s") and p.endswith(".pkl"))
    if not seeds:
        raise FileNotFoundError(f"Нет пилотных прогонов в {RUNS}")
    baseline = {s: load("none", s) for s in seeds}
    lines = [
        "# Парная фактическая ценность экономического порога тревог",
        "",
        f"Общие зёрна: {', '.join(map(str, seeds))}. Стоимость трафика `c_tr={C_TR}`. "
        "Положительное Δ потерь означает, что агент сохранил больше объектов.",
        "Чистая выгода = W × Δ потерь − c_tr × трафик переносов.",
        "Это эффект политики на весь мир, а не причинная оценка отдельного кризиса.",
        "",
        "| W | Δ потерь по зёрнам | среднее Δ потерь (95% t-интервал) | трафик переносов по зёрнам | "
        "средняя чистая выгода |",
        "|---:|---:|---:|---:|---:|",
    ]
    for W in VALUES:
        deltas, traffic, net = [], [], []
        for seed in seeds:
            base = baseline[seed]
            agent = load(f"W{W}", seed)
            d = int(base["metrics"]["Потеряно объектов"] - agent["metrics"]["Потеряно объектов"])
            tr = float(agent["migration_traffic"])
            deltas.append(d); traffic.append(tr); net.append(W * d - C_TR * tr)
        lo, hi = ci95(deltas)
        interval = f"[{lo:.2f}; {hi:.2f}]" if np.isfinite(lo) else "н/д"
        lines.append(f"| {W} | {' / '.join(map(str, deltas))} | {np.mean(deltas):.2f} {interval} | "
                     f"{' / '.join(f'{x:.0f}' for x in traffic)} | {np.mean(net):.1f} |")
    lines += [
        "",
        "Интервалы при трёх зёрнах крайне ненадёжны и приведены только как индикатор разброса. "
        "Эти результаты не подтверждают ни калибровку модельной ΔL, ни преимущество агента. "
        "Следующий этап — больше общих зёрен; модельную выгоду следует сохранять на момент тревоги, "
        "а фактическую оценивать парно по итоговым исходам, не суммируя перекрывающиеся эпизоды.",
        "",
    ]
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines))
    print(f"Сводка записана: {OUT}")
    print("\n".join(lines[7:11]))


if __name__ == "__main__":
    main()
