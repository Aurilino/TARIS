"""Сводка чувствительности годового S2-тарифа к reward0, 20 парных зёрен."""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
LOW_HIGH = os.path.join(ROOT, "runs", "tariff_reward_sensitivity_365")
BASE = os.path.join(ROOT, "runs", "tariff_agent_pair_365")


def read(label, seed):
    folder = BASE if label == "base" else LOW_HIGH
    path = os.path.join(folder, f"{'tariff' if label == 'base' else label}_s{seed}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def stats_row(label, metric):
    x = np.array([read(label, s)["metrics"][metric] for s in range(1, 21)], float)
    return x.mean(), x.std(ddof=1)


def paired(label, metric):
    base = np.array([read("base", s)["metrics"][metric] for s in range(1, 21)], float)
    x = np.array([read(label, s)["metrics"][metric] for s in range(1, 21)], float)
    d = x - base
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    p = float(stats.wilcoxon(d, zero_method="zsplit").pvalue)
    return float(d.mean()), float(lo), float(hi), p


def main():
    if len([f for f in os.listdir(LOW_HIGH) if f.endswith(".pkl")]) != 40:
        raise SystemExit("Ожидалось 40 прогонов low/high")
    cases = ("low", "base", "high")
    metrics = (("Потеряно объектов", "потери/год", 2),
               ("EAFDL (среднее по годам)", "EAFDL", 6),
               ("Трафик восстановления в год", "трафик/год", 1))
    lines = ["# Чувствительность тарифа reward0 (S2, 20 зёрен)", "",
             "Годовые пары, S2, одинаковые зёрна; low/base/high: reward0=0,15/0,30/0,60.", "",
             "| reward0 | потери, среднее ± SD | EAFDL, среднее ± SD | трафик, среднее ± SD | допуск |",
             "|---:|---:|---:|---:|---:|"]
    for label, r0 in (("low", "0,15"), ("base", "0,30"), ("high", "0,60")):
        vals = []
        for metric, _, digits in metrics:
            mean, sd = stats_row(label, metric)
            vals.append(f"{mean:.{digits}f} ± {sd:.{digits}f}")
        elig = np.mean([read(label, s)["tariff_eligible_share"] for s in range(1, 21)])
        lines.append(f"| {r0} | {vals[0]} | {vals[1]} | {vals[2]} | {elig:.1%} |")
    lines.extend(["", "Разность с базовой ценой (вариант − base), парный 95%-й t-интервал и двусторонний Wilcoxon:", "",
                  "| цена | метрика | средняя разность | 95%-й интервал | Wilcoxon p |",
                  "|---:|---|---:|---:|---:|"])
    for label, r0 in (("low", "0,15"), ("high", "0,60")):
        for metric, label_m, digits in metrics:
            mean, lo, hi, p = paired(label, metric)
            lines.append(f"| {r0} | {label_m} | {mean:.{digits}f} | {lo:.{digits}f}…{hi:.{digits}f} | {p:.4g} |")
    report = "\n".join(lines) + "\n"
    path = os.path.join(LOW_HIGH, "report.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
    print(report, f"Сохранено: {path}", sep="\n")


if __name__ == "__main__":
    main()
