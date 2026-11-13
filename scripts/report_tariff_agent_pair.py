"""Сводка парного S2-сравнения тарифного и агентского размещения."""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
DATA_ROOT = os.path.join(ROOT, "runs")
DATA = os.path.join(DATA_ROOT, "tariff_agent_pair_365")


def read(seed, kind):
    with open(os.path.join(DATA, f"{kind}_s{seed}.pkl"), "rb") as f:
        return pickle.load(f)


def summary(metric):
    agent = np.array([read(s, "agent")["metrics"][metric] for s in range(1, 21)], float)
    tariff = np.array([read(s, "tariff")["metrics"][metric] for s in range(1, 21)], float)
    delta = tariff - agent
    lo, hi = stats.t.interval(0.95, len(delta) - 1, loc=delta.mean(), scale=stats.sem(delta))
    p = float(stats.wilcoxon(delta, zero_method="zsplit").pvalue)
    return agent.mean(), tariff.mean(), delta.mean(), lo, hi, p, delta


def main():
    global DATA
    scn = next((x for x in sys.argv[1:] if x in ("S0", "S2")), "S2")
    DATA = os.path.join(DATA_ROOT, "tariff_agent_pair_365" if scn == "S2" else "tariff_agent_pair_365_S0")
    files = [f for f in os.listdir(DATA) if f.endswith(".pkl")]
    if len(files) != 40:
        raise SystemExit(f"Ожидалось 40 файлов (20 пар), найдено {len(files)}")
    specs = [
        ("Потери объектов на зерно", "Потеряно объектов", 2),
        ("EAFDL", "EAFDL (среднее по годам)", 6),
        ("Ремонтный трафик в год", "Трафик восстановления в год", 1),
    ]
    lines = [f"# Тарифное размещение против агентского ({scn}, 20 зёрен)", "",
             "Знак разности: тариф минус агент. Парный 95%-й t-интервал и двусторонний Wilcoxon.", "",
             "| метрика | агент | тариф | разность | 95%-й интервал | Wilcoxon p |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, metric, digits in specs:
        a, t, d, lo, hi, p, diffs = summary(metric)
        fmt = f"{{:.{digits}f}}"
        lines.append(f"| {label} | {fmt.format(a)} | {fmt.format(t)} | {fmt.format(d)} | "
                     f"{fmt.format(lo)}…{fmt.format(hi)} | {p:.4g} |")
        if label == "Потери объектов на зерно":
            lines.append("")
            lines.append(f"Из 20 пар: тариф лучше в {(diffs < 0).sum()}, равен в {(diffs == 0).sum()}, "
                         f"хуже в {(diffs > 0).sum()}.")
            lines.append("")
    elig = np.mean([read(s, "tariff")["tariff_eligible_share"] for s in range(1, 21)])
    lines.extend(["", f"Средняя доля допуска узлов тарифом на последнем снимке: {elig:.1%}.", ""])
    report = "\n".join(lines)
    path = os.path.join(DATA, "report.md")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(report)
    print(report)
    print(f"\nСохранено: {path}")


if __name__ == "__main__":
    main()
