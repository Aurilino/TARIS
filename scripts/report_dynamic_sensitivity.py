"""Отчёт по чувствительности порогов Dynamic-EC."""
from __future__ import annotations

import glob
import os
import pickle

import numpy as np
from scipy.stats import t as student_t


METRICS = ("Потеряно объектов", "EAFDL (среднее по годам)",
           "Трафик восстановления в год", "Средняя избыточность n/k")


def ci(values):
    x = np.asarray(values, dtype=float)
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    half = float(student_t.ppf(.975, len(x) - 1) * sd / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return mean, sd, mean - half, mean + half


def load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)["metrics"]


def main(root="runs/step4_dynamic_sensitivity", base_root="runs/step4"):
    paths = glob.glob(os.path.join(root, "S*", "*_s*.pkl"))
    by_key = {}
    for path in paths:
        m = load(path)
        by_key.setdefault((m["scenario"], m["variant"]), {})[m["seed"]] = m
    variants = {"low": ("1%", "3%"), "base": ("2%", "5%"), "high": ("4%", "8%")}
    lines = ["# Чувствительность порогов Dynamic-EC", "",
             f"Завершено файлов: {len(paths)}. Сравнение парное по зерну с `uniform13` из общего прогона шага 4.",
             "Среднее ± SD; Δ — политика минус `uniform13`, 95% t-интервал.", ""]
    for scn in sorted({key[0] for key in by_key}):
        lines += [f"## {scn}", "",
                  "| Пороги средний/высокий | n | Потери, среднее ± SD | Δ потерь [95% ДИ] | EAFDL | Трафик/год | Избыточность n/k |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for variant, (mid, high) in variants.items():
            runs = by_key.get((scn, variant), {})
            common = sorted(runs)
            base = {seed: load(os.path.join(base_root, scn, f"uniform13_s{seed}.pkl")) for seed in common}
            if not common:
                continue
            loss = ci([runs[z][METRICS[0]] for z in common])
            delta = ci([runs[z][METRICS[0]] - base[z][METRICS[0]] for z in common])
            eafdl = ci([runs[z][METRICS[1]] for z in common])
            traffic = ci([runs[z][METRICS[2]] for z in common])
            redundancy = ci([runs[z][METRICS[3]] for z in common])
            lines.append(f"| {mid}/{high} | {len(common)} | {loss[0]:.4g} ± {loss[1]:.3g} | "
                         f"{delta[0]:.4g} [{delta[2]:.4g}; {delta[3]:.4g}] | {eafdl[0]:.4g} | "
                         f"{traffic[0]:.5g} | {redundancy[0]:.4g} ± {redundancy[1]:.3g} |")
        lines.append("")
        for left, right in (("low", "base"), ("high", "base")):
            a, b = by_key.get((scn, left), {}), by_key.get((scn, right), {})
            common_pair = sorted(set(a) & set(b))
            if common_pair:
                d = ci([a[z][METRICS[0]] - b[z][METRICS[0]] for z in common_pair])
                lines.append(f"Парное Δ потерь {left} − {right}: {d[0]:.4g} "
                             f"[95% ДИ {d[2]:.4g}; {d[3]:.4g}], n={len(common_pair)}.")
        lines.append("")
    report = "\n".join(lines)
    path = os.path.join(root, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
