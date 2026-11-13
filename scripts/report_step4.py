"""Сводка парных прогонов шага 4 из runs/step4."""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np
from scipy.stats import t as student_t


METRICS = ("Потеряно объектов", "EAFDL (среднее по годам)", "Трафик восстановления в год",
           "Средняя избыточность n/k", "Размещено не полностью", "Операций восстановления")


def summarize(x):
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    if len(x) < 2:
        return mean, float("nan"), float("nan"), float("nan")
    se = float(np.std(x, ddof=1) / np.sqrt(len(x)))
    q = float(student_t.ppf(0.975, len(x) - 1))
    return mean, float(np.std(x, ddof=1)), mean - q * se, mean + q * se


def main(root="runs/step4"):
    files = glob.glob(os.path.join(root, "S*", "*_s*.pkl"))
    rows = {}
    diagnostics = {}
    for path in files:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        m = obj["metrics"]
        key = (str(m["scenario"]), str(m["policy"]))
        seed = int(m["seed"])
        rows.setdefault(key, {})[seed] = m
        diagnostics.setdefault(key, {})[seed] = obj.get("diagnostics", {})
    lines = ["# Шаг 4 — парные baseline-прогоны", "", f"Файлов результатов: {len(files)}", ""]
    for scn in sorted({k[0] for k in rows}):
        lines += [f"## {scn}", "", "| Политика | n | Потери | EAFDL | Трафик/год | Избыточность n/k | Неполных размещений |", "|---|---:|---:|---:|---:|---:|---:|"]
        for (s, policy), by_seed in sorted(rows.items()):
            if s != scn:
                continue
            metrics = [summarize([m.get(key, np.nan) for m in by_seed.values()]) for key in METRICS]
            def cell(v):
                mean, sd, lo, hi = v
                return f"{mean:.5g}" if not np.isfinite(lo) else f"{mean:.5g} ± {sd:.3g}"
            lines.append(f"| {policy} | {len(by_seed)} | {cell(metrics[0])} | {cell(metrics[1])} | {cell(metrics[2])} | {cell(metrics[3])} | {cell(metrics[4])} |")
        lines.append("")
        base = rows.get((scn, "uniform13"), {})
        if base:
            lines += ["Парные изменения против `uniform13` (policy минус uniform; зерно должно совпадать):", "",
                      "| Политика | общих зёрен | Δ потерь, 95% ДИ | Δ EAFDL, 95% ДИ | Δ трафика/год, 95% ДИ |",
                      "|---|---:|---:|---:|---:|"]
            for (s, policy), by_seed in sorted(rows.items()):
                if s != scn or policy == "uniform13":
                    continue
                common = sorted(set(base) & set(by_seed))
                if not common:
                    continue
                out = []
                for metric in METRICS[:3]:
                    delta = [by_seed[z][metric] - base[z][metric] for z in common]
                    mean, sd, lo, hi = summarize(delta)
                    out.append(f"{mean:.4g} [{lo:.4g}; {hi:.4g}]" if np.isfinite(lo) else f"{mean:.4g} (n=1)")
                lines.append(f"| {policy} | {len(common)} | {out[0]} | {out[1]} | {out[2]} |")
            lines.append("")
        for policy, key, label in (
            ("copysets13", "spill_count", "средний spillover Copysets на прогон"),
            ("dynamic_ec13", "target_n", "уровень избыточности Dynamic-EC по зёрнам"),
            ("storj13", "subnet_relaxations", "среднее число ослаблений подсети Storj на прогон"),
            ("aec13", "migration_candidates", "среднее число переносов AEC-DS на прогон"),
        ):
            vals = [d.get(key) for d in diagnostics.get((scn, policy), {}).values() if key in d]
            if vals:
                lines.append(f"- {label}: {np.mean(vals):.4g} (min {np.min(vals):.4g}, max {np.max(vals):.4g}).")
        nuts = [d for d in diagnostics.get((scn, "k2_nuts13"), {}).values()]
        if nuts:
            lines.append("- Для K2-NUTS R-hat и ESS отражают последний refit каждого зерна; число дивергенций суммируется по всем refit этого прогона.")
            def dvals(key):
                values = []
                for d in nuts:
                    if key not in d or not np.isfinite(d[key]):
                        continue
                    value = d[key]
                    if key == "fit_seconds":
                        value /= max(d.get("fit_count", 1), 1)
                    values.append(value)
                return np.asarray(values, float)
            for key, label, fmt in (
                ("fit_seconds", "медианное время одной NUTS-подгонки, с", "{:.1f}"),
                ("divergences", "расхождения NUTS суммарно по зернам", "{:.0f}"),
                ("max_rhat", "максимальный R-hat последнего refit по зёрнам", "{:.4f}"),
                ("min_ess_bulk", "минимальный bulk-ESS последнего refit по зёрнам", "{:.1f}"),
            ):
                vals = dvals(key)
                if len(vals):
                    if key == "fit_seconds":
                        value = np.median(vals)
                    elif key == "divergences":
                        value = np.sum(vals)
                    elif key == "min_ess_bulk":
                        value = np.min(vals)
                    else:
                        value = np.max(vals)
                    lines.append(f"- K2-NUTS: {label}: {fmt.format(value)}.")
        lines.append("")
    text = "\n".join(lines)
    path = os.path.join(root, "report.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/step4")
