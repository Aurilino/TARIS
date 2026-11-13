"""Отчёт по сопоставимому сравнению TARIS и baseline-политик на MAIN."""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np
from scipy.stats import t as student_t


METRICS = ("Потеряно объектов", "EAFDL (среднее по годам)",
           "Трафик восстановления в год", "Средняя избыточность n/k",
           "Размещено не полностью")
BASELINES = (
    "uniform13", "crush13", "copysets13", "filecoin13", "sia13", "storj13",
    "dynamic_ec13", "aec13", "gbm13", "k1_pg13", "k2_laplace13", "k2_nuts13", "oracle13",
)
RESERVE = (
    "reserve_oracle", "reserve_uniform", "reserve_agent", "reserve_agent_greedy",
    "reserve_agent_capped", "reserve_tariff",
)


def stats(values):
    x = np.asarray(values, float)
    mean = float(x.mean())
    if len(x) < 2:
        return mean, float("nan"), float("nan")
    half = float(student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return mean, mean - half, mean + half


def main(root="runs/step5_comparison"):
    rows = {}
    diagnostics = {}
    meta = []
    for path in glob.glob(os.path.join(root, "S*", "*_s*.pkl")):
        with open(path, "rb") as fh:
            result = pickle.load(fh)
        m = result["metrics"]
        rows.setdefault((m["scenario"], m["policy"]), {})[int(m["seed"])] = m
        diagnostics.setdefault((m["scenario"], m["policy"]), {})[int(m["seed"])] = result.get("diagnostics", {})
        meta.append(result.get("meta", {}))
    scenarios = sorted({s for s, _ in rows})
    lines = ["# Шаг 5.4 — TARIS и текущие политики на общей конфигурации MAIN", "",
             f"Результатов: {len(meta)}. Ожидаемый полный объём: "
             f"{len(scenarios)} сценария × 10 зёрен × {len(BASELINES) + len(RESERVE)} политик.",
             "Конфигурация мира не переопределялась; метрики агрегированы по зерну.", ""]
    for scn in scenarios:
        lines += [f"## {scn}", "",
                  "Среднее ± SD по прогонам; EAFDL и трафик нормированы на год.", "",
                  "| Политика | n | Потери | EAFDL | Трафик/год | n/k | Неполных размещений |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        policies = sorted(p for s, p in rows if s == scn)
        for pol in policies:
            by_seed = rows[(scn, pol)]
            vals = [stats([m.get(metric, np.nan) for m in by_seed.values()]) for metric in METRICS]
            cells = [f"{v[0]:.5g} ± {np.std([m.get(metric, np.nan) for m in by_seed.values()], ddof=1):.3g}"
                     if len(by_seed) > 1 else f"{v[0]:.5g}"
                     for v, metric in zip(vals, METRICS)]
            lines.append(f"| `{pol}` | {len(by_seed)} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} |")
        lines.append("")
        nuts = list(diagnostics.get((scn, "k2_nuts13"), {}).values())
        if nuts:
            fit_times = [d["fit_seconds"] / max(d.get("fit_count", 1), 1)
                         for d in nuts if "fit_seconds" in d]
            rhats = [d["max_rhat"] for d in nuts if "max_rhat" in d and np.isfinite(d["max_rhat"])]
            esses = [d["min_ess_bulk"] for d in nuts if "min_ess_bulk" in d and np.isfinite(d["min_ess_bulk"])]
            divs = [d.get("divergences", 0) for d in nuts]
            lines += [f"K2-NUTS: суммарно {int(np.sum(divs))} дивергенций; "
                      f"максимальный R-hat последнего refit {max(rhats):.4f}; "
                      f"минимальный bulk-ESS {min(esses):.1f}; "
                      f"медианное время refit {np.median(fit_times):.1f} с.", ""]

        for reference, group, title in (
            ("uniform13", BASELINES + RESERVE, "Системное сравнение против uniform13"),
            ("reserve_uniform", RESERVE, "Контролируемое размещение против reserve_uniform (R_min=2)"),
        ):
            base = rows.get((scn, reference), {})
            if not base:
                continue
            lines += [f"### {title}", "",
                      "Разность policy − reference; 95%-й парный t-интервал по общим зёрнам.", "",
                      "| Политика | общих зёрен | Δ потерь | Δ EAFDL | Δ трафика/год |",
                      "|---|---:|---:|---:|---:|"]
            for pol in group:
                if pol == reference or (scn, pol) not in rows:
                    continue
                curr = rows[(scn, pol)]
                common = sorted(set(base) & set(curr))
                if not common:
                    continue
                out = []
                for metric in METRICS[:3]:
                    delta = [curr[seed][metric] - base[seed][metric] for seed in common]
                    mean, lo, hi = stats(delta)
                    out.append(f"{mean:.4g} [{lo:.4g}; {hi:.4g}]")
                lines.append(f"| `{pol}` | {len(common)} | {out[0]} | {out[1]} | {out[2]} |")
            lines.append("")

    text = "\n".join(lines)
    out = os.path.join(root, "report.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/step5_comparison")
