"""Научный статический график условной цены по проверенной таблице I4.5b."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    folder = Path(__file__).resolve().parent
    directory = folder/"resource_cost_evaluation_i45b"
    source = directory/"results.json"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    data = json.loads(source.read_text(encoding="utf-8"))
    assert data["legacy_metrics_verified"] and data["scope"] == "observed_resources_only"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharey=True)
    x = np.geomspace(1., 100., 150)
    colors = {"classic": "#2563eb", "graph": "#15803d", "hybrid": "#c2410c"}
    names = {"classic": "Classic", "graph": "TARIS-G", "hybrid": "Hybrid"}
    cohorts = (("anchor_i44a", "I4.4a: без обходов"), ("paired_i44b", "I4.4b: старейший proof"),
               ("paired_i44c", "I4.4c: адресный proof"))
    crossings = []
    for axis, (cohort, title) in zip(axes, cohorts):
        coefficients = {}
        for variant in ("classic", "graph", "hybrid"):
            rows = [r for r in data["summary"] if r["cohort"] == cohort and r["variant"] == variant
                    and r["profile"] == "constrained" and r["receiver_ratio"] == 1.]
            row = next(r for r in rows if r["central_ratio"] == 1.)
            intercept = row["disk"]+row["receiver_prefix"]+row["channel"]
            slope = row["full_buffer"]+row["decoded_prefix"]
            coefficients[variant] = intercept, slope
            for r in rows:
                assert math.isclose(intercept+slope*r["central_ratio"], r["observed_resource_cost"], abs_tol=1e-8)
            axis.plot(x, intercept+slope*x, color=colors[variant], label=names[variant], linewidth=2)
            axis.scatter([r["central_ratio"] for r in rows], [r["observed_resource_cost"] for r in rows],
                         color=colors[variant], s=20)
        for i, first in enumerate(("classic", "graph", "hybrid")):
            for second in ("classic", "graph", "hybrid")[i+1:]:
                a1, b1 = coefficients[first]
                a2, b2 = coefficients[second]
                if abs(b1-b2) > 1e-14:
                    price = (a2-a1)/(b1-b2)
                    if 1 <= price <= 100:
                        crossings.append({"cohort": cohort, "first": first, "second": second, "central_ratio": price})
        axis.set_xscale("log")
        axis.set_xticks([1, 10, 100], labels=["1", "10", "100"])
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("Цена RAM / цена объёма диска", fontsize=9)
        axis.grid(alpha=.25)
    axes[0].set_ylabel("Наблюдённые расходы, усл. ед.", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .90), ncol=3, frameon=False)
    fig.suptitle("Расходы зависят от цены памяти", y=.98, fontsize=14)
    fig.text(.5, .035, "Миры 24–26, медленный канал. Ожидаемый ущерб и будущие обязательства не оценены.\n"
             "Запас разных номеров и хвост работ приведены отдельно в RESULTS.md.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .82))
    fig.savefig(directory/"memory_price_sensitivity.png", dpi=160)
    fig.savefig(directory/"memory_price_sensitivity.svg")
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    metadata = {"source_report_hash": digest, "plot_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "crossings": crossings, "scope": "observed_cost_only_not_policy_ranking"}
    (directory/"plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(crossings, ensure_ascii=False))


if __name__ == "__main__":
    main()
