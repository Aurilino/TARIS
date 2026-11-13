"""Средние и диапазоны пяти миров; полосы не являются доверительными интервалами."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    output = Path(__file__).parent/"stress_sensitivity_i51"
    source = output/"results.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    assert report["worlds"] == 5
    rows = {(r["mode"], r["profile"], r["variant"]): r for r in report["summary"]}
    modes, gaps = ("close", "after_decision", "medium", "far"), (.5, 1.5, 3., 7.)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for axis, profile, title in zip(axes, ("ample", "constrained"), ("Быстрый канал", "Медленный канал")):
        for variant, name, color in (("classic", "Classic", "#326aa0"), ("graph", "TARIS-G", "#c9782c"),
                                      ("hybrid", "Hybrid", "#36836a")):
            group = [rows[mode, profile, variant] for mode in modes]
            axis.plot(gaps, [r["mean_losses"] for r in group], marker="o", color=color, label=name)
            axis.fill_between(gaps, [r["min_losses"] for r in group], [r["max_losses"] for r in group], color=color, alpha=.1)
        axis.axvline(1., color="gray", linestyle=":", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Интервал между волнами, сутки")
        axis.set_xticks(gaps)
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.2)
    axes[0].set_ylabel("Потерянных объектов из 45")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .07), ncol=3, frameon=False)
    fig.suptitle("Две волны: среднее и диапазон пяти миров", fontsize=13)
    fig.text(.5, .02, "Полосы — диапазон миров, не доверительный интервал. Пунктир — следующее суточное решение.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, .94))
    fig.savefig(output/"loss_sensitivity.png", dpi=180)
    fig.savefig(output/"loss_sensitivity.svg")
    plt.close(fig)
    metadata = {"source_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
        "plotter_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "worlds": 5,
        "bands": "min_max_not_confidence_interval"}
    (output/"plot_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
