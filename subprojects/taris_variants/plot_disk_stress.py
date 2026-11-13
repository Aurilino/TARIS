"""Статическая научная иллюстрация одного стресс-пилота."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    output = Path(__file__).parent/"stress_evaluation_i5"
    source = output/"results.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    assert report["worlds"] == 1
    rows = {(r["mode"], r["profile"], r["variant"]): r for r in report["rows"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    variants = ("classic", "graph", "hybrid")
    for axis, profile, title in zip(axes, ("ample", "constrained"), ("Быстрый канал: 2 объёма/час", "Медленный канал: 0,02 объёма/час")):
        for offset, mode, label, color in ((-1, "control", "Без вмешательства", "#8c99a5"),
            (0, "close", "Волны через 0,5 суток", "#c45a45"), (1, "far", "Волны через 7 суток", "#337d93")):
            values = [rows[mode, profile, v]["losses"] for v in variants]
            bars = axis.bar(np.arange(3)+offset*.24, values, width=.24, label=label, color=color)
            axis.bar_label(bars, padding=3)
        axis.set_title(title)
        axis.set_xticks(range(3), ("Classic", "TARIS-G", "Hybrid"))
        axis.set_ylim(0, 9.5)
        axis.set_yticks(range(0, 10))
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Безвозвратно потерянных объектов из 45")
    fig.suptitle("Две волны разрушения дисков: один мир, одинаковые цели", fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .08), ncol=3, frameon=False)
    fig.text(.5, .025, "Решения раз в сутки; ошибка диска сообщается сразу. Пилот не оценивает вероятность катастрофы.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .17, 1, .93))
    fig.savefig(output/"losses.png", dpi=180)
    fig.savefig(output/"losses.svg")
    plt.close(fig)
    (output/"plot_metadata.json").write_text(json.dumps({"source_hash": hashlib.sha256(source.read_bytes()).hexdigest(),
        "plotter_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "worlds": 1,
        "interpretation": "paired_stress_pilot_not_agent_ranking"}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
