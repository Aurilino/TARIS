"""I6.2: парные изменения потерь и хвоста, без смешивания критериев."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np


def main():
    folder = Path(__file__).parent/"queue_validation_i62"
    path = folder/"results.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["worlds"] == 3
    rows = [r for r in report["paired_contrasts"] if r["comparison"] == "pressure_minus_legacy"]
    variants, stresses, profiles = ("classic", "graph", "hybrid"), ("close", "far"), ("ample", "constrained")
    matrices = {}
    for metric in ("loss_difference", "unfinished_difference"):
        for profile in profiles:
            matrices[metric, profile] = np.array([[sum(r[metric] for r in rows if
                (r["variant"], r["stress_mode"], r["profile"]) == (variant, stress, profile))/3
                for stress in stresses] for variant in variants])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for row_index, (metric, label) in enumerate((("loss_difference", "Изменение числа потерь"),
                                                ("unfinished_difference", "Изменение хвоста заданий"))):
        scale = max(1., max(float(np.abs(matrices[metric, profile]).max()) for profile in profiles))
        norm = TwoSlopeNorm(vcenter=0., vmin=-scale, vmax=scale)
        for col_index, profile in enumerate(profiles):
            axis = axes[row_index, col_index]
            values = matrices[metric, profile]
            plotted = axis.imshow(values, cmap="RdYlGn_r", norm=norm, aspect="auto")
            axis.set_xticks((0, 1), ("0,5 суток", "7 суток"))
            axis.set_yticks((0, 1, 2), ("Classic", "TARIS-G", "Hybrid"))
            axis.set_title(("Быстрый" if profile == "ample" else "Медленный")+" канал\n"+label)
            axis.set_xlabel("Интервал волн")
            for y in range(3):
                for x in range(2):
                    value = values[y, x]
                    cell_label = "0.00" if abs(value) < 1e-12 else f"{value:+.2f}"
                    color = "white" if abs(value) > .65*scale else "black"
                    axis.text(x, y, cell_label, ha="center", va="center", color=color)
        fig.colorbar(plotted, ax=list(axes[row_index]), shrink=.8, label="Среднее парное изменение")
    fig.suptitle("Новые правила минус исходные: три новых мира", fontsize=13)
    fig.supxlabel("Зелёное: меньше потерь/хвоста. Красное: больше. Средние трёх миров, не доверительные интервалы.\n"
                  "Хвост — принятые незавершённые заявки; состав и объём потребности могут различаться.", fontsize=9)
    fig.savefig(folder/"paired_effects.png", dpi=180)
    fig.savefig(folder/"paired_effects.svg")
    plt.close(fig)
    (folder/"plot_metadata.json").write_text(json.dumps({
        "source_hash": hashlib.sha256(path.read_bytes()).hexdigest(),
        "plotter_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison": "pressure_minus_legacy", "worlds": 3,
        "loss_and_queue_separate": True}, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
