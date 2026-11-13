"""Объём неутраченных данных в конце серии H5/H6; без новых прогонов."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = [json.loads((ROOT / f"runs/h5_h6/main_{seed:02d}.json").read_text(encoding="utf-8"))
               for seed in range(1, 21)]
    policies = ("uniform", "agent", "protected")
    labels = ("Без выбора узлов по риску\nРавномерное размещение",
              "ИИ-агент\nРазмещение по риску", "ИИ-агент с защитой\nОграничения коалиций")
    colors = ("#718096", "#c47743", "#16877e")
    means, errors = [], []
    for policy in policies:
        values = []
        for record in records:
            row = next(r for r in record["runs"] if r["policy"] == policy and r["p_leak"] == .3)
            assert row["requested_objects"] == 90
            assert record["config"]["object_size"] == 1.
            values.append(row["requested_objects"] - row["total_losses"])
        values = np.asarray(values, dtype=float)
        means.append(float(values.mean()))
        errors.append(float(t.ppf(.975, 19) * values.std(ddof=1) / np.sqrt(20)))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    fig.subplots_adjust(left=.1, right=.96, top=.79, bottom=.27)
    x = np.arange(3)
    ax.bar(x, means, width=.58, color=colors, zorder=3)
    ax.errorbar(x, means, yerr=errors, fmt="none", ecolor="#253548", capsize=6, lw=1.6, zorder=4)
    for pos, mean, error in zip(x, means, errors):
        ax.text(pos, mean+error+2.5, f"{mean:.2f}".replace(".", ","), ha="center", va="bottom", fontsize=20, fontweight="bold", color="#253548")
        ax.text(pos, mean*.45, f"{100*mean/90:.1f}%".replace(".", ","), color="white", fontsize=15, ha="center", va="center")
    ax.axhline(90, color="#8593a3", ls="--", lw=1)
    ax.text(2.48, 90.9, "Поступило: 90 объектов", ha="right", va="bottom", fontsize=11, color="#596776")
    ax.set_ylim(0, 98)
    ax.set_yticks(np.arange(0, 91, 15))
    ax.set_xticks(x, labels, fontsize=11)
    ax.set_ylabel("Сохранившиеся данные, объектов", labelpad=12)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e2e8ee")
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=14)
    fig.text(.1, .94, "Сколько данных сохранилось к концу прогона", fontsize=20, fontweight="bold", color="#172d43")
    fig.text(.1, .875, "Сибилла: номинальная доля 30% · 120 суток · среднее по 20 общим мирам", fontsize=12, color="#536779")
    fig.text(.1, .115, "Один объект = одна условная единица исходных данных. Учитываются необратимые потери,\nа не объём резервных копий. Вертикальные отрезки — 95%-е интервалы по мирам.", fontsize=10, color="#536779", linespacing=1.6)
    fig.text(.1, .035, "S3 · код 10/13 · одинаковый порог восстановления · вероятность утечки связей 0,3\n«Без агента» в этом сравнении означает равномерный выбор узлов; правило восстановления общее.", fontsize=9, color="#687c8e", linespacing=1.5)
    output = ROOT / "figures"
    fig.savefig(output / "agent_data_comparison.png", dpi=180, facecolor="white")
    fig.savefig(output / "agent_data_comparison.svg", facecolor="white")
    plt.close(fig)
    print(dict(zip(policies, means)))


if __name__ == "__main__":
    main()
