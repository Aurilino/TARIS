"""Сводка полностью завершённой серии H5/H6; статистическая единица — зерно."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
import json
import math
import numpy as np
from scipy.stats import t as student_t, wilcoxon, beta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.run_h5_h6 import REVISION, code_hash

LABELS = {"uniform": "Равномерное", "agent": "Агент", "protected": "Полная защита",
          "novice_only": "Только лимит новичков", "no_novice": "Без лимита новичков",
          "no_penalty": "Без мягкого штрафа"}
METRICS = ("total_losses", "initial_failures", "later_losses", "repair_traffic", "allocated_fragment_days",
           "plaintext_objects", "ciphertext_objects", "full_initial_sets")


def summary(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return None
    mean = float(x.mean())
    error = float(student_t.ppf(.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x)>1 else 0.
    return {"n": len(x), "mean": mean, "lo": mean-error, "hi": mean+error,
            "min": float(x.min()), "max": float(x.max())}


def paired(left, right):
    difference = np.asarray(left, float)-np.asarray(right, float)
    difference[np.abs(difference) < 1e-8] = 0.
    value = summary(difference)
    value["wilcoxon_p"] = float(wilcoxon(difference).pvalue) if np.any(difference) else 1.
    return value


def bound_any(values):
    outcomes = np.asarray(values) > 0
    n, positive = len(outcomes), int(outcomes.sum())
    return {"worlds": n, "worlds_with_disclosure": positive,
            "one_sided_95_upper": float(beta.ppf(.95, positive+1, n-positive)) if positive<n else 1.}


def collect(directory):
    cases = {}
    expected_hash = code_hash()
    for group, count in (("main", 20), ("s4", 10), ("small", 10)):
        for seed in range(1, count+1):
            path = directory / f"{group}_{seed:02d}.json"
            if not path.exists():
                raise ValueError(f"Серия не завершена: нет {path}")
            record = json.loads(path.read_text(encoding="utf-8"))
            spec = record["specification"]
            if spec != dict(revision=REVISION, group=group, seed=seed, source_hash=expected_hash):
                raise ValueError(f"Не совпадает протокол или код: {path}")
            entries = {(r["policy"], r["p_leak"]): r for r in record["runs"]}
            if len(entries) != (9 if group=="main" else 3):
                raise ValueError(f"Неполный набор режимов: {path}")
            if any(r["requested_objects"] != 90 for r in entries.values()):
                raise ValueError(f"Изменился поток объектов: {path}")
            cases[group, seed] = record
    return cases


def run_values(cases, group, policy, metric, p=.3):
    output = []
    for (g, seed), record in sorted(cases.items()):
        if g == group:
            row = next(r for r in record["runs"] if r["policy"]==policy and r["p_leak"]==p)
            output.append(row[metric])
    return output


def number(value):
    return f"{value:.2f}"


def interval(value):
    return f"{value['mean']:+.2f} [{value['lo']:+.2f}; {value['hi']:+.2f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "runs/h5_h6")
    args = ap.parse_args()
    cases = collect(args.data)
    aggregates = {}
    contrasts = {}
    policies = ("uniform", "agent", "protected", "novice_only", "no_novice", "no_penalty")
    for group in ("main", "s4", "small"):
        entries = {}
        for policy in (policies if group=="main" else policies[:3]):
            entries[policy] = {m: summary(run_values(cases, group, policy, m)) for m in METRICS}
            entries[policy]["world_disclosure_bound"] = bound_any(run_values(cases, group, policy, "plaintext_objects"))
        aggregates[group] = entries
        contrasts[group] = {m: paired(run_values(cases, group, "protected", m),
                                      run_values(cases, group, "agent", m)) for m in METRICS}
    sensitivity = {}
    h6 = {}
    for p in (0., .1, .3, .6):
        sensitivity[str(p)] = {m: summary(run_values(cases, "main", "protected", m, p)) for m in METRICS}
        sensitivity[str(p)]["paired_vs_agent"] = {m: paired(run_values(cases, "main", "protected", m, p),
                                                             run_values(cases, "main", "agent", m)) for m in METRICS}
        selected = [next(d for d in record["h6"] if d["p_leak"]==p)
                    for (g, _), record in sorted(cases.items()) if g=="main"]
        h6[str(p)] = {"pair_sybil_recall": summary([d["pair"]["sybil_recall"] for d in selected]),
                     "pair_precision": summary([d["pair"]["precision"] for d in selected if d["pair"]["precision"] is not None]),
                     "cluster_sybil_recall": summary([d["ownership_cluster"]["sybil_recall"] for d in selected]),
                     "cluster_precision": summary([d["ownership_cluster"]["precision"] for d in selected if d["ownership_cluster"]["precision"] is not None])}
    summary_json = {"schema": "taris-h5h6-summary/1", "case_count": len(cases),
                    "policy_runs": sum(len(r["runs"]) for r in cases.values()),
                    "aggregates": aggregates, "paired_protected_minus_agent": contrasts,
                    "sensitivity": sensitivity, "h6": h6,
                    "agent_minus_uniform": {m: paired(run_values(cases, "main", "agent", m),
                                                       run_values(cases, "main", "uniform", m)) for m in METRICS},
                    "source_hash": code_hash(), "unit": "world seed; exploratory paired intervals"}
    (args.data / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    primary = aggregates["main"]
    loss_reduction = 100*(primary["agent"]["total_losses"]["mean"]-primary["protected"]["total_losses"]["mean"])/primary["agent"]["total_losses"]["mean"]
    repair_increase = 100*(primary["protected"]["repair_traffic"]["mean"]/primary["agent"]["repair_traffic"]["mean"]-1)
    lines = ["# H5/H6: результаты синтетической серии", "", "04.11.2026. Протокол: [[63_h5_h6_protocol]]. Реализация противника: [[62_adversary]].",
             "", "Серия завершена: 40 миров и 240 запусков политик. Основная серия — 20 общих зёрен S3 с долей Сибиллы 30%; дополнительные — по 10 зёрен S4 и S3 с долей 5%. Горизонт 120 суток, первые 30 суток без объектов, затем 90 объектов на мир. Код 10/13, порог ремонта 2. Все защитные параметры фиксированы до расчёта.",
             "", "## Основные выводы", "",
             f"В основной серии защита снизила средние потери с {primary['agent']['total_losses']['mean']:.2f} до {primary['protected']['total_losses']['mean']:.2f} объекта из 90 ({loss_reduction:.1f}%), а средний ремонтный трафик вырос на {repair_increase:.1f}%. Парный интервал потерь не включает ноль; в дополнительных сериях интервалы шире и включают ноль. Преимущество по полной стоимости не установлено.",
             "", "При слабых публичных связях одного лимита найденной группы недостаточно: реальный владелец может быть разбит на несколько кластеров. Защита сильнее помогает при более заметных утечках, но увеличивает обслуживание. В основной серии абляции без лимита новичков дают тот же результат, а без мягкого штрафа выигрыш по потерям исчезает. Это относится к данному протоколу с начальным наблюдением сети и не доказывает бесполезности лимита новичков при других регистрациях.",
             "", "Нулевое раскрытие в защищённой выборке не подтверждает крайне малую вероятность раскрытия: данных мало. H5 в полном объёме не подтверждена. H6 показывает зависимость восстановления владельца от силы заданных синтетических сигналов, а не внешнюю валидацию на реальных атаках.",
             "", "## Основная серия, утечка связей 0,3", "",
             "Средние по 20 зёрнам. Столбец потерь включает начальные отказы; ремонтный трафик дан за весь прогон, хранение — в выделенных фрагменто-днях.", "",
             "| Режим | Потери | Из них при старте | Ремонт | Фрагменто-дни | Раскрыто объектов |", "|---|---:|---:|---:|---:|---:|"]
    for policy in policies:
        a = aggregates["main"][policy]
        lines.append(f"| {LABELS[policy]} | {number(a['total_losses']['mean'])} | {number(a['initial_failures']['mean'])} | {number(a['repair_traffic']['mean'])} | {number(a['allocated_fragment_days']['mean'])} | {number(a['plaintext_objects']['mean'])} |")
    lines += ["", "## Парное сравнение полной защиты с агентом", "",
              "Разность: защита минус агент. 95%-й t-интервал и парный Wilcoxon по зёрнам; это описание ограниченной выборки, а не подтверждение общего превосходства. Разности менее 10⁻⁸ по абсолютной величине считаются численным нулём.", "",
              "| Серия | Потери: разность и интервал | p | Ремонт: разность и интервал | p |", "|---|---|---:|---|---:|"]
    for group, label in (("main", "S3, доля 30%, 20 зёрен"), ("s4", "S4, доля 30%, 10 зёрен"), ("small", "S3, доля 5%, 10 зёрен")):
        a, b = contrasts[group]["total_losses"], contrasts[group]["repair_traffic"]
        lines.append(f"| {label} | {interval(a)} | {a['wilcoxon_p']:.4g} | {interval(b)} | {b['wilcoxon_p']:.4g} |")
    base = summary_json["agent_minus_uniform"]["total_losses"]
    lines += ["", f"Дополнительное сравнение агента с равномерным размещением в основной серии: разность потерь {interval(base)}, Wilcoxon p={base['wilcoxon_p']:.4g}. Это исследовательское сравнение. Надёжное поведение Сибиллы способно привлекать фрагменты при выборе только по риску отказа."]
    lines += ["", "## Чувствительность полной защиты к публичным утечкам", "",
              "Во всех строках одни и те же 20 физических миров S3; меняется только публичный граф финансирования. Параметры защиты не меняются.", "",
              "| Вероятность утечки пары | Потери | Начальные отказы | Ремонт | Раскрыто объектов |", "|---:|---:|---:|---:|---:|"]
    for p, a in sensitivity.items():
        lines.append(f"| {p} | {number(a['total_losses']['mean'])} | {number(a['initial_failures']['mean'])} | {number(a['repair_traffic']['mean'])} | {number(a['plaintext_objects']['mean'])} |")
    lines += ["", "## H6: парные сигналы и кластеризация владения", "",
              "Снимок непосредственно перед атакой. Только зарегистрированные узлы коалиции. Кластеры в этой диагностике строятся по владению; политика использует общий граф с инфраструктурной составляющей.", "",
              "| Утечка пары | Полнота пар Сибиллы | Полнота кластеров Сибиллы | Точность пар* | Точность кластеров* |", "|---:|---:|---:|---:|---:|"]
    for p, a in h6.items():
        def mean_or_dash(v):
            return number(v["mean"]) if v is not None else "—"
        lines.append(f"| {p} | {mean_or_dash(a['pair_sybil_recall'])} | {mean_or_dash(a['cluster_sybil_recall'])} | {mean_or_dash(a['pair_precision'])} | {mean_or_dash(a['cluster_precision'])} |")
    lines += ["", "*Точность усредняется только по мирам, где есть предсказанные связи; число таких миров сохранено в summary.json. В генераторе нет ложных финансовых связей между разными владельцами. Высокая точность поэтому не доказывает устойчивости к загрязнённым сигналам."]
    diagnostic_path = args.data / "diagnostics/seed06_p06.json"
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
        if diagnostic.get("source_hash") == code_hash():
            snaps = {v["t"]: v for v in diagnostic["snapshots"]}
            first = diagnostic["objects"][0]
            lines += ["", "## Почему сильный сигнал не всегда предотвращает потери", "",
                      f"Исследовательская диагностика зерна 6 при утечке 0,6: потеряно {diagnostic['total_losses']} объектов, в момент атаки — {diagnostic['losses_at_attack']}. На сутках 30 общий граф защиты объединял {100*snaps[30]['combined_cluster_recall']:.1f}% пар Сибиллы, на сутках 60 — {100*snaps[60]['combined_cluster_recall']:.1f}%, на сутках 90 — 100%, но атака произошла на сутках 84.",
                      f"В первом объекте было {first['sybil_fragments']} фрагмента у истинного оператора Сибиллы, а максимальная группа по известному тогда графу содержала {first['max_group']}. Лимит три соблюдался для найденных групп, но один реальный владелец оказался разделён между ними. При последующем объединении групп политика не пересобирает прежние наборы автоматически. Это ограничение текущего механизма, а не основание объявить предсказание успешным по более позднему графу.",
                      "", "Диагностика выбрана после просмотра исходов и не использовалась для подбора параметров. Воспроизведение: `python scripts/diagnose_protection.py --seed 6 --p-leak 0.6`."]
    lines += ["", "## Границы выводов", "",
              "Раскрытие оценивается при явно заданной утечке ключа доступа на сутках 30. Без неё сбор фрагментов не означает чтение открытого текста. Нулевые раскрытия сами по себе не доказывают малый целевой риск."]
    for group, entries in aggregates.items():
        b = entries["protected"]["world_disclosure_bound"]
        lines.append(f"Для полной защиты в серии {group}: миров с раскрытием {b['worlds_with_disclosure']} из {b['worlds']}; односторонняя верхняя 95%-я граница вероятности мира с хотя бы одним раскрытием — {100*b['one_sided_95_upper']:.2f}%. Это не граница вероятности раскрытия одного объекта.")
    lines += ["", "H5 целиком не проверена: нет денежного тарифа в этом сравнении, вероятностного допуска по раскрытию и адаптивного противника. Ограничение текущего набора не ограничивает автоматически исторические копии коалиции. При нехватке допустимых узлов защита отказывает в операции; начальные отказы включены в суммарные потери.",
              "", "H6 здесь проверяет синтетический механизм известных сигналов. Парная полнота близка к частоте внедрённых финансовых рёбер по построению, а кластеризация может объединять лишь часть владельца. Для внешней валидации нужны реальные или независимо заданные загрязнённые сигналы.",
              "", "Все интервалы относятся к мирам-зёрнам; объекты внутри мира не объявляются независимыми наблюдениями. Абляции и чувствительность исследовательские. Результаты не переносятся автоматически на длинный горизонт, S7 и несогласованный мир.",
              "", "## Воспроизведение", "", "```powershell", ".venv\\Scripts\\python.exe scripts\\check_protection.py", ".venv\\Scripts\\python.exe scripts\\run_h5_h6.py --workers 3", ".venv\\Scripts\\python.exe scripts\\report_h5_h6.py", "```", "",
              "Машинные результаты: `runs/h5_h6/`, итог — `summary.json`. Каждый файл мира содержит конфигурацию, хэши физических таблиц, список режимов и отпечаток исходного кода. Кэш другого протокола или кода не принимается. Базовые сценарии и стандартная политика не менялись; выключенная новая защита проверена на совпадение с агентом.",
              "", "Графики: `figures/h5_protection_tradeoff.png`, `figures/h6_ownership_leakage.png`."]
    report = "\n".join(lines) + "\n"
    (args.data / "report.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/64_h5_h6_results.md").write_text(report, encoding="utf-8")

    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ps = np.array([0., .1, .3, .6])
    for ax, key, title in zip(axes, ("total_losses", "repair_traffic"), ("Потери, включая начальные отказы", "Ремонтный трафик за прогон")):
        values = [sensitivity[str(p)][key] for p in ps]
        means = np.array([v["mean"] for v in values])
        errors = np.array([v["hi"]-v["mean"] for v in values])
        ax.errorbar(ps, means, yerr=errors, marker="o", capsize=4, color="#087f8c", label="Полная защита, 95% ДИ")
        baseline = aggregates["main"]["agent"][key]["mean"]
        ax.axhline(baseline, color="#b76437", linestyle="--", label="Агент без новой защиты")
        ax.set(xlabel="Вероятность публичной утечки пары", title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    fig.suptitle("S3: 20 общих миров, доля Сибиллы 30%, 120 суток", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures / "h5_protection_tradeoff.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label, color in (("pair_sybil_recall", "Парные сигналы владения", "#087f8c"),
                              ("cluster_sybil_recall", "Кластеризация владения", "#b76437")):
        values = [h6[str(p)][key] for p in ps]
        ax.errorbar(ps, [v["mean"] for v in values], yerr=[v["hi"]-v["mean"] for v in values],
                    marker="o", capsize=4, label=label, color=color)
    ax.set(xlabel="Вероятность публичной утечки пары", ylabel="Полнота на парах Сибиллы",
           title="H6: связь пар и восстановление групп владельца, 20 миров", ylim=(-.03, 1.06))
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "h6_ownership_leakage.png", dpi=160)
    plt.close(fig)
    print(ROOT / "docs/64_h5_h6_results.md")
    print(json.dumps(contrasts["main"]["total_losses"], ensure_ascii=False))


if __name__ == "__main__":
    main()
