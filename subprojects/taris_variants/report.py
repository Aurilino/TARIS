"""Краткий отчёт из сохранённого пилота трёх вариантов."""
import argparse
import json
from pathlib import Path


def render(data):
    lines = ["# Пилот трёх вариантов TARIS", "", "Дата: 10 ноября 2026 года.", "",
        "Интеграционная проверка при фиксированном начальном числе чанков. Потери — по действующей семантике Ledger, включая административное признание после длительного простоя.", "",
        "| Мир | Вариант | Потери | Ремонты | Трафик ремонта | Фрагменто-сутки | Средняя недоступность до потери |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for case in data["cases"]:
        for value in case["variants"].values():
            risk = value["object_unavailability_mean_before_loss"]
            share = f"{risk:.4%}" if risk is not None else "нет оценки"
            lines.append(f"| {case['scenario']}, s{case['seed']} | {value['name']} | {value['lost_objects']} | "
                f"{value['repair_operations']} | {value['repair_traffic']:.2f} | {value['fragment_days']:.0f} | {share} |")
    lines += ["", "## Проверка изоляции", "",
        f"Исходники основной среды неизменны: **{data['main_sources_unchanged']}**.",
        f"Таблицы мира неизменны во всех случаях: **{all(c['world_unchanged'] for c in data['cases'])}**.",
        f"Случайность мира, наблюдений и потока данных одинакова: **{all(c['common_random_numbers'] for c in data['cases'])}**.",
        "", "## Численные предупреждения", ""]
    for case in data["cases"]:
        for key, value in case["variants"].items():
            if value["runtime_warnings"]:
                lines.append(f"- {case['scenario']} s{case['seed']} {key}: {value['runtime_warnings']}. Конечность проверена: {value['numerical_finite']}.")
    lines += ["", "## Границы вывода", "",
        "Это малый пилот исполнения, не статистическое доказательство превосходства. Средняя недоступность рассчитана как среднее по объектам до их потери; её нельзя трактовать отдельно от числа потерь или как долю времени сохранности после потери.", "",
        "S0 и S2 на коротком горизонте могут иметь одинаковые события, если шоки не реализовались. Такие случаи не являются независимыми свидетельствами поведения в кризисе.", "",
        "Меньше ремонтов не обязательно означает лучшую защиту. Сравнивать нужно одновременно потери, недоступность, хранение и трафик. В режиме thiele TARIS-G отличается также немедленным восстановлением; для отделения этого эффекта предусмотрен режим fixed.", "",
        "Начальное число чанков одинаково. Реализованный объём хранения во времени может различаться из-за ремонта и потерь. Строгое сравнение при одинаковых фрагменто-сутках — отдельный эксперимент.", "",
        "В текущем профиле прогноз кризисов и новый исполнитель автономного прототипа не включены. Полная минимизация числа чанков и независимая калибровка графовой модели — следующие этапы.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("comparison_results.json"))
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("RESULTS.md"))
    args = parser.parse_args()
    args.output.write_text(render(json.loads(args.input.read_text(encoding="utf-8"))), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
