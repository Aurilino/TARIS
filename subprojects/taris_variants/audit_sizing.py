"""Внешняя проверка I3 по истинным параметрам той же модели окна.

Ничего не передаёт в агента и не подбирает параметры по будущей истине.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
from collections import defaultdict
import hashlib
import json
from statistics import mean

from taris_sim.config import Config
from taris_sim.world import World
from taris_sim.reserve import TrueRates, placement_loss_probability
from subprojects.taris_variants.run_compare import source_hashes, table_hash


def summarize_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["profile"], row["variant"]].append(row)
    summary = []
    for (profile, variant), group in sorted(grouped.items()):
        scored = [r for r in group if r["scored"]]
        passed = [r for r in group if r["estimated_passed"]]
        summary.append({"profile": profile, "variant": variant, "admitted": len(group),
            "estimated_passed": len(passed), "true_parameter_cap_exceeded": sum(r["true_exceeded"] for r in group),
            "true_parameter_upper_exceeded": sum(r.get("upper_exceeded", r["true_exceeded"]) for r in group),
            "false_safe_among_estimated_passed": sum(r["true_exceeded"] for r in passed),
            "true_feasible_but_estimated_not_passed": sum(not r.get("upper_exceeded", r["true_exceeded"])
                and not r["estimated_passed"] for r in group),
            "mean_true_window_loss": mean(r["true_window_loss_upper"] for r in group),
            "mean_absolute_error_scored": mean(abs(r["estimated_window_loss"] - r["true_window_loss_upper"])
                                                for r in scored) if scored else None})
    return summary


def render(report):
    lines = ["# I3. Внешняя проверка размеров по истинным параметрам", "", "Дата: 10 ноября 2026 года.", "",
        "Контроллеры не видят эти оценки. Проверяются нижняя и верхняя оценки потери за то же окно в той же модели, но с истинными интенсивностями вместо оценённых. Превышение в таблице означает, что даже нижняя оценка выше лимита. Это не точная вероятность физического уничтожения на всём жизненном пути.", "",
        "| Профиль | Вариант | Принято | Оценочно прошли | Истинный cap превышен | Ложно безопасные среди прошедших |",
        "|---|---|---:|---:|---:|---:|"]
    for row in report["summary"]:
        lines.append(f"| {row['profile']} | {row['variant']} | {row['admitted']} | {row['estimated_passed']} | "
                     f"{row['true_parameter_cap_exceeded']} | {row['false_safe_among_estimated_passed']} |")
    lines += ["", "Колонка превышения включает холодный старт. Последняя колонка относится только к размещениям со статусом успешной проверки; эти числа нельзя складывать как независимые события.",
        "", "TrueRates также включает длительные простои как потери. Расчёт использует постоянные интенсивности на окне и независимость шоков кластеров. Нижняя оценка исключает отброшенные сценарии более двух шоков; верхняя добавляет всю их массу. Проверяется ошибка оценивания в этих допущениях, а не их абсолютная истинность.",
        "", "Миры восстановлены по сохранённым Config и сверены по хешам. Параметры и решения агентов после аудита не менялись. Отдельная проверка истинной доступности во времени здесь не выполняется; аудит относится к требованию потери за окно.",
        "", f"Изоляция основной среды: {report['main_source_unchanged']}. Совпадение миров: {report['worlds_verified']}.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Внешний аудит размеров I3")
    parser.add_argument("--directory", type=Path, default=Path(__file__).with_name("series_i3"))
    args = parser.parse_args()
    before = source_hashes()
    worlds, rows = {}, []
    defaults = Config()
    files = sorted(args.directory.glob("s*_*.json"))
    verified = True
    for path in files:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["status"] != "completed":
            continue
        if saved["provenance"]["main"] != before:
            raise ValueError("Основные исходники изменились относительно результатов I3")
        case = saved["case"]
        config = case["config"]
        cfg = Config(**{key: tuple(value) if isinstance(getattr(defaults, key), tuple) else value
                        for key, value in config.items()})
        key = cfg.hash()
        if key not in worlds:
            world = World(cfg)
            hashes = {"nodes": table_hash(world.nodes), "outages": table_hash(world.outages), "silent": table_hash(world.silent)}
            if hashes != case["world_hashes"]:
                raise AssertionError("Восстановленный мир не совпал с исходным")
            worlds[key] = TrueRates(world), hashes
        rates, hashes = worlds[key]
        if hashes != case["world_hashes"]:
            raise AssertionError("Случаи одной конфигурации содержат разные миры")
        for variant, result in case["variants"].items():
            for decision in result["placement_decisions"]:
                if not decision["nodes"]:
                    continue
                true_risk, dropped = placement_loss_probability(rates, decision["nodes"], decision["time"],
                    decision["window_days"], cfg.k)
                lower = max(0.0, true_risk - dropped)
                estimate = decision["selected"]["estimated_window_loss"]
                rows.append({"seed": case["seed"], "profile": case["profile"], "variant": variant,
                    "obj": decision["obj"], "n": decision["n"], "scored": decision["scored"],
                    "estimated_passed": decision["requirements_passed"], "estimated_window_loss": estimate,
                    "true_window_loss_upper": true_risk, "true_window_loss_lower": lower,
                    "true_exceeded": lower > decision["window_loss_limit"],
                    "upper_exceeded": true_risk > decision["window_loss_limit"],
                    "limit": decision["window_loss_limit"], "dropped_mass": dropped})
    if not rows:
        raise ValueError("Нет завершённых размещений для внешней проверки")
    report = {"schema": "taris-sizing-audit/1", "main_source_unchanged": before == source_hashes(),
              "worlds_verified": verified, "worlds": len(worlds), "placements": len(rows),
              "audit_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "summary": summarize_rows(rows), "rows": rows}
    (args.directory / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.directory / "AUDIT.md").write_text(render(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
