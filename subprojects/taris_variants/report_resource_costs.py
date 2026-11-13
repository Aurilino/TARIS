"""I4.5b: цены наблюдённых ресурсов рядом с запасом и незавершённой работой."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dataclasses import asdict
import hashlib
import json
import math
import csv
from subprojects.taris_variants.resource_costs import resource_usage, observed_charges, ResourcePrices


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8), (a, b)


def main():
    folder = Path(__file__).resolve().parent
    reserve_path = folder/"object_evaluation_i45a"/"results.json"
    reserve_bytes = reserve_path.read_bytes()
    reserves = json.loads(reserve_bytes)
    for name, digest in reserves["source_hashes"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    keyed_reserves = {(r["cohort"], r["seed"], r["profile"], r["variant"]): r for r in reserves["records"]}
    groups = (("anchor_i44a", folder/"series_i44c"/"anchor_i44a", (24, 25, 26)),
              ("paired_i44b", folder/"series_i44b", (24, 25, 26)),
              ("paired_i44c", folder/"series_i44c", (24, 25, 26)),
              ("validation_i44c", folder/"series_i44c", (27, 28, 29)))
    resources, charges, verified_hashes, coding_units = [], [], {}, set()
    for cohort, directory, seeds in groups:
        for seed in seeds:
            path = directory/f"s{seed}.json"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            name = str(path.relative_to(folder))
            assert reserves["input_hashes"][name] == digest
            verified_hashes[name] = digest
            case = json.loads(path.read_text(encoding="utf-8"))["case"]
            cfg = case["config"]
            end, k, size = cfg["horizon_days"], cfg["k"], cfg["object_size"]
            coding_units.add((k, size))
            for profile, variants in case["results"].items():
                for variant, record in variants.items():
                    key = cohort, seed, profile, variant
                    stock = keyed_reserves[key]
                    m = record["metrics"]
                    usage = resource_usage(record, end, k, size)
                    close(usage["physical_copy_days"], stock["physical_copy_days"])
                    close(usage["physical_size_days"], stock["physical_size_days"])
                    close(usage["decoded_prefix_size_days"], m["decoded_prefix_size_days"])
                    close(usage["receiver_prefix_size_days"], m["receiver_prefix_size_days"])
                    close(usage["full_buffer_size_days"], m["buffer_object_days"]*size)
                    close(usage["peak_central_size"], m["peak_central_size"])
                    for phase in set(usage["served_volume_by_phase"]) | set(m["serviced_volume"]):
                        close(usage["served_volume_by_phase"].get(phase, 0.), m["serviced_volume"].get(phase, 0.))
                    assert sum(j["finished"] is None for j in record["jobs"]) == m["unfinished_jobs"]
                    controls = {name: stock[name] for name in ("objects", "object_days", "below_target_object_days",
                        "target_shortfall_fragment_days", "bare_code_object_days", "disk_deficit_object_days", "buffer_only_object_days")}
                    controls.update({name: m[name] for name in ("unfinished_jobs", "open_buffers", "unavailable_object_days", "irrecoverable_objects")})
                    base = {"cohort": cohort, "seed": seed, "profile": profile, "variant": variant}
                    resources.append({**base, **usage, **controls})
                    for central in (1., 10., 100.):
                        for receiver in (1., 10.):
                            prices = ResourcePrices(central_ratio=central, receiver_ratio=receiver)
                            billed = observed_charges(usage, prices, k, size)
                            charges.append({**base, "prices": asdict(prices), **billed, **controls,
                                "scheduled_unserved_volume": usage["scheduled_unserved_volume"],
                                "future_liability_cost": None})
    assert len(resources) == 72 and len(charges) == 432 and coding_units == {(2, 1.0)}
    summary = []
    fields = ("disk", "full_buffer", "decoded_prefix", "receiver_prefix", "channel", "observed_resource_cost",
        "objects", "object_days", "below_target_object_days", "target_shortfall_fragment_days", "bare_code_object_days",
        "disk_deficit_object_days", "buffer_only_object_days", "unfinished_jobs", "open_buffers", "unavailable_object_days",
        "irrecoverable_objects", "scheduled_unserved_volume")
    for cohort, _, _ in groups:
        for profile in ("ample", "constrained"):
            for variant in ("classic", "graph", "hybrid"):
                for central in (1., 10., 100.):
                    for receiver in (1., 10.):
                        subset = [r for r in charges if (r["cohort"],r["profile"],r["variant"],
                            r["prices"]["central_ratio"],r["prices"]["receiver_ratio"]) == (cohort,profile,variant,central,receiver)]
                        assert len(subset) == 3
                        summary.append({"cohort": cohort, "profile": profile, "variant": variant,
                            "central_ratio": central, "receiver_ratio": receiver,
                            **{key: sum(r[key] for r in subset) for key in fields},
                            "expected_loss_cost": None, "future_liability_cost": None, "full_actuarial_reserve": None})
    for name, digest in verified_hashes.items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256(reserve_path.read_bytes()).hexdigest() == hashlib.sha256(reserve_bytes).hexdigest()
    output = folder/"resource_cost_evaluation_i45b"
    output.mkdir(exist_ok=True)
    report = {"schema": "taris-resource-cost-evaluation/1", "date": "2026-11-11", "discount_rate": 0.,
        "scope": "observed_resources_only", "unknown_expected_loss_cost": True, "unknown_future_liability": True,
        "input_hashes": verified_hashes, "reserve_report_hash": hashlib.sha256(reserve_bytes).hexdigest(),
        "source_hashes": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                          for name in ("resource_costs.py", "report_resource_costs.py")},
        "legacy_metrics_verified": True, "coding_units": {"k": 2, "object_size": 1.},
        "price_basis": asdict(ResourcePrices()), "summary": summary, "resources": resources, "charges": charges}
    (output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    csv_fields = list(summary[0])
    with (output/"summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(summary)
    lines = ["# I4.5b. Условная стоимость наблюдённых ресурсов", "", "Дата: 11 ноября 2026 года.", "",
        "В таблице — медленный канал, цена временной области получателя равна цене объёма диска. "
        "Каждая строка объединяет три мира, 135 объектов. Цены модельные, без дисконтирования. "
        "Ожидаемый ущерб и будущие обязательства не оценены и не приравнены к нулю.", "",
        "| Серия | Агент | RAM ×1 | RAM ×10 | RAM ×100 | Ниже цели n=4, объекто-сутки | Хвост |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for cohort, _, _ in groups:
        for variant in ("classic", "graph", "hybrid"):
            rows = [r for r in summary if r["cohort"] == cohort and r["variant"] == variant
                    and r["profile"] == "constrained" and r["receiver_ratio"] == 1.]
            rows.sort(key=lambda r: r["central_ratio"])
            lines.append(f"| {cohort} | {variant} | {rows[0]['observed_resource_cost']:.4f} | "
                f"{rows[1]['observed_resource_cost']:.4f} | {rows[2]['observed_resource_cost']:.4f} | "
                f"{rows[0]['below_target_object_days']:.3f} | {rows[0]['unfinished_jobs']} |")
    lines += ["", "Полная сетка (цена RAM ×1/10/100, временной области ×1/10) — в summary.csv и results.json. "
        "Показатели сохранности, незавершённой работы и недоступности сохранены рядом с ценой. "
        "Это не стоимость окончательного завершения работ и не критерий универсального выбора агента.", ""]
    (output/"RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Исполнений: {len(resources)}; ценовых расчётов: {len(charges)}; сводных строк: {len(summary)}. "
          "Ресурсы сверены с прежними метриками, исходные журналы не изменились.")


if __name__ == "__main__":
    main()
