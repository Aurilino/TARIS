"""I4.5a: пересчёт существующих журналов, без новой генерации миров."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import Counter
import hashlib
import json
import math
from subprojects.taris_variants.object_reserve import object_reserve


def main():
    folder = Path(__file__).resolve().parent
    groups = (("anchor_i44a", folder/"series_i44c"/"anchor_i44a", (24, 25, 26)),
              ("paired_i44b", folder/"series_i44b", (24, 25, 26)),
              ("paired_i44c", folder/"series_i44c", (24, 25, 26)),
              ("validation_i44c", folder/"series_i44c", (27, 28, 29)))
    cases, hashes, records = {}, {}, []
    for cohort, directory, seeds in groups:
        for seed in seeds:
            path = directory/f"s{seed}.json"
            hashes[str(path.relative_to(folder))] = hashlib.sha256(path.read_bytes()).hexdigest()
            saved = json.loads(path.read_text(encoding="utf-8"))
            case = saved["case"]
            assert case["world_unchanged"] and case["common_random_numbers"]
            fingerprint = {key: case[key] for key in ("world_hashes", "input_hashes", "config", "settings")}
            if seed in cases:
                assert cases[seed] == fingerprint
            else:
                cases[seed] = fingerprint
            cfg = case["config"]
            for profile, variants in case["results"].items():
                for variant, record in variants.items():
                    result = object_reserve(record, cfg["k"], case["settings"]["n"], cfg["horizon_days"], cfg["object_size"])
                    m = record["metrics"]
                    assert result["objects"] == m["objects"]
                    for new, old in (("object_days", "object_days"), ("physical_copy_days", "physical_fragment_days"),
                        ("disk_deficit_object_days", "disk_deficit_object_days"), ("buffer_only_object_days", "buffer_only_object_days")):
                        assert math.isclose(result[new], m[old], rel_tol=1e-9, abs_tol=1e-8), (cohort, seed, variant, new)
                    records.append({"cohort": cohort, "seed": seed, "profile": profile, "variant": variant, **result})
    summary = []
    fields = ("objects", "object_days", "distinct_fragment_days", "physical_copy_days", "physical_size_days",
        "duplicate_copy_days", "below_target_object_days", "target_shortfall_fragment_days",
        "bare_code_object_days", "disk_deficit_object_days", "buffer_only_object_days")
    for cohort, _, _ in groups:
        for profile in ("ample", "constrained"):
            for variant in ("classic", "graph", "hybrid"):
                subset = [r for r in records if (r["cohort"],r["profile"],r["variant"]) == (cohort,profile,variant)]
                histogram = Counter()
                for r in subset:
                    histogram.update(r["rank_days"])
                summary.append({"cohort": cohort, "profile": profile, "variant": variant,
                    **{key: sum(r[key] for r in subset) for key in fields}, "rank_days": dict(sorted(histogram.items()))})
    for name, digest in hashes.items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    output = folder/"object_evaluation_i45a"
    output.mkdir(exist_ok=True)
    report = {"schema": "taris-object-reserve-evaluation/1", "date": "2026-11-11",
        "input_hashes": hashes, "paired_inputs_verified": True, "existing_metrics_verified": True,
        "source_hashes": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest()
                          for name in ("object_reserve.py", "report_object_reserve.py")},
        "summary": summary, "records": records}
    (output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# I4.5a. Физический запас разных номеров объекта", "", "Дата: 11 ноября 2026 года.", "",
        "В каждой строке 135 объектов и 3105 объекто-суток. Пересчитаны существующие журналы; новых миров нет.", "",
        "| Серия | Канал | Агент | Ниже n=4, объекто-сутки | Недовосполнение, номеро-сутки | Ровно k=2, объекто-сутки | Дубликаты, копие-сутки |",
        "|---|---|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['cohort']} | {r['profile']} | {r['variant']} | {r['below_target_object_days']:.3f} | "
            f"{r['target_shortfall_fragment_days']:.3f} | {r['bare_code_object_days']:.3f} | {r['duplicate_copy_days']:.3f} |")
    lines += ["", "Истинные физические копии учитываются независимо от активной карты и доступности узла. "
        "Дубликат не увеличивает число разных номеров, но может защищать один номер от отказа носителя. "
        "Это ретроспективные показатели запаса, не прогноз вероятности потери и не данные для агента.", ""]
    (output/"RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Проверено исполнений: {len(records)}, объект-размещений: {sum(r['objects'] for r in records)}. "
          "Экспозиция, хранение, дефицит и буфер сверены с прежними аудитами.")


if __name__ == "__main__":
    main()
