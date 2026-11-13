"""Парное сравнение I4.2/I4.3 и отдельная сводка новых миров."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections import Counter
import hashlib
import json
from subprojects.taris_variants.run_resumable import provenance


def aggregate(records):
    metrics = [r["metrics"] for r in records]
    keys = ("objects", "irrecoverable_objects", "initial_incomplete_objects", "object_days", "unavailable_object_days",
            "disk_deficit_object_days", "buffer_only_object_days", "buffer_object_days", "unfinished_jobs",
            "queued_jobs", "open_buffers", "started_wait_days_sum", "pending_wait_days_sum", "started_jobs")
    result = {key: sum(m[key] for m in metrics) for key in keys}
    result["written_chunks"] = sum(row[-1] == "written" for r in records for row in r["repair_outcomes"])
    result["full_buffers_created"] = sum(len(r["buffers"]) for r in records)
    result["serviced_volume"] = dict(sum((Counter(m["serviced_volume"]) for m in metrics), Counter()))
    result["served_total"] = sum(result["serviced_volume"].values())
    result["queue_rejections"] = dict(sum((Counter(m["queue_rejections"]) for m in metrics), Counter()))
    result["job_statuses"] = dict(sum((Counter(m["job_statuses"]) for m in metrics), Counter()))
    result["observed_wait_days"] = result["started_wait_days_sum"]+result["pending_wait_days_sum"]
    result["peak_central_size"] = max((r["metrics"].get("peak_central_size",
        max((b["size"] for b in r["buffers"]), default=0.)) for r in records), default=0.)
    for key in ("decoded_prefix_size_days", "receiver_prefix_size_days", "block_attempts", "failed_blocks", "successful_close_checks"):
        result[key] = sum(m.get(key, 0) for m in metrics)
    return result


def main():
    folder, before = Path(__file__).resolve().parent, provenance()
    old, new = {}, {}
    for seed in (15, 16, 17):
        saved = json.loads((folder/"series_i42"/f"s{seed}.json").read_text(encoding="utf-8"))
        assert saved["provenance"]["main"] == before["main"]
        for name, digest in saved["provenance"]["integration"].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
        old[seed] = saved["case"]
    for seed in range(15, 21):
        saved = json.loads((folder/"series_i43"/f"s{seed}.json").read_text(encoding="utf-8"))
        assert saved["provenance"] == before
        new[seed] = saved["case"]
        if seed in old:
            for key in ("world_hashes", "input_hashes", "config", "settings"):
                assert old[seed][key] == new[seed][key], (seed, key)
    rows = []
    for cohort, cases in (("paired_i42", list(old.values())), ("paired_i43", [new[s] for s in (15, 16, 17)]),
                          ("validation_i43", [new[s] for s in (18, 19, 20)])):
        for profile in ("ample", "constrained"):
            for variant in ("classic", "graph", "hybrid"):
                rows.append({"cohort": cohort, "profile": profile, "variant": variant,
                             **aggregate([c["results"][profile][variant] for c in cases])})
    assert provenance() == before
    report = {"schema": "taris-resume-comparison/1", "paired_inputs_verified": True,
        "source_hashes_verified": True, "paired_worlds": 3, "new_worlds": 3,
        "report_source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "rows": rows}
    directory = folder/"series_i43"
    (directory/"comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# I4.3. Парное сравнение и новая проверка", "",
        "paired_i42/paired_i43 — одни и те же три мира 15–17. validation_i43 — новые миры 18–20, без нового базиса I4.2. В каждой строке 135 объектов.", "",
        "| Серия | Канал | Стратегия | Новых чанков | Полных буферов | Потери | Бюджет | Хвост | Наблюдённое ожидание, сут. |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['cohort']} | {row['profile']} | {row['variant']} | {row['written_chunks']} | "
            f"{row['full_buffers_created']} | {row['irrecoverable_objects']} | {row['served_total']:.3f} | "
            f"{row['unfinished_jobs']} | {row['observed_wait_days']:.3f} |")
    lines += ["", "Ожидание включает возраст ещё не начатых заданий. Хвост суммируется по трём прогонам; бюджет включает "
        "неуспешные блоки, проверки возврата и закрытия буфера. Меньший расход не означает завершённость ремонта.",
        "", "Сохранность редких событий и преимущество одной политики этими тремя проверочными мирами не доказаны.", ""]
    (directory/"COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")
    print("Парность входов и исходники подтверждены; comparison.json и COMPARISON.md сохранены.")


if __name__ == "__main__":
    main()
