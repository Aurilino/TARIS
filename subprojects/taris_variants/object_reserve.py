"""Внешняя оценка запаса информации; не наблюдения или прогноз агента."""
from collections import Counter, defaultdict
from math import isfinite


def object_reserve(record, k, target_n, end, object_size=1.):
    if not (isinstance(k, int) and isinstance(target_n, int) and 1 <= k <= target_n
            and isfinite(end) and end > 0 and isfinite(object_size) and object_size > 0):
        raise ValueError("Некорректные параметры оценки запаса")
    created = {int(row[0]): float(row[1]) for row in record["initial_writes"]}
    if len(created) != len(record["initial_writes"]) or any(not isfinite(t) or t < 0 for t in created.values()):
        raise ValueError("Некорректная таблица создания объектов")
    fragments, buffers = defaultdict(list), defaultdict(list)
    for f in record["physical_fragments"]:
        if f["obj"] not in created:
            raise ValueError("Копия неизвестного объекта")
        a = float(f["start"])
        b = float(f["physical_end"]) if f["physical_end"] is not None else end
        if not isfinite(a) or not isfinite(b) or a < created[f["obj"]] or b < a and a < end:
            raise ValueError("Некорректный интервал копии")
        if a < min(b, end):
            fragments[f["obj"]].append((a, min(b, end), int(f["idx"])))
    for row in record["buffers"]:
        if row["obj"] not in created:
            raise ValueError("Буфер неизвестного объекта")
        a, b = float(row["start"]), row["end"]
        b = float(b) if b is not None else end
        if not isfinite(a) or not isfinite(b) or a < created[row["obj"]] or b < a and a < end:
            raise ValueError("Некорректный интервал буфера")
        if a < min(b, end):
            buffers[row["obj"]].append((a, min(b, end)))
    rows = []
    for obj, start in sorted(created.items()):
        if start >= end:
            continue
        changes = defaultdict(list)
        changes[start], changes[end]
        for a, b, idx in fragments[obj]:
            changes[a].append((idx, 1))
            changes[b].append((idx, -1))
        for a, b in buffers[obj]:
            changes[a].append((None, 1))
            changes[b].append((None, -1))
        counts, histogram, full_buffers = Counter(), Counter(), 0
        previous, copy_days, buffer_only = start, 0., 0.
        for t in sorted(changes):
            dt = t-previous
            rank = sum(value > 0 for value in counts.values())
            if dt > 0:
                histogram[rank] += dt
                copy_days += sum(counts.values())*dt
                buffer_only += dt * (rank < k and full_buffers > 0)
            # Полуоткрытые интервалы: замена точно в один момент не создаёт
            # искусственного перекрытия из-за порядка строк входного файла.
            deltas = Counter()
            for idx, delta in changes[t]:
                if idx is None:
                    full_buffers += delta
                else:
                    deltas[idx] += delta
            counts.update(deltas)
            if not 0 <= full_buffers <= 1:
                raise ValueError("Перекрывающиеся полные буферы одного объекта")
            if any(value < 0 for value in counts.values()):
                raise ValueError("Отрицательное число копий")
            previous = t
        distinct_days = sum(n*days for n, days in histogram.items())
        rows.append({"obj": obj, "object_days": end-start, "rank_days": dict(sorted(histogram.items())),
            "minimum_distinct": min(histogram), "distinct_fragment_days": distinct_days,
            "physical_copy_days": copy_days, "physical_size_days": copy_days*object_size/k,
            "duplicate_copy_days": max(0., copy_days-distinct_days),
            "below_target_object_days": sum(days for n, days in histogram.items() if n < target_n),
            "target_shortfall_fragment_days": sum(max(0, target_n-n)*days for n, days in histogram.items()),
            "bare_code_object_days": histogram.get(k, 0.),
            "disk_deficit_object_days": sum(days for n, days in histogram.items() if n < k),
            "buffer_only_object_days": buffer_only})
    fields = ("object_days", "distinct_fragment_days", "physical_copy_days", "physical_size_days",
        "duplicate_copy_days", "below_target_object_days", "target_shortfall_fragment_days",
        "bare_code_object_days", "disk_deficit_object_days", "buffer_only_object_days")
    histogram = Counter()
    for row in rows:
        histogram.update(row["rank_days"])
    return {"objects": len(rows), "target_n": target_n, "k": k,
            **{key: sum(r[key] for r in rows) for key in fields},
            "rank_days": dict(sorted(histogram.items())), "rows": rows}
