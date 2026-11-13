"""Внешние posthoc-метки; не импортировать в код политики."""
from math import isfinite


def label_snapshots(snapshots, jobs, segments, block_results, end):
    if not isfinite(end) or end <= 0:
        raise ValueError("Некорректный горизонт")
    rows = []
    for snapshot in snapshots:
        t = snapshot["time"]
        if not 0 <= t < end:
            raise ValueError("Снимок вне горизонта")
        for observed in snapshot["jobs"]:
            key = observed["id"]
            final = jobs[key]
            if final["id"] != key or final["submitted"] > t:
                raise ValueError("Не соответствует история заявки")
            finished = final["finished"]
            if finished is not None and finished < t:
                raise ValueError("В снимке завершённое ранее задание")
            completed = finished is not None and finished < end
            cutoff = finished if completed else end
            volume = 0.
            for segment in segments:
                if segment["job"] != key:
                    continue
                a, b = segment["start"], segment["end"]
                if not isfinite(a) or not isfinite(b) or b <= a:
                    raise ValueError("Некорректная фаза")
                volume += segment["volume"] * max(0., min(cutoff, b)-max(t, a))/(b-a)
            failures = sum(not result["ok"] for result in block_results
                if segments[result["segment"]]["job"] == key
                and t <= segments[result["segment"]]["end"]
                and (segments[result["segment"]]["end"] <= cutoff if completed
                     else segments[result["segment"]]["end"] < cutoff))
            # В срезе t- завершение ровно t ещё является будущим ответом.
            rows.append({"time": t, "job": key, "kind": observed["kind"],
                "censored": not completed, "observed_future_volume": volume,
                "full_future_volume": volume if completed else None,
                "time_to_completion_days": finished-t if completed else None,
                "observed_followup_days": cutoff-t,
                "future_failed_blocks_observed": failures,
                "final_status": final["status"] if completed else None})
    return rows
