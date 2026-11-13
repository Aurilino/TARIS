"""I4.9: ограниченная цель, не полный будущий резерв."""
from math import isfinite
from .conditional_work import conditional_work


def observed_window_volume(segments, job_ids, start, horizon, end):
    if (any(not isfinite(v) for v in (start, horizon, end)) or start < 0
            or horizon <= 0 or start+horizon > end):
        raise ValueError("Окно не полностью наблюдаемо")
    stop = start+horizon
    total = 0.
    for segment in segments:
        if segment["job"] not in job_ids:
            continue
        a, b, v = segment["start"], segment["end"], segment["volume"]
        if b <= a or not all(isfinite(x) for x in (a, b, v)) or v < 0:
            raise ValueError("Некорректная фаза")
        total += v*max(0., min(stop, b)-max(start, a))/(b-a)
    return total


def dataset_rows(case, horizon=7.):
    end = case["config"]["horizon_days"]
    rows = []
    for profile, variants in case["results"].items():
        for variant, record in variants.items():
            for snapshot in record["observations"]:
                t = snapshot["time"]
                if t+horizon > end:
                    continue
                ids = {job["id"] for job in snapshot["jobs"] if job["kind"] == "repair"}
                known = conditional_work(snapshot)["known_repair_volume"]
                cap = snapshot["queue_settings"]["rate_size_per_hour"]*24*horizon
                actual = observed_window_volume(record["segments"], ids, t, horizon, end)
                if actual > cap+1e-8:
                    raise ValueError("Обслуживание превышает пропускную способность")
                rows.append({"seed": case["seed"], "profile": profile, "variant": variant,
                    "time": t, "horizon_days": horizon, "known_work": known, "capacity": cap,
                    "current_repair_jobs": len(ids), "actual_window_volume": actual})
    return rows


def predict(row, coefficient=1.):
    if (not isfinite(coefficient) or coefficient < 0
            or not isfinite(row["known_work"]) or row["known_work"] < 0
            or not isfinite(row["capacity"]) or row["capacity"] <= 0):
        raise ValueError("Некорректный прогноз")
    return min(coefficient*row["known_work"], row["capacity"])


def fit_profile(rows):
    if not rows:
        raise ValueError("Нет обучающих строк")
    if not any(row["known_work"] > 0 for row in rows):
        return {"coefficient": 1., "status": "untrained", "edge_of_grid": False}
    seeds = sorted({r["seed"] for r in rows})
    groups = [[r for r in rows if r["seed"] == seed] for seed in seeds]
    best = None
    for tick in range(401):
        coefficient = tick/100
        score = sum(sum((predict(r, coefficient)-r["actual_window_volume"])**2 for r in group)/len(group)
                    for group in groups)/len(groups)
        if best is None or score < best[0]:
            best = score, coefficient
    return {"coefficient": best[1], "status": "fitted", "training_world_mean_mse": best[0],
            "edge_of_grid": best[1] in (0., 4.), "training_seeds": seeds}
