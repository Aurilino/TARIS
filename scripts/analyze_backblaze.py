"""Summarize Backblaze daily drive exposure, failures, and SMART-9 age bands."""
from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

DAY_PER_YEAR = 365.25
AGE_BANDS = ("0-1 powered years", "1-3 powered years", "3-5 powered years",
             "5-7 powered years", "7+ powered years")
AGE_LABELS = dict(zip(AGE_BANDS, range(len(AGE_BANDS))))
CSV_COLUMNS = {"date", "serial_number", "model", "failure", "smart_9_raw"}


def analyze(archive: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
    model_stats = defaultdict(lambda: {"drive_days": 0, "failures": 0, "rows": 0,
                                       "serials": set(), "failure_serials": set()})
    age_stats = defaultdict(lambda: {"drive_days": 0, "failures": 0})
    model_age_stats = defaultdict(lambda: {"drive_days": 0, "failures": 0})
    active_by_model = defaultdict(int)
    serial_model: dict[str, str] = {}
    serial_model_changes = 0
    observed_dates = set()
    files = rows_total = failures_total = missing_age_rows = bad_rows = 0

    with zipfile.ZipFile(archive) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        if not names:
            raise ValueError("ZIP archive contains no CSV files")
        final_date = Path(names[-1]).stem
        for file_index, name in enumerate(names, 1):
            with zf.open(name) as raw:
                reader = pd.read_csv(raw, usecols=lambda c: c.strip().lower() in CSV_COLUMNS,
                                     dtype={"date": "string", "serial_number": "string", "model": "string"},
                                     chunksize=250_000, low_memory=False)
                file_date = Path(name).stem
                files += 1
                for chunk in reader:
                    # Normalize selected headers and keep only the requested measures.
                    chunk.columns = [c.strip().lower() for c in chunk.columns]
                    required = {"date", "serial_number", "model", "failure"}
                    if not required.issubset(chunk.columns):
                        raise ValueError(f"{name}: missing required columns {sorted(required - set(chunk.columns))}")
                    chunk["failure"] = pd.to_numeric(chunk["failure"], errors="coerce")
                    valid = chunk["serial_number"].notna() & chunk["failure"].isin([0, 1])
                    bad_rows += int((~valid).sum())
                    if not valid.all():
                        chunk = chunk.loc[valid].copy()
                    if chunk.empty:
                        continue
                    chunk["model"] = chunk["model"].fillna("(unknown)").astype(str)
                    chunk["serial_number"] = chunk["serial_number"].astype(str)
                    chunk["failure"] = chunk["failure"].astype("int8")
                    observed_dates.update(chunk["date"].dropna().astype(str).unique())
                    rows_total += len(chunk)
                    failures_total += int(chunk["failure"].sum())

                    for model, group in chunk.groupby("model", sort=False):
                        s = model_stats[model]
                        s["rows"] += len(group)
                        s["drive_days"] += len(group)
                        s["failures"] += int(group["failure"].sum())
                        s["serials"].update(group["serial_number"].unique().tolist())
                        failed = group.loc[group["failure"].eq(1), "serial_number"].unique()
                        s["failure_serials"].update(failed.tolist())

                    for serial, model in chunk[["serial_number", "model"]].drop_duplicates().itertuples(index=False, name=None):
                        old = serial_model.setdefault(serial, model)
                        serial_model_changes += old != model

                    if file_date == final_date:
                        for model, count in chunk.groupby("model")["serial_number"].nunique().items():
                            active_by_model[model] += int(count)

                    if "smart_9_raw" in chunk.columns:
                        hours = pd.to_numeric(chunk["smart_9_raw"], errors="coerce")
                        age_years = hours / (24.0 * DAY_PER_YEAR)
                        band = pd.cut(age_years, [0, 1, 3, 5, 7, np.inf], right=False,
                                      labels=AGE_BANDS, include_lowest=True)
                        missing_age_rows += int(band.isna().sum())
                        age_frame = pd.DataFrame({"model": chunk["model"], "band": band,
                                                  "failure": chunk["failure"]}).dropna(subset=["band"])
                        grouped = age_frame.groupby(["model", "band"], observed=True)["failure"].agg(["count", "sum"])
                        for (model, label), value in grouped.iterrows():
                            band_name = str(label)
                            s = age_stats[band_name]
                            s["drive_days"] += int(value["count"])
                            s["failures"] += int(value["sum"])
                            ma = model_age_stats[(model, band_name)]
                            ma["drive_days"] += int(value["count"])
                            ma["failures"] += int(value["sum"])
                    else:
                        missing_age_rows += len(chunk)
            print(f"{file_index}/{len(names)} {Path(name).name}", flush=True)

    by_model = []
    for model, s in sorted(model_stats.items()):
        by_model.append({"model": model, "drive_days": s["drive_days"],
                         "drives_on_quarter_end": active_by_model[model],
                         "unique_drives_in_quarter": len(s["serials"]), "failures": s["failures"],
                         "failed_drives": len(s["failure_serials"]),
                         "afr": s["failures"] * DAY_PER_YEAR / s["drive_days"] if s["drive_days"] else None})
    threshold_models = [m for m in by_model if m["unique_drives_in_quarter"] > 100 and m["drive_days"] > 10_000]
    threshold_days = sum(m["drive_days"] for m in threshold_models)
    threshold_failures = sum(m["failures"] for m in threshold_models)
    summary = {
        "archive": archive.name, "csv_files": files, "dates": len(observed_dates),
        "first_date": min(observed_dates) if observed_dates else None,
        "last_date": max(observed_dates) if observed_dates else None,
        "rows": rows_total, "drive_days": rows_total, "failures": failures_total,
        "afr": failures_total * DAY_PER_YEAR / rows_total if rows_total else None,
        "unique_drives_in_period_all_models": sum(len(s["serials"]) for s in model_stats.values()),
        "drives_on_quarter_end_all_models": sum(active_by_model.values()),
        "serials_with_model_change": serial_model_changes,
        "models": len(model_stats), "models_meeting_basic_size_thresholds": len(threshold_models),
        "threshold_drive_days": threshold_days, "threshold_failures": threshold_failures,
        "threshold_afr": threshold_failures * DAY_PER_YEAR / threshold_days if threshold_days else None,
        "rows_missing_smart9": missing_age_rows, "bad_rows": bad_rows,
    }
    by_age = []
    for label in AGE_BANDS:
        s = age_stats[label]
        by_age.append({"powered_age_band": label, "drive_days": s["drive_days"],
                       "failures": s["failures"],
                       "afr": s["failures"] * DAY_PER_YEAR / s["drive_days"] if s["drive_days"] else None})
    model_age = []
    for (model, label), s in sorted(model_age_stats.items()):
        model_age.append({"model": model, "powered_age_band": label, "drive_days": s["drive_days"],
                          "failures": s["failures"],
                          "afr": s["failures"] * DAY_PER_YEAR / s["drive_days"] if s["drive_days"] else None})
    return summary, by_model, by_age, model_age


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=Path("data/real/data_Q4_2025.zip"))
    parser.add_argument("--out", type=Path, default=Path("runs/real_data/backblaze_q4_2025"))
    args = parser.parse_args()
    summary, models, ages, model_ages = analyze(args.archive)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for filename, rows in (("by_model.csv", models), ("by_powered_age.csv", ages),
                           ("by_model_powered_age.csv", model_ages)):
        with (args.out / filename).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"model_groups={len(models)} age_bands_with_exposure={sum(x['drive_days'] > 0 for x in ages)}")
    print(f"results={args.out}")


if __name__ == "__main__":
    main()
