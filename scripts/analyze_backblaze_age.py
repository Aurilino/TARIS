"""Aggregate Backblaze failures by model and SMART-9 powered-on age."""
from __future__ import annotations

import argparse
import csv
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

DAY_PER_YEAR = 365.25
AGE_BANDS = ("0-1 powered years", "1-3 powered years", "3-5 powered years",
             "5-7 powered years", "7+ powered years")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", nargs="?", type=Path, default=Path("data/real/data_Q4_2025.zip"))
    parser.add_argument("--out", type=Path, default=Path("runs/real_data/backblaze_q4_2025/by_model_powered_age.csv"))
    args = parser.parse_args()
    stats = defaultdict(lambda: [0, 0])
    with zipfile.ZipFile(args.archive) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
        for idx, name in enumerate(names, 1):
            with zf.open(name) as raw:
                reader = pd.read_csv(raw, usecols=lambda c: c.strip().lower() in
                                     {"model", "failure", "smart_9_raw"}, chunksize=500_000,
                                     dtype={"model": "string"}, low_memory=False)
                for chunk in reader:
                    chunk.columns = [c.strip().lower() for c in chunk.columns]
                    if not {"model", "failure", "smart_9_raw"}.issubset(chunk.columns):
                        raise ValueError(f"{name}: required fields absent")
                    failure = pd.to_numeric(chunk["failure"], errors="coerce")
                    age = pd.to_numeric(chunk["smart_9_raw"], errors="coerce") / (24 * DAY_PER_YEAR)
                    band = pd.cut(age, [0, 1, 3, 5, 7, float("inf")], right=False,
                                  labels=AGE_BANDS, include_lowest=True)
                    data = pd.DataFrame({"model": chunk["model"].fillna("(unknown)"),
                                         "band": band, "failure": failure}).dropna(subset=["band", "failure"])
                    grouped = data.groupby(["model", "band"], observed=True)["failure"].agg(["count", "sum"])
                    for (model, label), row in grouped.iterrows():
                        acc = stats[(str(model), str(label))]
                        acc[0] += int(row["count"])
                        acc[1] += int(row["sum"])
            print(f"{idx}/{len(names)} {Path(name).name}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "powered_age_band", "drive_days", "failures", "afr"])
        for (model, band), (days, failures) in sorted(stats.items()):
            writer.writerow([model, band, days, failures, failures * DAY_PER_YEAR / days if days else ""])
    print(f"model_age_groups={len(stats)} output={args.out}")


if __name__ == "__main__":
    main()
