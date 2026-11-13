"""Exploratory bridge from Backblaze drive AFR to TARIS node downtime.

This is an arithmetic sensitivity only. It does not edit Config or inject events
into the simulator. The bridge probability and repair time are explicit unknowns.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--afr", type=float, default=0.01115521,
                   help="raw annualized Backblaze Q4 2025 all-row AFR")
    p.add_argument("--trep-hours", type=float, default=24.0)
    p.add_argument("--out", type=Path,
                   default=Path("runs/real_data/backblaze_q4_2025/node_mapping_sensitivity.csv"))
    args = p.parse_args()
    if not 0 <= args.afr <= 1 or args.trep_hours < 0:
        p.error("AFR must be in [0,1] and T_rep must be nonnegative")

    rows = []
    for bridge in (0.01, 0.10, 1.00):
        annual_node_events = args.afr * bridge
        for repair_hours in (6.0, 24.0, 48.0, 168.0):
            annual_unavailable_fraction = annual_node_events * repair_hours / 8760.0
            declared_outages = annual_node_events if repair_hours > args.trep_hours else 0.0
            rows.append({
                "raw_drive_afr": args.afr,
                "p_node_outage_given_drive_failure": bridge,
                "mean_repair_hours_assumed": repair_hours,
                "node_outage_events_per_1000_node_years": annual_node_events * 1000,
                "unavailable_node_percent_of_time": annual_unavailable_fraction * 100,
                "outages_exceeding_Trep_per_1000_node_years": declared_outages * 1000,
                "Trep_hours": args.trep_hours,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"raw_afr={args.afr:.8%}; bridge values=1%, 10%, 100%; "
          f"repair hours=6, 24, 48, 168; rows={len(rows)}; output={args.out}")


if __name__ == "__main__":
    main()
