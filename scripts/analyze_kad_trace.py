"""Stream a bzip2-compressed .avt availability trace and summarize sessions/gaps."""
from __future__ import annotations

import argparse
import bz2
import math
from array import array
from pathlib import Path

import numpy as np


def summarize(path: Path) -> dict[str, float | int]:
    durations = array("d")
    gaps = array("d")
    first_starts = array("d")
    last_ends = array("d")
    peers = sessions = zero_session_peers = one_session_peers = malformed = 0
    t_min, t_max = math.inf, -math.inf
    first_at_boundary = last_at_boundary = 0

    with bz2.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            row = line.strip()
            if not row or row.startswith("#"):
                continue
            fields = row.split()
            try:
                n = int(fields[1])
                values = [float(x) for x in fields[2:]]
                if n < 0 or len(values) != 2 * n or not all(map(math.isfinite, values)):
                    raise ValueError("invalid session count or time values")
                pairs = list(zip(values[::2], values[1::2]))
                if any(end < start for start, end in pairs):
                    raise ValueError("negative session duration")
                peers += 1
                sessions += n
                zero_session_peers += n == 0
                one_session_peers += n == 1
                first_starts.append(pairs[0][0] if n else math.nan)
                last_ends.append(pairs[-1][1] if n else math.nan)
                for i, (start, end) in enumerate(pairs):
                    durations.append(end - start)
                    t_min = min(t_min, start)
                    t_max = max(t_max, end)
                    if i:
                        gaps.append(start - pairs[i - 1][1])
            except (ValueError, IndexError):
                malformed += 1
                continue

    if not peers or not durations:
        raise ValueError("trace has no valid availability sessions")

    d = np.frombuffer(durations, dtype=np.float64)
    g = np.frombuffer(gaps, dtype=np.float64)
    gf = g[g >= 15.0 * 60.0]
    span = t_max - t_min
    # Boundary intervals are censored by the trace window; allow sub-second rounding.
    eps = 1.0
    first_at_boundary = sum(abs(x - t_min) <= eps for x in first_starts if math.isfinite(x))
    last_at_boundary = sum(abs(x - t_max) <= eps for x in last_ends if math.isfinite(x))

    qd = np.quantile(d, [0.5, 0.9, 0.95]) / 86400.0
    qg = np.quantile(g, [0.5, 0.9, 0.95]) / 86400.0 if len(g) else [math.nan] * 3
    qgf = np.quantile(gf, [0.5, 0.9, 0.95]) / 86400.0 if len(gf) else [math.nan] * 3
    gap_shape = (math.log(math.log(10.0) / math.log(2.0)) /
                 math.log(qg[1] / qg[0])) if len(g) and qg[0] > 0 and qg[1] > qg[0] else math.nan
    gap_scale_days = qg[0] / (math.log(2.0) ** (1.0 / gap_shape)) if math.isfinite(gap_shape) else math.nan
    gapf_shape = (math.log(math.log(10.0) / math.log(2.0)) /
                  math.log(qgf[1] / qgf[0])) if len(gf) and qgf[0] > 0 and qgf[1] > qgf[0] else math.nan
    gapf_scale_days = qgf[0] / (math.log(2.0) ** (1.0 / gapf_shape)) if math.isfinite(gapf_shape) else math.nan
    return {
        "peers": peers,
        "sessions": sessions,
        "zero_session_peers": zero_session_peers,
        "one_session_peers": one_session_peers,
        "malformed_rows": malformed,
        "trace_span_days": span / 86400.0,
        "session_days_mean": float(d.mean() / 86400.0),
        "session_days_p50": float(qd[0]),
        "session_days_p90": float(qd[1]),
        "session_days_p95": float(qd[2]),
        "gap_count": len(g),
        "gap_days_mean": float(g.mean() / 86400.0) if len(g) else math.nan,
        "gap_days_p50": float(qg[0]),
        "gap_days_p90": float(qg[1]),
        "gap_days_p95": float(qg[2]),
        "offline_transitions_observed": len(g),
        "offline_transition_rate_per_online_day": float(len(g) * 86400.0 / d.sum()),
        "rejoin_rate_per_offline_day": float(len(g) * 86400.0 / g.sum()) if len(g) else math.nan,
        "gap_weibull_shape_quantile_fit": float(gap_shape),
        "gap_weibull_scale_hours_quantile_fit": float(gap_scale_days * 24.0),
        "gaps_over_franchise_count": len(gf),
        "gaps_over_franchise_mean_hours": float(gf.mean() / 3600.0) if len(gf) else math.nan,
        "gaps_over_franchise_p50_hours": float(qgf[0] * 24.0),
        "gaps_over_franchise_p90_hours": float(qgf[1] * 24.0),
        "gaps_over_franchise_p95_hours": float(qgf[2] * 24.0),
        "gaps_over_franchise_weibull_shape_quantile_fit": float(gapf_shape),
        "gaps_over_franchise_weibull_scale_hours_quantile_fit": float(gapf_scale_days * 24.0),
        "first_sessions_at_left_boundary": first_at_boundary,
        "last_sessions_at_right_boundary": last_at_boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path,
                        default=Path("data/real/steiner-kad.avt.bz2"))
    args = parser.parse_args()
    result = summarize(args.trace)
    print(f"trace={args.trace}")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"{key}={value:.6g}")
        else:
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
