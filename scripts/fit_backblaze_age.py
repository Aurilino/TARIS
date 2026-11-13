"""Fit powered-on-age hazard ratios conditional on drive model."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

BANDS = ("0-1 powered years", "1-3 powered years", "3-5 powered years",
         "5-7 powered years", "7+ powered years")


def fit(directory: Path) -> dict:
    with (directory / "by_model.csv").open(encoding="utf-8", newline="") as f:
        models = {r["model"]: r for r in csv.DictReader(f)}
    exposure = defaultdict(lambda: np.zeros(len(BANDS), dtype=float))
    failures = defaultdict(lambda: np.zeros(len(BANDS), dtype=float))
    with (directory / "by_model_powered_age.csv").open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            model, band = row["model"], row["powered_age_band"]
            if band not in BANDS:
                continue
            # Published quarterly minimums, without claiming to reproduce their HDD/boot exclusions.
            m = models[model]
            if int(m["unique_drives_in_quarter"]) <= 100 or int(m["drive_days"]) <= 10_000:
                continue
            j = BANDS.index(band)
            exposure[model][j] = float(row["drive_days"])
            failures[model][j] = float(row["failures"])

    strata = [(e, y) for m, e in exposure.items() if (e > 0).sum() >= 2
              for y in (failures[m],) if y.sum() > 0]
    if len(strata) < 5:
        raise ValueError(f"too few model strata for age fit: {len(strata)}")
    E = np.stack([x[0] for x in strata])
    Y = np.stack([x[1] for x in strata])
    free = len(BANDS) - 1  # reference is 0–1 powered years

    def objective_gradient(beta):
        b = np.r_[0.0, beta]
        z = np.where(E > 0, np.log(np.maximum(E, 1.0)) + b, -np.inf)
        p = np.exp(z - logsumexp(z, axis=1, keepdims=True))
        value = float((Y.sum(axis=1) * logsumexp(z, axis=1) - (Y * b).sum(axis=1)).sum())
        gradient = (Y.sum(axis=1, keepdims=True) * p - Y).sum(axis=0)[1:]
        return value, gradient

    result = minimize(objective_gradient, np.zeros(free), jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 5000, "maxls": 50})
    beta = np.r_[0.0, result.x]
    # Observed conditional Fisher information for the log-rate ratios.
    z = np.where(E > 0, np.log(np.maximum(E, 1.0)) + beta, -np.inf)
    p = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    hess = np.zeros((free, free), dtype=float)
    for i in range(len(strata)):
        pi = p[i, 1:]
        hess += Y[i].sum() * (np.diag(pi) - np.outer(pi, pi))
    cov = np.linalg.pinv(hess)
    out = []
    for j, band in enumerate(BANDS):
        se = 0.0 if j == 0 else float(np.sqrt(max(0.0, cov[j - 1, j - 1])))
        b = float(beta[j])
        out.append({"powered_age_band": band, "rate_ratio_vs_0_1": float(np.exp(b)),
                    "ci95_low": float(np.exp(b - 1.96 * se)), "ci95_high": float(np.exp(b + 1.96 * se)),
                    "log_rate_ratio": b, "se": se,
                    "drive_days": int(E[:, j].sum()), "failures": int(Y[:, j].sum())})
    _, grad = objective_gradient(result.x)
    return {"strata_models": len(strata), "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message), "models_use_minimums_only": True,
            "gradient_inf_norm": float(np.max(np.abs(grad))),
            "age_ratios": out}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("runs/real_data/backblaze_q4_2025"))
    args = p.parse_args()
    result = fit(args.dir)
    path = args.dir / "age_model_adjusted.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"results={path}")


if __name__ == "__main__":
    main()
