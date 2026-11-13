"""I4.10: условные коэффициенты с минимальной поддержкой и CV gate."""
from .restricted_forecast import fit_profile, predict

STATES = ("empty_repair_queue", "parked_full_buffer", "repair_transfer", "proof_transfer", "waiting_or_setup")


def fit_states(rows):
    models = {}
    for state in STATES:
        subset = [r for r in rows if r["state"] == state and r["known_work"] > 0]
        worlds = sorted({r["seed"] for r in subset})
        if len(worlds) < 2 or len(subset) < 5:
            model = {"coefficient": 1., "status": "insufficient_support", "edge_of_grid": False}
        else:
            model = fit_profile(subset)
        models[state] = {**model, "support_worlds": worlds, "support_rows": len(subset)}
    return models


def state_prediction(row, models):
    return predict(row, models[row["state"]]["coefficient"])


def train_gated(rows):
    seeds = sorted({r["seed"] for r in rows})
    if len(seeds) != 3:
        raise ValueError("Gate определён для трёх обучающих миров")
    folds = []
    for seed in seeds:
        models = fit_states([r for r in rows if r["seed"] != seed])
        test = [r for r in rows if r["seed"] == seed]
        baseline = sum((predict(r)-r["actual_window_volume"])**2 for r in test)/len(test)
        candidate = sum((state_prediction(r, models)-r["actual_window_volume"])**2 for r in test)/len(test)
        folds.append({"heldout_seed": seed, "baseline_mse": baseline, "candidate_mse": candidate,
                      "models": models})
    b = sum(r["baseline_mse"] for r in folds)/3
    c = sum(r["candidate_mse"] for r in folds)/3
    return {"use_state_model": c < b-1e-12, "cv_baseline_mse": b, "cv_candidate_mse": c,
            "folds": folds, "models": fit_states(rows)}
