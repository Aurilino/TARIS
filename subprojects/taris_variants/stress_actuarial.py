"""Условная стоимость явно заданной стресс-смеси; не калиброванный тариф."""
from math import isclose, isfinite

MODES = ("control", "close", "after_decision", "medium", "far")


def mixture_components(scenario_means, weights=None):
    if weights is None:
        return {"expected_resource_cost": None, "expected_lost_volume": None,
                "status": "unknown_scenario_weights"}
    if set(weights) != set(MODES) or any(not isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Нужны пять неотрицательных весов сценариев")
    if not isclose(sum(weights.values()), 1., abs_tol=1e-12, rel_tol=0.):
        raise ValueError("Сумма весов должна быть единицей")
    rows = {row["mode"]: row for row in scenario_means}
    if len(rows) != len(scenario_means) or set(rows) != set(MODES):
        raise ValueError("Нужна ровно одна средняя строка каждой ветви")
    for row in rows.values():
        if any(not isfinite(row[key]) or row[key] < 0 for key in ("resource_cost", "lost_admitted_volume")):
            raise ValueError("Некорректная стоимость или тяжесть")
    return {"expected_resource_cost": sum(weights[mode]*rows[mode]["resource_cost"] for mode in MODES),
            "expected_lost_volume": sum(weights[mode]*rows[mode]["lost_admitted_volume"] for mode in MODES),
            "status": "conditional_on_supplied_weights"}


def conditional_cost(scenario_means, weights=None, loss_price=None):
    if loss_price is not None and (not isfinite(loss_price) or loss_price < 0):
        raise ValueError("Цена информации должна быть неотрицательной")
    result = mixture_components(scenario_means, weights)
    known = loss_price is not None and result["expected_resource_cost"] is not None
    return {**result, "loss_price": loss_price,
        "conditional_total_cost": result["expected_resource_cost"]+loss_price*result["expected_lost_volume"] if known else None,
        "full_actuarial_reserve": None, "real_world_scenario_probabilities_estimated": False}


def break_even_loss_price(left, right, weights):
    a, b = mixture_components(left, weights), mixture_components(right, weights)
    if a["expected_resource_cost"] is None:
        return {"status": "unknown_scenario_weights", "loss_price": None}
    cost_diff = a["expected_resource_cost"]-b["expected_resource_cost"]
    loss_diff = a["expected_lost_volume"]-b["expected_lost_volume"]
    if loss_diff == 0:
        return {"status": "equal_for_all_prices" if cost_diff == 0 else "no_finite_crossing", "loss_price": None}
    price = -cost_diff/loss_diff
    return {"status": "nonnegative_crossing" if price >= 0 else "negative_crossing_outside_domain",
            "loss_price": price}
