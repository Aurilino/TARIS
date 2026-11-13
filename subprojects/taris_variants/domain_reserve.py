"""I7: точный минимум разных номеров для ограниченного класса доменных отказов.

counts/capacities относятся к новым разным номерам, один номер на узел.
Физические дубликаты и неизвестная целостность не входят в сертификат.
"""
class SearchLimit(Exception):
    pass


def validate_count(value, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("Нужны целые неотрицательные количества")


def worst_surviving(counts, failed_domains):
    counts = tuple(counts)
    validate_count(failed_domains)
    for value in counts:
        validate_count(value)
    return sum(sorted(counts, reverse=True)[failed_domains:])


def unconstrained_minimum(k, failed_domains, available_domains):
    validate_count(k, positive=True)
    validate_count(failed_domains)
    validate_count(available_domains)
    if available_domains <= failed_domains:
        return None
    # Целочисленная форма ceil исключает погрешность больших целых.
    remaining = available_domains-failed_domains
    level = (k+remaining-1)//remaining
    return k+failed_domains*level


def minimum_layout(capacities, k, failed_domains, max_n=None, max_states=1_000_000):
    capacities = tuple(capacities)
    validate_count(k, positive=True)
    validate_count(failed_domains)
    for value in capacities:
        validate_count(value)
    if max_n is not None:
        validate_count(max_n)
    if max_states is not None:
        validate_count(max_states, positive=True)
    domains = sum(value > 0 for value in capacities)
    lower = unconstrained_minimum(k, failed_domains, domains)
    base = {"k": k, "failed_domains": failed_domains, "capacities": list(capacities),
        "max_n": max_n, "max_states": max_states,
        "available_domains": domains, "minimum_n": None, "layout": None,
        "certified_minimum": False, "certified_infeasible": False,
        "risk_probability": None, "real_world_guarantee": False}
    if lower is None:
        return {**base, "status": "infeasible_domains", "feasible": False,
                "certified_infeasible": True, "states": 0, "lower_bound_n": None}
    if worst_surviving(capacities, failed_domains) < k:
        return {**base, "status": "infeasible_capacity", "feasible": False,
                "certified_infeasible": True, "states": 0, "lower_bound_n": lower}
    upper = min(sum(capacities), max_n if max_n is not None else sum(capacities))
    if upper < lower:
        return {**base, "status": "infeasible_budget", "feasible": False,
                "certified_infeasible": True, "states": 0, "lower_bound_n": lower}
    suffix = [0]*(len(capacities)+1)
    for index in range(len(capacities)-1, -1, -1):
        suffix[index] = suffix[index+1]+capacities[index]
    states = 0

    def search(index, remaining, prefix):
        nonlocal states
        states += 1
        if max_states is not None and states > max_states:
            raise SearchLimit
        if remaining < 0 or remaining > suffix[index]:
            return None
        if index == len(capacities):
            return prefix if remaining == 0 and worst_surviving(prefix, failed_domains) >= k else None
        smallest = max(0, remaining-suffix[index+1])
        largest = min(capacities[index], remaining)
        for value in range(smallest, largest+1):
            result = search(index+1, remaining-value, prefix+(value,))
            if result is not None:
                return result
        return None

    for n in range(lower, upper+1):
        try:
            layout = search(0, n, ())
        except SearchLimit:
            return {**base, "status": "search_limit", "feasible": None,
                    "states": states-1, "lower_bound_n": n}
        if layout is not None:
            return {**base, "status": "optimal", "feasible": True, "minimum_n": n,
                "layout": list(layout), "certified_minimum": True, "states": states,
                "lower_bound_n": n, "worst_surviving": worst_surviving(layout, failed_domains)}
    return {**base, "status": "infeasible_budget", "feasible": False,
            "certified_infeasible": True, "states": states, "lower_bound_n": upper+1}
