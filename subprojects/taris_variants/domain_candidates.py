"""I7.1: план нового объекта по явно переданному публичному снимку.

Не адаптер NetworkView: вызывающая сторона ещё должна собрать эти поля
и применить дополнительные ограничения политики. План не подтверждает запись.
"""
import math

from .domain_reserve import minimum_layout, validate_count


FIELDS = frozenset({"node", "dc", "free_bytes", "known_online", "eligible"})


def plan_candidates(rows, *, fragment_bytes, k, failed_domains,
                    excluded_nodes=(), pending_nodes=(), max_n=None,
                    max_states=1_000_000):
    """Один новый номер на узел; dc=-1 не считается независимым доменом.

    eligible — предварительный допуск, а не замена совместных ограничений
    на владельцев/провайдеров. Такие ограничения здесь не сертифицируются.
    """
    if type(fragment_bytes) not in (int, float) or not math.isfinite(fragment_bytes) or fragment_bytes <= 0:
        raise ValueError("Размер фрагмента должен быть положительным конечным числом")
    blocked = set()
    for node in (*excluded_nodes, *pending_nodes):
        validate_count(node)
        blocked.add(node)
    seen = set()
    domains = {}
    rejected = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Нужны только явно разрешённые поля публичного снимка")
        node, dc = row["node"], row["dc"]
        validate_count(node)
        if node in seen:
            raise ValueError("Узел повторяется в снимке")
        seen.add(node)
        if type(dc) is not int or dc < -1:
            raise ValueError("Домен должен быть целым номером ДЦ или -1")
        free = row["free_bytes"]
        if type(free) not in (int, float) or not math.isfinite(free) or free < 0:
            raise ValueError("Некорректная наблюдаемая свободная ёмкость")
        if type(row["known_online"]) is not bool or type(row["eligible"]) is not bool:
            raise ValueError("Признаки допуска и доступности должны быть логическими")
        reasons = []
        if node in blocked:
            reasons.append("excluded_or_pending")
        if dc == -1:
            reasons.append("unknown_failure_domain")
        if not row["known_online"]:
            reasons.append("not_known_online")
        if not row["eligible"]:
            reasons.append("not_eligible")
        if free < fragment_bytes:
            reasons.append("insufficient_space")
        if reasons:
            rejected.append({"node": node, "reasons": reasons})
        else:
            domains.setdefault(dc, []).append(node)
    domain_ids = sorted(domains)
    buckets = [sorted(domains[dc]) for dc in domain_ids]
    result = minimum_layout([len(nodes) for nodes in buckets], k, failed_domains,
                            max_n=max_n, max_states=max_states)
    plan = []
    if result["feasible"] is True:
        for dc, nodes, count in zip(domain_ids, buckets, result["layout"]):
            for node in nodes[:count]:
                plan.append({"index": len(plan), "node": node, "dc": dc})
    return {"schema": "taris-domain-plan/1", "solver": result,
            "domain_ids": domain_ids, "fragment_bytes": fragment_bytes,
            "plan": plan, "excluded": sorted(rejected, key=lambda row: row["node"]),
            "admission_decision": "not_executed", "acknowledged_fragments": 0,
            "certificate_scope": "fresh_plan_known_dc_one_index_per_node",
            "joint_owner_provider_constraints_checked": False,
            "risk_probability": None, "real_world_guarantee": False}
