"""I7: проверяемые минимумы на малых доменных планах, не случайные миры."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from itertools import combinations
import hashlib
import json
from subprojects.taris_variants.domain_reserve import minimum_layout, unconstrained_minimum


def independent_minimum(caps, k, b):
    nodes = [d for d, count in enumerate(caps) for _ in range(count)]
    domains = sorted(set(nodes))
    failures = list(combinations(domains, min(b, len(domains))))
    for n in range(k, len(nodes)+1):
        for selected in combinations(range(len(nodes)), n):
            if all(sum(nodes[node] not in failed for node in selected) >= k for failed in failures):
                return n
    return None


def main():
    definitions = [
        {"capacities": [1]*4, "k": 2, "failed_domains": 2},
        {"capacities": [1]*5, "k": 2, "failed_domains": 3},
        {"capacities": [2]*4, "k": 2, "failed_domains": 3},
        {"capacities": [2]*5, "k": 5, "failed_domains": 2},
        {"capacities": [1, 1, 3, 3, 3], "k": 5, "failed_domains": 2},
        {"capacities": [2, 2, 2, 1], "k": 2, "failed_domains": 3},
        {"capacities": [2]*4, "k": 2, "failed_domains": 3, "max_n": 7},
        {"capacities": [5, 5], "k": 2, "failed_domains": 2},
        {"capacities": [2]*4, "k": 2, "failed_domains": 2, "max_states": 1},
        {"capacities": [1, 1], "k": 2, "failed_domains": 0}]
    rows = []
    for inputs in definitions:
        result = minimum_layout(**inputs)
        actual = independent_minimum(inputs["capacities"], inputs["k"], inputs["failed_domains"])
        if result["certified_minimum"]:
            assert result["minimum_n"] == actual
        if result["certified_infeasible"]:
            assert actual is None or actual > inputs.get("max_n", sum(inputs["capacities"]))
        if result["status"] == "search_limit":
            assert result["feasible"] is None
        rows.append({"inputs": inputs, "result": result, "independent_minimum_n": actual,
            "capacity_free_lower_bound": unconstrained_minimum(inputs["k"], inputs["failed_domains"],
                sum(c > 0 for c in inputs["capacities"]))})
    folder = Path(__file__).parent
    output = folder/"domain_reserve_i7"
    output.mkdir(exist_ok=True)
    report = {"schema": "taris-domain-reserve-example/1", "random_worlds": 0,
        "independent_node_subset_check": True, "rows": rows,
        "source_hashes": {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in
                          ("domain_reserve.py", "report_domain_reserve.py", "EXPERIMENT_I7_DOMAIN.md")},
        "real_world_risk_probability": None}
    (output/"results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    lines = ["# I7 — минимум доменного запаса", "", "Один новый разный номер на узел. Это условная задача планирования, не оценка частоты катастроф.", "",
        "| Узлы по ДЦ | k | b | Нижняя граница без ёмкостей | Результат | Минимум в бюджете | Независимый минимум без бюджетного ограничения |",
        "|---|---:|---:|---:|---|---:|---:|"]
    for row in rows:
        inputs, result = row["inputs"], row["result"]
        display = lambda value: "—" if value is None else str(value)
        lines.append(f'| {inputs["capacities"]} | {inputs["k"]} | {inputs["failed_domains"]} | '
            f'{display(row["capacity_free_lower_bound"])} | {result["status"]} | {display(result["minimum_n"])} | '
            f'{display(row["independent_minimum_n"])} |')
    lines += ["", "search_limit означает неизвестную разрешимость, не отказ. При max_n=7математический минимум8не помещается в бюджет. "
        "При нехватке доменных ёмкостей нижняя граница может быть недостижима. Ни один результат не сертифицирует скрытую "
        "целостность текущих данных, истинность меток ДЦ или успешность будущей записи."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("I7: 10доменных задач и независимые минимумы сохранены")


if __name__ == "__main__":
    main()
