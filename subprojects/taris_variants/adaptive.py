"""I3: выбор начального числа чанков с общим проверяющим модулем.

Политики I1/I2 не изменяются. Минимум относится только к найденным предложениям.
"""
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
import numpy as np

from taris_sim.experience import Experience, Registry
from taris_sim.risk import RiskModel, EstimatedRates
from taris_sim.reserve import placement_risk_metrics, window_tau, effective_reserve
from .agents import make_agent
from .risk_graph import ObservedRiskGraph


@dataclass(frozen=True)
class SizingRequirements:
    min_n: int = 3
    max_n: int = 6
    trials: int = 4
    window_loss_limit: float = 0.00004
    availability_limit: float = 0.08
    min_observed_hours: int = 48
    fallback: str = "best_effort"

    def validate(self, k):
        integers = (self.min_n, self.max_n, self.trials, self.min_observed_hours)
        if (any(type(v) is not int for v in integers) or self.min_n <= k or self.max_n < self.min_n
                or self.trials < 1 or self.min_observed_hours < 0
                or not np.isfinite(self.window_loss_limit) or not 0 < self.window_loss_limit <= 1
                or not np.isfinite(self.availability_limit) or not 0 < self.availability_limit <= 1
                or self.fallback not in ("best_effort", "reject")):
            raise ValueError("Некорректные требования к числу чанков")


class LearningOnlyView:
    def __init__(self, view):
        self.view = view

    def needs_repair(self):
        return []

    def __getattr__(self, name):
        return getattr(self.view, name)


@contextmanager
def object_size(policy, n, k):
    old_n = policy.n
    old_threshold = getattr(policy, "R_fixed", None)
    old_memo = getattr(policy, "memo", None)
    if old_memo is not None:
        if not hasattr(policy, "_sizing_memos"):
            policy._sizing_memos = {old_n: old_memo}
        policy.memo = policy._sizing_memos.setdefault(n, {})
    policy.n = n
    if old_threshold is not None:
        policy.R_fixed = n - k - 1
    try:
        yield
    finally:
        policy.n = old_n
        if old_threshold is not None:
            policy.R_fixed = old_threshold
        if old_memo is not None:
            policy.memo = old_memo


class AdaptiveAgent:
    def __init__(self, variant, world, settings, requirements):
        requirements.validate(world.cfg.k)
        self.variant, self.world, self.k = variant, world, world.cfg.k
        self.requirements = requirements
        self.settings = replace(settings, n=requirements.max_n)
        self.base = make_agent(variant, world, self.settings)
        self.name = self.base.name + " / sizing"
        self.n = requirements.max_n
        self.graph = getattr(self.base, "graph", None)
        if self.graph is None:
            self.graph = ObservedRiskGraph(Registry(world))
        if variant == "graph":
            # Общий датчик риска не управляет моментом ремонта G и не использует Тиле.
            self.exp = Experience(world, settings.block_days)
            self.model = RiskModel(self.exp, claimed=True)
            self.rates = EstimatedRates(world, self.model)
            self.fits = []
        else:
            self.exp, self.model, self.rates = self.base.exp, self.base.model, self.base.rates
            self.fits = self.base.fits
        self.cache = {}
        self.placement_log = []
        self.repair_request_log = []
        self.status_counts = Counter()
        self.placement_risk = {"checked": 0, "feasible": 0, "violations": 0, "unscored": 0}
        self._observed_hours = np.zeros(len(world.nodes), int)

    @property
    def w(self):
        return self.base.w

    @property
    def graph_stats(self):
        return getattr(self.base, "graph_stats", None)

    def _estimate(self, nodes, t):
        nodes = list(map(int, nodes))
        availability = self.graph.object_unavailability(nodes, self.k)
        observed = min((int(self._observed_hours[node]) for node in nodes), default=0)
        scored = bool(self.model.fitted and observed >= self.requirements.min_observed_hours)
        loss = cost = dropped = None
        if self.model.fitted:
            loss, dropped, cost = placement_risk_metrics(self.rates, nodes, t,
                                                       window_tau(self.world.cfg), self.k)
            if not all(np.isfinite(v) for v in (loss, dropped, cost)):
                raise ArithmeticError("Неконечная оценка риска размещения")
        passed = bool(scored and loss <= self.requirements.window_loss_limit
                      and availability <= self.requirements.availability_limit)
        return {"nodes": nodes, "n": len(nodes), "estimated_unavailability": float(availability),
            "estimated_window_loss": loss, "actuarial_cost_proxy": cost,
            "dropped_shock_mass": dropped, "min_observed_hours": observed,
            "scored": scored, "requirements_passed": passed,
            "graph_score": self.graph.score(nodes)}

    def _rank(self, row):
        if self.variant == "graph":
            return row["graph_score"], row["actuarial_cost_proxy"] or 0, tuple(sorted(row["nodes"]))
        return row["actuarial_cost_proxy"] or 0, row["graph_score"], tuple(sorted(row["nodes"]))

    def _propose(self, candidates, n, held=()):
        with object_size(self.base, n + len(held), self.k):
            return self.base._choose_spread(self.base._g, candidates, n, held=held)

    def _choose(self, obj, t, view):
        req = self.requirements
        candidates = view.candidates()
        proposals, seen = [], set()
        warm = bool(self.model.fitted)
        sizes = range(req.min_n, min(req.max_n, len(candidates)) + 1) if warm else range(
            min(req.max_n, len(candidates)), req.min_n - 1, -1)
        for n in sizes:
            local = []
            for _ in range(req.trials):
                nodes = self._propose(candidates, n)
                key = tuple(sorted(nodes))
                if len(nodes) != n or key in seen:
                    continue
                seen.add(key)
                row = self._estimate(nodes, t)
                proposals.append(row)
                if row["requirements_passed"]:
                    local.append(row)
            if local:
                chosen = min(local, key=self._rank)
                return self._decision(obj, t, "estimated_feasible", chosen, proposals)
            if not warm and proposals:
                break
        if not proposals:
            status = "insufficient_capacity" if len(candidates) < req.min_n else "no_placement_found"
            return self._decision(obj, t, status, None, proposals)
        scored = [row for row in proposals if row["scored"]]
        if scored:
            chosen = min(scored, key=lambda row: (
                max(0.0, row["estimated_window_loss"] / req.window_loss_limit - 1)
                + max(0.0, row["estimated_unavailability"] / req.availability_limit - 1),
                -row["n"], self._rank(row)))
            status = "no_feasible_found_best_effort"
        else:
            max_n = max(row["n"] for row in proposals)
            chosen = min((row for row in proposals if row["n"] == max_n), key=self._rank)
            status = "unscored_cold_start"
        if req.fallback == "reject":
            return self._decision(obj, t, "rejected_" + status, None, proposals)
        return self._decision(obj, t, status, chosen, proposals)

    def _decision(self, obj, t, status, chosen, proposals):
        return {"obj": int(obj), "time": float(t), "status": status,
            "nodes": chosen["nodes"].copy() if chosen else [],
            "n": chosen["n"] if chosen else 0,
            "requirements_passed": bool(chosen and chosen["requirements_passed"]),
            "scored": bool(chosen["scored"] if chosen else any(row["scored"] for row in proposals)), "selected": chosen,
            "unique_proposals_checked": len(proposals),
            "window_days": float(window_tau(self.world.cfg)),
            "window_loss_limit": self.requirements.window_loss_limit,
            "availability_limit": self.requirements.availability_limit,
            "minimum_proven": False, "scope": "initial_placement"}

    def target_fragments(self, obj, t, view):
        key = (int(obj), float(t))
        if key not in self.cache:
            self.cache[key] = self._choose(obj, t, view)
        return max(self.requirements.min_n, self.cache[key]["n"])

    def place(self, obj, t, view):
        key = (int(obj), float(t))
        if key not in self.cache:
            self.cache[key] = self._choose(obj, t, view)
        decision = self.cache.pop(key)
        self.placement_log.append(decision)
        self.status_counts[decision["status"]] += 1
        if decision["scored"]:
            self.placement_risk["checked"] += 1
            self.placement_risk["feasible" if decision["requirements_passed"] else "violations"] += 1
        else:
            self.placement_risk["unscored"] += 1
        return decision["nodes"].copy()

    def on_epoch(self, t, view):
        # Обновляем исходные модели, но решения ремонта строим с размером данного объекта.
        self.base.on_epoch(t, LearningOnlyView(view))
        if self.variant == "classic":
            self.graph.update(t, view)
        if self.variant == "graph":
            self.exp.update(t, view)
            if t > 0 and int(t) % self.settings.block_days == 0:
                self.model.fit(t)
                if self.model.fitted:
                    self.fits.append(float(t))
            if self.model.fitted:
                self.model.refresh(t)
        self._observed_hours = sum((known.astype(int) for _, _, known in self.graph.rows),
                                   np.zeros(self.graph.N, int))
        requests = []
        for obj in view.needs_repair():
            holders = view.holders(obj)
            if len(holders) < self.k:
                continue
            n_obj = int(view.L.obj_n[obj])  # наблюдаемый размер кода, не истина потерь
            with object_size(self.base, n_obj, self.k):
                if self.variant != "graph":
                    threshold = self.base._threshold(obj, holders, t, view, n_obj=n_obj)
                    reserve = len(holders) - self.k
                    if self.base.corr and self.model.fitted:
                        reserve, _ = effective_reserve(self.rates, view, obj, t, self.k)
                    if reserve > threshold:
                        continue
                indices = sorted(view.missing(obj))
                if not indices:
                    continue
                candidates, choices = view.candidates(), []
                for _ in range(self.requirements.trials):
                    nodes = self.base._choose_spread(self.base._g, candidates, len(indices), held=holders)
                    if len(nodes) == len(indices):
                        estimate = self._estimate(sorted(set(nodes) | holders), t)
                        choices.append((nodes, estimate))
                if not choices:
                    continue
                feasible = [row for row in choices if row[1]["requirements_passed"]]
                eligible = feasible if feasible else choices
                if not feasible and self.requirements.fallback == "reject":
                    continue
                def repair_rank(row):
                    estimate = row[1]
                    if feasible or not estimate["scored"]:
                        return (not estimate["scored"], 0.0, self._rank(estimate))
                    excess = max(0.0, estimate["estimated_window_loss"] / self.requirements.window_loss_limit - 1)
                    excess += max(0.0, estimate["estimated_unavailability"] / self.requirements.availability_limit - 1)
                    return False, excess, self._rank(estimate)
                nodes, estimate = min(eligible, key=repair_rank)
                requests.append((obj, indices, nodes))
                self.repair_request_log.append({"obj": int(obj), "time": float(t), "n_obj": n_obj,
                    "indices": indices, "targets": list(map(int, nodes)),
                    "requirements_passed": estimate["requirements_passed"],
                    "status": "estimated_feasible" if estimate["requirements_passed"] else "repair_best_effort"})
        return requests

    def alarm(self, t, view):
        return self.base.alarm(t, view)

    def migrate(self, t, view):
        return []

    def release(self, t, view):
        return []
