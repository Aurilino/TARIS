"""Три политики для неизменённого основного Ledger.

Classic — исходный ReservePolicy; G — граф и немедленный ремонт;
Hybrid — исходный актуарный резерв и графовая доработка размещения.
"""
from dataclasses import dataclass
import numpy as np

from taris_sim.experience import Registry
from taris_sim.policy import OraclePolicy, ReservePolicy
from taris_sim.reserve import placement_loss_probability, window_tau

from .risk_graph import ObservedRiskGraph


@dataclass(frozen=True)
class VariantSettings:
    n: int = 13
    block_days: int = 30
    reserve_horizon_days: int = 365
    loss_limit: float = 0.001
    availability_limit: float = 0.08
    candidate_limit: int = 16
    swap_rounds: int = 2
    repair_mode: str = "thiele"

    def validate(self, k):
        if (any(type(value) is not int for value in (self.n, self.block_days, self.reserve_horizon_days,
                                                   self.candidate_limit, self.swap_rounds))
                or not np.isfinite(self.loss_limit) or not np.isfinite(self.availability_limit)
                or self.n <= k or self.block_days < 1 or self.reserve_horizon_days < 1
                or self.candidate_limit < 1 or self.swap_rounds < 0
                or not 0 < self.loss_limit <= 1 or not 0 <= self.availability_limit <= 1
                or self.repair_mode not in ("fixed", "thiele")):
            raise ValueError("Некорректные настройки вариантов; требуется n > k")


class GraphRefinement:
    def _initialize_graph(self, world, settings):
        self.settings = settings
        self.network_registry = Registry(world)
        self.graph = ObservedRiskGraph(self.network_registry)
        self.graph_stats = {"calls": 0, "accepted_swaps": 0, "actuarial_rejections": 0,
                            "availability_rejections": 0, "unavailability_exceeded": 0, "cold_calls": 0}
        self.decision_log = []
        self.placement_log = []
        self.repair_request_log = []

    def place(self, obj, t, view):
        nodes = super().place(obj, t, view)
        self.placement_log.append({"obj": int(obj), "time": float(t), "nodes": list(map(int, nodes)),
            "graph_score": self.graph.score(nodes),
            "estimated_unavailability": self.graph.object_unavailability(nodes, self.k)})
        return nodes

    def _record_requests(self, requests, t):
        self.repair_request_log.extend({"obj": int(obj), "time": float(t),
            "indices": list(map(int, indices)), "targets": list(map(int, targets))}
            for obj, indices, targets in requests)
        return requests

    def _valid_concentration(self, nodes, original):
        # Сохраняем максимальную концентрацию исходного допустимого набора каждого типа.
        for domains in (self.dc, self.prov, self.reg, self.own, self.subnet_domain):
            if domains is None:
                continue
            limit = int(np.unique(domains[original], return_counts=True)[1].max())
            if int(np.unique(domains[nodes], return_counts=True)[1].max()) > limit:
                return False
        return True

    def _refine(self, chosen, candidates, held, t, actuarial=False):
        if not chosen or self.graph.fitted_at is None:
            self.graph_stats["cold_calls"] += 1
            return chosen
        self.graph_stats["calls"] += 1
        held = set(map(int, held))
        chosen = list(map(int, chosen))
        original = np.asarray(sorted(held | set(chosen)), int)
        pool = sorted(set(map(int, candidates)) - held,
                      key=lambda node: (float(self.graph.own[node]), node))[:self.settings.candidate_limit]
        base_loss = None
        if actuarial and self.model.fitted:
            base_loss = placement_loss_probability(self.rates, original, t, window_tau(self.world.cfg), self.k)[0]
        before = self.graph.score(original)
        for _ in range(self.settings.swap_rounds):
            best = None
            current_nodes = sorted(held | set(chosen))
            current_score = self.graph.score(current_nodes)
            current_availability = self.graph.object_unavailability(current_nodes, self.k)
            for pos in range(len(chosen)):
                for node in pool:
                    if node in chosen:
                        continue
                    trial = chosen.copy()
                    trial[pos] = node
                    all_nodes = np.asarray(sorted(held | set(trial)), int)
                    if not self._valid_concentration(all_nodes, original):
                        continue
                    score = self.graph.score(all_nodes)
                    if score >= current_score - 1e-12:
                        continue
                    if self.graph.object_unavailability(all_nodes, self.k) > max(
                            current_availability, self.settings.availability_limit) + 1e-12:
                        self.graph_stats["availability_rejections"] += 1
                        continue
                    if base_loss is not None:
                        loss = placement_loss_probability(self.rates, all_nodes, t,
                                                          window_tau(self.world.cfg), self.k)[0]
                        if loss > base_loss + 1e-12:
                            self.graph_stats["actuarial_rejections"] += 1
                            continue
                    row = (score, tuple(sorted(trial)), trial)
                    if best is None or row[:2] < best[:2]:
                        best = row
            if best is None:
                break
            chosen = best[2]
            self.graph_stats["accepted_swaps"] += 1
        all_nodes = sorted(held | set(chosen))
        availability = self.graph.object_unavailability(all_nodes, self.k)
        exceeded = availability > self.settings.availability_limit
        self.graph_stats["unavailability_exceeded"] += int(exceeded)
        self.decision_log.append({"kind": "candidate_refinement", "time": float(t), "held": sorted(held), "chosen": chosen.copy(),
            "graph_before": before, "graph_after": self.graph.score(all_nodes),
            "estimated_unavailability": availability, "availability_target_exceeded": exceeded,
            "actuarial_guard_active": base_loss is not None,
            "actuarial_reference_loss": base_loss})
        return chosen


class GraphPolicy(GraphRefinement, OraclePolicy):
    def __init__(self, world, settings):
        settings.validate(world.cfg.k)
        super().__init__(world, settings.n, oracle=False, claimed=True)
        self._initialize_graph(world, settings)
        self.time = 0.0
        self.name = "TARIS-G"

    def _choose_spread(self, rng, cand, m, held=()):
        baseline = super()._choose_spread(rng, cand, m, held)
        return self._refine(baseline, cand, held, self.time)

    def on_epoch(self, t, view):
        self.time = float(t)
        self.graph.update(t, view)
        self.w = 1 / np.maximum(np.nan_to_num(self.graph.own, nan=1.0), 1e-6)
        return self._record_requests(super().on_epoch(t, view), t)


class HybridPolicy(GraphRefinement, ReservePolicy):
    def __init__(self, world, settings):
        settings.validate(world.cfg.k)
        super().__init__(world, settings.n, mode=settings.repair_mode,
            rates="agent", placement="agent_capped", block=settings.block_days,
            H=settings.reserve_horizon_days, eps=settings.loss_limit,
            R_fixed=settings.n - world.cfg.k - 1, placement_max_n=settings.n)
        self._initialize_graph(world, settings)
        self.time = 0.0
        self.name = "TARIS-Hybrid"

    def _choose_spread(self, rng, cand, m, held=()):
        baseline = super()._choose_spread(rng, cand, m, held)
        return self._refine(baseline, cand, held, self.time, actuarial=True)

    def on_epoch(self, t, view):
        self.time = float(t)
        self.graph.update(t, view)
        return self._record_requests(super().on_epoch(t, view), t)


def make_agent(variant, world, settings):
    settings.validate(world.cfg.k)
    if variant == "classic":
        policy = ReservePolicy(world, settings.n, mode=settings.repair_mode,
            rates="agent", placement="agent_capped", block=settings.block_days,
            H=settings.reserve_horizon_days, eps=settings.loss_limit,
            R_fixed=settings.n - world.cfg.k - 1, placement_max_n=settings.n)
        policy.name = "TARIS Classic"
        return policy
    if variant == "graph":
        return GraphPolicy(world, settings)
    if variant == "hybrid":
        return HybridPolicy(world, settings)
    raise ValueError("Вариант должен быть classic, graph или hybrid")
