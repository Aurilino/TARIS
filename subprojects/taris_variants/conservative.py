"""I3.2: защитные сценарии параметров и временных блоков.

Только прошлый сетевой опыт. TrueRates применяется исключительно внешним аудитором.
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import gamma

from taris_sim.reserve import placement_risk_metrics, window_tau
from .adaptive import AdaptiveAgent


@dataclass(frozen=True)
class GuardSettings:
    quantile: float = 0.95
    prior_node_days: float = 30.0
    own_rate_floor: float = 0.003
    silent_rate_prior: float = 0.0005
    prior_cluster_days: float = 365.0
    cluster_rate_floor: float = 0.0003
    bootstrap_replicas: int = 16
    block_hours: int = 24
    seed: int = 21001

    def validate(self):
        if (not 0.5 < self.quantile < 1 or any(not np.isfinite(v) or v <= 0 for v in (
                self.prior_node_days, self.own_rate_floor, self.silent_rate_prior,
                self.prior_cluster_days, self.cluster_rate_floor))
                or type(self.bootstrap_replicas) is not int or self.bootstrap_replicas < 1
                or type(self.block_hours) is not int or self.block_hours < 1
                or type(self.seed) is not int or self.seed < 0):
            raise ValueError("Некорректные параметры защитного слоя")


def gamma_rate_upper(counts, exposure, prior_rate, prior_exposure, quantile):
    counts, exposure = np.asarray(counts, float), np.asarray(exposure, float)
    if (not np.isfinite(counts).all() or not np.isfinite(exposure).all()
            or np.any(counts < 0) or np.any(exposure < 0)
            or not all(np.isfinite(v) and v > 0 for v in (prior_rate, prior_exposure))
            or not 0 < quantile < 1):
        raise ValueError("Неверные счётчики или экспозиция")
    return gamma.ppf(quantile, prior_rate * prior_exposure + counts,
                     scale=1 / (prior_exposure + exposure))


class GuardRates:
    def __init__(self, experience, point_rates, settings):
        settings.validate()
        self.exp, self.point, self.settings = experience, point_rates, settings
        self.time = None
        self.own_upper = np.zeros(experience.N)
        self.silent_upper = np.zeros(experience.N)
        self.cluster_upper = {}
        self.diagnostics = {}

    def refresh(self, t):
        if abs(self.exp.t - t) > 1e-9:
            raise ValueError("Для исторической оценки нужен сохранённый снимок опыта")
        if self.time == t:
            return
        cfg, exp = self.settings, self.exp
        intervals = exp.intervals(t)
        table = exp.table(t, intervals)
        exposure = (np.bincount(table.node.to_numpy(), weights=table.e.to_numpy(), minlength=exp.N)
                    if len(table) else np.zeros(exp.N))
        events = exp.events(t, intervals)
        # Неразрешённый исход не записывается как благополучный.
        adverse = events[(events.total == 1) | events.total.isna()]
        counts = np.bincount(adverse.node.to_numpy().astype(int), minlength=exp.N).astype(float)
        pooled = max(cfg.own_rate_floor, float(counts.sum()) / max(float(exposure.sum()), 1.0))
        self.own_upper = gamma_rate_upper(counts, exposure, pooled, cfg.prior_node_days, cfg.quantile)
        silent = np.bincount([int(node) for observed_at, node in exp.silent if observed_at <= t], minlength=exp.N)
        self.silent_upper = gamma_rate_upper(silent, exp.node_days_stored,
            cfg.silent_rate_prior, cfg.prior_node_days, cfg.quantile)
        episodes, _ = exp.shock_episodes(t, intervals)
        r = exp.reg
        membership = {"dc": r.dc, "provider": r.provider, "region": r.region}
        cluster_data = {}
        for kind, labels in membership.items():
            total_exposure = 0.0
            for group in np.unique(labels[np.isfinite(exp.join)]):
                if group < 0:
                    continue
                joined = exp.join[(labels == group) & np.isfinite(exp.join)]
                total_exposure += max(0.0, t - max(0.0, float(joined.min())))
            count = int((episodes.ctype == kind).sum()) if len(episodes) else 0
            rate = float(gamma_rate_upper(count, total_exposure, cfg.cluster_rate_floor,
                                         cfg.prior_cluster_days, cfg.quantile))
            self.cluster_upper[kind] = rate
            cluster_data[kind] = {"episodes": count, "cluster_days": total_exposure, "upper_rate": rate}
        self.time = float(t)
        self.diagnostics = {"time": float(t), "adverse_events": float(counts.sum()),
            "available_node_days": float(exposure.sum()), "pooled_prior_rate": pooled,
            "silent_events": int(silent.sum()), "clusters": cluster_data}

    def fragment_loss(self, nodes, t):
        self.refresh(t)
        nodes = np.asarray(nodes, int)
        return np.maximum(self.point.fragment_loss(nodes, t), self.own_upper[nodes] + self.silent_upper[nodes])

    def shock_clusters(self, nodes, t=0):
        self.refresh(t)
        nodes = np.asarray(nodes, int)
        point = {(kind, int(group)): float(rate) for kind, group, rate, _ in self.point.shock_clusters(nodes, t)}
        r = self.exp.reg
        membership = {"dc": r.dc, "provider": r.provider, "region": r.region}
        result = []
        for kind, labels in membership.items():
            selected = labels[nodes]
            for group in np.unique(selected):
                if group >= 0:
                    rate = max(self.cluster_upper[kind], point.get((kind, int(group)), 0.0))
                    result.append((kind, int(group), rate, selected == group))
        return result


class AvailabilityGuard:
    def __init__(self, graph, settings):
        self.graph, self.settings = graph, settings
        self.version = None
        self.weights = None

    def prepare(self):
        graph, cfg = self.graph, self.settings
        if self.version == graph.updates:
            return
        historical = len(graph.rows)
        prior_count = len(graph.states) - historical
        raw = graph.weights * (graph.prior_hours + historical)
        replicas = []
        groups = {}
        for index, (observed_at, _, _) in enumerate(graph.rows):
            block = (int(round(observed_at * 24)) - 1) // cfg.block_hours
            groups.setdefault(block, []).append(index)
        keys = sorted(groups)
        rng = np.random.default_rng([cfg.seed, int(round(graph.fitted_at * 24))])
        for _ in range(cfg.bootstrap_replicas):
            counts = np.zeros(historical)
            if keys:
                for key in rng.choice(keys, size=len(keys), replace=True):
                    counts[groups[int(key)]] += 1
            weights = raw.copy()
            weights[prior_count:] *= counts
            weights /= weights.sum()
            replicas.append(weights)
        self.weights = np.asarray(replicas)
        self.version = graph.updates

    def upper(self, nodes, k):
        self.prepare()
        nodes = np.asarray(nodes, int)
        failure = self.graph.states[:, nodes].sum(axis=1) > len(nodes) - k
        return max(self.graph.object_unavailability(nodes, k), float((self.weights @ failure).max()))


class ConservativeAgent(AdaptiveAgent):
    def __init__(self, variant, world, settings, requirements, guard=None):
        super().__init__(variant, world, settings, requirements)
        self.guard_settings = guard or GuardSettings()
        self.guard_settings.validate()
        self.guard_rates = GuardRates(self.exp, self.rates, self.guard_settings)
        self.availability_guard = AvailabilityGuard(self.graph, self.guard_settings)
        self.name += " / conservative"

    def _estimate(self, nodes, t):
        row = super()._estimate(nodes, t)
        row["point_window_loss"] = row["estimated_window_loss"]
        row["point_unavailability"] = row["estimated_unavailability"]
        row["estimated_unavailability"] = self.availability_guard.upper(nodes, self.k)
        row["guard_window_loss"] = None
        if self.model.fitted:
            stress, _, _ = placement_risk_metrics(self.guard_rates, nodes, t, window_tau(self.world.cfg), self.k)
            if not np.isfinite(stress):
                raise ArithmeticError("Неконечная защитная оценка риска")
            row["guard_window_loss"] = stress
            row["estimated_window_loss"] = max(row["estimated_window_loss"], stress)
        row["requirements_passed"] = bool(row["scored"]
            and row["estimated_window_loss"] <= self.requirements.window_loss_limit
            and row["estimated_unavailability"] <= self.requirements.availability_limit)
        row["guard_quantile"] = self.guard_settings.quantile
        row["object_confidence_level"] = None
        return row
