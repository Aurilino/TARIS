"""Мир: узлы, истинный риск и заранее сгенерированные шкалы событий (шаг 3, пункты 3.1 и 3.4).

Всё, что здесь генерируется, не зависит от решений политик: общие случайные числа.
Микропрерывания генерируются позже, в модуле chain, из потоков rng.MICRO по (узел, эпоха).
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd
from scipy import stats

from .config import Config, HOUR, YEAR
from . import rng as R
from .crisis import Crises, OUTAGE_TAG

CLUSTER_TYPES = ("dc", "provider", "region")


class World:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._build_clusters()
        self._build_nodes()
        self._build_lifetimes()
        self._calibrate()
        self.crisis = None
        if cfg.crises_enabled:
            self._apply_crises()
        self._build_outages()
        self._build_silent_losses()
        if self.crisis is not None:
            self._finalize_crises()

    # ------------------------------------------------------------------ 3.1
    def _build_clusters(self):
        c = self.cfg
        g = R.stream(c.seed, R.NODES, 0)
        self.dc_provider = g.integers(0, c.n_providers, c.n_dc)
        self.dc_region = g.integers(0, c.n_regions, c.n_dc)
        self.dc_weight = g.dirichlet(np.full(c.n_dc, 2.0))          # дата-центры разного размера
        self.beta_region = g.normal(0.0, c.beta_region_sd, c.n_regions)

    def _build_nodes(self):
        c = self.cfg
        g = R.stream(c.seed, R.OPERATORS, 0)
        mean_size = 1.0 / c.operator_size_p
        H = c.horizon_days

        # Операторы: начальные (уже в сети) и приходящие пуассоновским потоком
        n_ops0 = int(round(c.n_initial_nodes / mean_size))
        ops_join = list(-g.exponential(c.initial_age_mean_days, n_ops0))
        n_new = g.poisson(c.node_join_rate / mean_size * H)
        ops_join += list(np.sort(g.uniform(0, H, n_new)))

        rows = []
        subnet_counter = np.zeros(c.n_dc, dtype=int)
        for o, t0 in enumerate(ops_join):
            size = int(g.geometric(c.operator_size_p))
            home_dc = int(g.choice(c.n_dc, p=self.dc_weight))
            subnet_counter[home_dc] += 1
            home_subnet = subnet_counter[home_dc]
            t = t0
            for k in range(size):
                if k > 0:
                    t = t + g.exponential(c.operator_join_gap_days)
                if t >= H:
                    break
                same_dc = (k == 0) or (g.random() < c.operator_same_dc_p)
                dc = home_dc if same_dc else int(g.choice(c.n_dc, p=self.dc_weight))
                if same_dc and (k == 0 or g.random() < c.operator_same_subnet_p):
                    subnet = home_subnet
                else:
                    subnet_counter[dc] += 1
                    subnet = subnet_counter[dc]
                rows.append(dict(owner=o, join_t=t, dc=dc, subnet=f"{dc}.{subnet}"))

        nodes = pd.DataFrame(rows)
        nodes.index.name = "node"
        n = len(nodes)
        gn = R.stream(c.seed, R.NODES, 1)
        nodes["provider"] = self.dc_provider[nodes["dc"]]
        nodes["region"] = self.dc_region[nodes["dc"]]
        nodes["wallet"] = nodes["owner"]                         # кошелёк выплат — у оператора общий
        nodes["hw_true"] = gn.choice(len(c.hw_classes), size=n, p=c.hw_probs)
        nodes["hw_claimed"] = nodes["hw_true"]                    # ложные заявления — в модуле adversary

        # Истинный множитель риска Theta (среднее 1, дисперсия a)
        a = c.heterogeneity_a
        if c.world_mode == "matched":
            theta = gn.gamma(1.0 / a, a, n)
        else:
            s2 = np.log1p(a)
            theta = np.exp(gn.normal(-s2 / 2, np.sqrt(s2), n))
        nodes["theta"] = theta

        # Индивидуальный разброс тяжести и шум микропрерываний
        nodes["sev_u"] = gn.normal(0.0, c.severity_node_sd, n)
        sm2 = c.micro_noise_sd ** 2
        nodes["micro_eps"] = np.exp(gn.normal(-sm2 / 2, c.micro_noise_sd, n))

        nodes["lin_pred"] = np.asarray(c.beta_hw)[nodes["hw_true"]] + self.beta_region[nodes["region"]]
        nodes["lam_silent"] = c.silent_rate
        self.beta0 = 0.0          # калибруется в _calibrate после построения жизней узлов
        self.nodes = nodes

    def _calibrate(self):
        """beta_0 подбирается так, чтобы средняя по сети частота страховых случаев равнялась mean_event_rate.
        Так оба режима мира (согласованный и несогласованный) имеют одинаковую общую частоту."""
        c, nd = self.cfg, self.nodes
        # по базовой жизни: кризисы (пункт 3.9) не должны менять калибровку
        expo = (nd["alive_to_base"] - nd["alive_from"]).clip(lower=0).to_numpy()
        e0 = self.expected_own_counts(life_end="alive_to_base")
        self.beta0 = float(np.log(c.mean_event_rate * expo.sum() / e0.sum()))
        lam_avg = np.where(expo > 0, e0 * np.exp(self.beta0) / np.maximum(expo, 1e-9), np.nan)
        lam_avg = np.where(np.isnan(lam_avg), np.nanmean(lam_avg), lam_avg)
        nd["lam_avg"] = lam_avg                                   # средняя истинная интенсивность за жизнь узла
        # Микропрерывания: lambda_micro = micro_rate * (lambda / средняя lambda)^b * eps
        nd["lam_micro"] = c.micro_rate * (lam_avg / lam_avg.mean()) ** c.micro_info_b * nd["micro_eps"]

    # ---- истинная интенсивность страховых случаев
    def age_factor(self, age_days: np.ndarray) -> np.ndarray:
        c = self.cfg
        age = np.maximum(age_days, 0.0)
        if c.world_mode == "matched":
            return (1.0 + age / c.age_unit_days) ** (c.kappa_age - 1.0)
        return 1.0 + c.b_inf * np.exp(-age / c.A_inf_days) + c.b_wear * (age / c.A_wear_days) ** 2

    def rate(self, node_idx, t) -> np.ndarray:
        """lambda_true узла в момент t (событий в сутки), без бурь."""
        nd = self.nodes
        node_idx = np.asarray(node_idx)
        # возраст — от исходной регистрации: отмена регистрации кризисом не должна менять потоки (3.9)
        join = nd["join_t_base"] if "join_t_base" in nd else nd["join_t"]
        age = np.asarray(t) - join.to_numpy()[node_idx]
        return (np.exp(self.beta0 + nd["lin_pred"].to_numpy()[node_idx])
                * nd["theta"].to_numpy()[node_idx] * self.age_factor(age))

    def churn_rate(self, age_days):
        c = self.cfg
        return c.churn_extra_new * np.exp(-np.maximum(age_days, 0) / c.churn_A_days) + c.churn_base

    # ------------------------------------------------------------ жизнь узлов
    @staticmethod
    def _thin(g, rate_fn, t0, t1, lam_max, first_only=False):
        """Неоднородный пуассоновский процесс методом прореживания на [t0, t1)."""
        if t1 <= t0:
            return np.empty(0)
        m = g.poisson(lam_max * (t1 - t0))
        ts = np.sort(g.uniform(t0, t1, m))
        keep = g.random(m) < rate_fn(ts) / lam_max
        ts = ts[keep]
        return ts[:1] if first_only else ts

    def _build_lifetimes(self):
        c, nd = self.cfg, self.nodes
        H = c.horizon_days
        leave = np.full(len(nd), np.inf)
        for i, j0 in enumerate(nd["join_t"].to_numpy()):
            g = R.stream(c.seed, R.CHURN, i)
            start = max(0.0, j0)
            lam_max = float(self.churn_rate(start - j0)) * 1.0001
            ts = self._thin(g, lambda t: self.churn_rate(t - j0), start, H, lam_max, first_only=True)
            if len(ts):
                leave[i] = ts[0]
        # Базовая жизнь — без кризисов. Все потоки узла генерируются на ней и потом обрезаются
        # по итоговой жизни (alive_to), поэтому кризисы не меняют остальные события узла.
        nd["leave_t_base"] = leave
        nd["alive_from"] = np.maximum(0.0, nd["join_t"])
        nd["alive_to_base"] = np.minimum(H, nd["leave_t_base"])
        nd["leave_t"] = nd["leave_t_base"]
        nd["alive_to"] = nd["alive_to_base"]

    # ------------------------------------------------------------------ 3.9
    def _apply_crises(self):
        """Кризисы строятся по базовой жизни; здесь из них получается итоговая жизнь узлов:
        принудительный уход или гибель (износ, массовый уход) и отменённые регистрации."""
        c, nd = self.cfg, self.nodes
        self.crisis = Crises(self).build()
        fr = self.crisis.frames()
        nd["join_t_base"] = nd["join_t"]
        nd["leave_reason"] = np.where(np.isfinite(nd["leave_t_base"]), "churn", "")
        forced = fr["forced"].sort_values(["node", "t"], kind="mergesort").drop_duplicates("node")
        idx = forced["node"].to_numpy()
        earlier = forced["t"].to_numpy() < nd["leave_t_base"].to_numpy()[idx]
        nd.loc[idx[earlier], "leave_t"] = forced["t"].to_numpy()[earlier]
        nd.loc[idx[earlier], "leave_reason"] = forced["reason"].to_numpy()[earlier]
        nd["alive_to"] = np.minimum(c.horizon_days, nd["leave_t"])
        # отменённая регистрация: узел так и не пришёл в сеть
        canc = fr["cancelled"]["node"].unique()
        nd["cancelled"] = False
        nd.loc[canc, "cancelled"] = True
        nd.loc[canc, "join_t"] = np.inf
        nd.loc[canc, "leave_t"] = np.inf
        nd.loc[canc, "leave_reason"] = ""
        nd.loc[canc, "alive_to"] = nd.loc[canc, "alive_from"]

    def _finalize_crises(self):
        """Таблицы истины кризисов (пункт 3.9), обрезанные по итоговой жизни узлов."""
        nd, cr = self.nodes, self.crisis
        fr = cr.frames()
        alive_to = nd["alive_to"].to_numpy()
        self.crises = cr.table()
        pe = fr["pre_events"]
        self.pre_events = pe[pe["start"].to_numpy() < alive_to[pe["node"].to_numpy()]].reset_index(drop=True)
        pa = fr["pre_audit"]
        pa = pa.assign(end=np.minimum(pa["end"].to_numpy(), alive_to[pa["node"].to_numpy()]))
        self.pre_audit = pa[pa["end"] > pa["start"]].reset_index(drop=True)
        self.forced = fr["forced"]
        self.cancelled = fr["cancelled"]
        self.versions = cr.versions
        nu = cr.node_updates
        self.node_updates = nu[nu["t"].to_numpy() < alive_to[nu["node"].to_numpy()]].reset_index(drop=True)

    # ------------------------------------------------------------------ 3.4
    def sample_duration(self, g, node_idx: int, m: int) -> np.ndarray:
        """Длительности простоев длиннее франшизы d (в сутках)."""
        c = self.cfg
        if m == 0:
            return np.empty(0)
        d_h = c.franchise_d_hours
        hw = int(self.nodes["hw_true"].iat[node_idx])
        u_i = float(self.nodes["sev_u"].iat[node_idx])
        U = g.random(m)
        if c.world_mode == "matched":
            k = c.weibull_shape
            th = c.weibull_scale_hours * np.exp(c.alpha_hw[hw] + u_i)
            L = th * ((d_h / th) ** k - np.log(U)) ** (1.0 / k)
        else:
            mu = np.log(c.lognorm_median_hours) + c.alpha_hw[hw] + u_i
            s = c.lognorm_sigma
            Fd = stats.norm.cdf((np.log(d_h) - mu) / s)
            L = np.exp(mu + s * stats.norm.ppf(Fd + U * (1 - Fd)))
        return L * HOUR

    def true_phi(self, node_idx: int) -> float:
        """Истинная вероятность полной гибели: P(L > T_rep | L > d)."""
        c = self.cfg
        hw = int(self.nodes["hw_true"].iat[node_idx])
        u_i = float(self.nodes["sev_u"].iat[node_idx])
        d_h, T_h = c.franchise_d_hours, c.T_rep_hours
        if c.world_mode == "matched":
            k = c.weibull_shape
            th = c.weibull_scale_hours * np.exp(c.alpha_hw[hw] + u_i)
            return float(np.exp(-(T_h / th) ** k + (d_h / th) ** k))
        mu = np.log(c.lognorm_median_hours) + c.alpha_hw[hw] + u_i
        s = c.lognorm_sigma
        return float(stats.norm.sf((np.log(T_h) - mu) / s) / stats.norm.sf((np.log(d_h) - mu) / s))

    def _build_outages(self):
        c, nd = self.cfg, self.nodes
        rows = []
        # Собственные простои узлов
        alive_to = nd["alive_to"].to_numpy()
        for i in range(len(nd)):
            j0, a0, a1 = nd["join_t"].iat[i], nd["alive_from"].iat[i], nd["alive_to_base"].iat[i]
            if a1 <= a0:
                continue
            grid = np.linspace(a0, a1, 200)
            lam_max = float(self.rate(np.full(200, i), grid).max()) * 1.05
            g = R.stream(c.seed, R.OUTAGE, i)
            ts = self._thin(g, lambda t: self.rate(np.full(len(t), i), t), a0, a1, lam_max)
            L = self.sample_duration(R.stream(c.seed, R.DURATION, i), i, len(ts))
            for t, l in zip(ts, L):
                if t < alive_to[i]:
                    rows.append((i, t, t + l, "own", -1, False))
        # Сырой поток собственных простоев (до слияния перекрытий) — для проверки частоты
        self.raw_own = pd.DataFrame([(r[0], r[1], r[2] - r[1]) for r in rows], columns=["node", "start", "dur"])
        self.raw_own_counts = self.raw_own["node"].value_counts().reindex(range(len(nd)), fill_value=0)

        # Шоки кластеров (согласованный мир) или бури (несогласованный)
        shock_rows = []
        if c.shocks_enabled:
            members = {
                "dc": nd.groupby("dc").groups,
                "provider": nd.groupby("provider").groups,
                "region": nd.groupby("region").groups,
            }
            for ti, ctype in enumerate(CLUSTER_TYPES):
                rate = c.shock_rate_per_year[ti] / YEAR
                for cl, idx in members[ctype].items():
                    g = R.stream(c.seed, R.SHOCK, ti, int(cl))
                    m = g.poisson(rate * c.horizon_days)
                    for t in np.sort(g.uniform(0, c.horizon_days, m)):
                        tag = ti * 1000 + int(cl)
                        if c.world_mode == "matched":
                            L, destroyed = self.shock_draw(g, c.shock_scale_hours[ti], c.shock_total_share)
                            hit = self.shock_rows(rows, idx, t, L, destroyed, "shock", tag)
                            shock_rows.append((ctype, int(cl), t, t + L, destroyed, hit))
                        else:
                            Dl = c.storm_hours * HOUR
                            gs = R.stream(c.seed, R.STORM, ti, int(cl), int(t * 1000))
                            hit = self.storm_rows(rows, gs, idx, t, Dl, c.storm_multiplier, "storm", tag)
                            shock_rows.append((ctype, int(cl), t, t + Dl, False, hit))
        self.shocks = pd.DataFrame(shock_rows, columns=["ctype", "cluster", "start", "end", "destroyed", "n_nodes"])

        # Кризисы (пункт 3.9): простои от исходов и короткие простои-предвестники
        if self.crisis is not None:
            self.crisis.outage_rows(rows)
            pe = self.crisis.frames()["pre_events"]
            for r in pe[pe["channel"] == "short"].itertuples(index=False):
                rows.append((int(r.node), r.start, r.end, "pre_short", OUTAGE_TAG + int(r.episode), False))

        out = pd.DataFrame(rows, columns=["node", "start", "end", "kind", "cluster", "destroyed"])
        out = self._merge_overlaps(out)
        alive_to = nd["alive_to"].to_numpy()[out["node"]]
        out = out[out["start"] < alive_to].copy()
        dur = out["end"] - out["start"]
        out["cls"] = np.where(out["destroyed"] | (dur > c.T_rep), "total", "partial")
        self.outages = out.reset_index(drop=True)

    # ---- общий шок и буря: используются шоками кластеров (3.4) и кризисами (3.9)
    def alive_at(self, idx, t, life_end: str = "alive_to") -> np.ndarray:
        """Узлы из idx, живые в момент t."""
        nd = self.nodes
        idx = np.asarray(idx)
        return idx[(nd["alive_from"].to_numpy()[idx] <= t) & (nd[life_end].to_numpy()[idx] > t)]

    def shock_draw(self, g, scale_hours: float, total_share: float):
        """Длительность общего шока (Вейбулл, длиннее франшизы) и признак полной гибели."""
        c = self.cfg
        k, th, d_h = c.shock_shape, scale_hours, c.franchise_d_hours
        L = th * ((d_h / th) ** k - np.log(g.random())) ** (1.0 / k) * HOUR
        destroyed = bool(g.random() < total_share)
        if destroyed:
            L = max(L, 1.5 * c.T_rep)
        return L, destroyed

    def shock_rows(self, rows, idx, t, L, destroyed, kind, tag) -> int:
        """Все живые узлы кластера уходят в простой одной длительности; возвращает число затронутых узлов."""
        alive = self.alive_at(idx, t)
        for i in alive:
            rows.append((int(i), t, t + L, kind, tag, destroyed))
        return len(alive)

    def storm_rows(self, rows, gs, idx, t, Dl, multiplier, kind, tag) -> int:
        """Буря: интенсивность простоев узлов кластера умножается на multiplier на время Dl.
        Случайные числа тянутся для всех узлов, живых по базовой жизни, а строки остаются только
        у живых по итоговой: так кризисы не сдвигают поток бури у остальных узлов."""
        alive = set(self.alive_at(idx, t).tolist())
        for i in self.alive_at(idx, t, "alive_to_base"):
            lam = float(self.rate([i], [t])[0]) * (multiplier - 1.0)
            m_i = gs.poisson(lam * Dl)
            if m_i:
                ts = np.sort(gs.uniform(t, t + Dl, m_i))
                L = self.sample_duration(gs, int(i), m_i)
                if int(i) in alive:
                    for tt, l in zip(ts, L):
                        rows.append((int(i), tt, tt + l, kind, tag, False))
        return len(alive)

    @staticmethod
    def _merge_overlaps(df: pd.DataFrame) -> pd.DataFrame:
        """Узел не может начать новый простой, не закончив текущий: перекрывающиеся простои сливаются."""
        df = df.sort_values(["node", "start"]).reset_index(drop=True)
        merged = []
        cur = None
        for r in df.itertuples(index=False):
            if cur is not None and r.node == cur[0] and r.start < cur[2]:
                cur[2] = max(cur[2], r.end)
                cur[5] = cur[5] or r.destroyed
                if cur[3] == "own" and r.kind != "own":
                    cur[3], cur[4] = r.kind, r.cluster
            else:
                if cur is not None:
                    merged.append(tuple(cur))
                cur = list(r)
        if cur is not None:
            merged.append(tuple(cur))
        return pd.DataFrame(merged, columns=df.columns)

    def _build_silent_losses(self):
        c, nd = self.cfg, self.nodes
        rows = []
        for i in range(len(nd)):
            a0, a1 = nd["alive_from"].iat[i], nd["alive_to_base"].iat[i]
            g = R.stream(c.seed, R.SILENT, i)
            lam = float(nd["lam_silent"].iat[i])
            for t in self._thin(g, lambda t: np.full(len(t), lam), a0, a1, lam):
                if t < nd["alive_to"].iat[i]:
                    rows.append((i, t))
        # тихие потери-предвестники износа (пункт 3.9)
        if self.crisis is not None:
            for r in self.crisis.frames()["pre_silent"].itertuples(index=False):
                if r.t < nd["alive_to"].iat[int(r.node)]:
                    rows.append((int(r.node), r.t))
        self.silent = pd.DataFrame(rows, columns=["node", "t"])

    # ------------------------------------------------------------- сервис
    def expected_own_counts(self, step_days: float = 1.0, life_end: str = "alive_to") -> np.ndarray:
        """Ожидаемое число собственных простоев каждого узла: интеграл lambda_true по жизни узла."""
        nd = self.nodes
        exp_n = np.zeros(len(nd))
        for i in range(len(nd)):
            a0, a1 = nd["alive_from"].iat[i], nd[life_end].iat[i]
            if a1 <= a0:
                continue
            m = max(2, int(np.ceil((a1 - a0) / step_days)) + 1)
            t = np.linspace(a0, a1, m)
            exp_n[i] = np.trapezoid(self.rate(np.full(m, i), t), t)
        return exp_n

    def fingerprint(self) -> str:
        """Хэш мира: два построения с одним зерном обязаны совпасть."""
        h = hashlib.sha256()
        h.update(self.cfg.to_json().encode())
        h.update(pd.util.hash_pandas_object(self.outages, index=False).values.tobytes())
        h.update(pd.util.hash_pandas_object(self.nodes[["owner", "join_t", "leave_t", "dc", "theta"]], index=False).values.tobytes())
        return h.hexdigest()[:16]
