"""Модель риска агента TARIS (шаг 5, пункт 5.1, подшаги А2–А5, А7).

Только по таблице опыта (experience.py), без истины мира:
- частота (А2): пуассоновская GLM с экспозицией, IRLS на numpy (пункт 2.2);
- тяжесть (А3): вероятность полной гибели phi — логистическая GLM по страховым случаям с ясным исходом
  (альтернатива пункта 2.2; уход узла — простой, который не кончился, то есть полная гибель); условная
  вероятность P(L > T_rep | L > a) для RBNS — эмпирическая по случаям, дожившим до a;
- кредибилити (А4): Бюльман–Штраубу по отношению «фактическое к ожидаемому» (пункт 2.3);
- тихие потери (А5): пуассоновская оценка по узло-суткам с фрагментами, отдельно для устройств;
- EstimatedRates (А7): тот же интерфейс, что reserve.TrueRates, — оценки подставляются в резерв информации.
- шоки кластеров (А6): эпизоды одновременных простоев большинства узлов кластера (experience.shock_episodes);
  их простои исключаются из собственной частоты узлов, частота шоков — по типу кластера с поправкой к общей.

Варианты факторов: claimed=True — с заявленными классом оборудования и признаком устройства;
claimed=False — только проверяемые (регион, возраст, опыт микропрерываний).
"""
from __future__ import annotations

import numpy as np

from .config import HOUR

AGE_UNIT = 30.0           # возраст в формуле — ln(1 + age / 30 суток), форма Вейбулла по возрасту (пункт 2.2)


# ------------------------------------------------------------------ GLM (IRLS)
def glm_poisson(X, y, offset, iters: int = 50, tol: float = 1e-9, ridge: float = 1e-6):
    """Пуассоновская GLM с логарифмической связью и смещением ln e: возвращает (beta, ковариационная матрица)."""
    X, y, off = np.asarray(X, float), np.asarray(y, float), np.asarray(offset, float)
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(max(y.sum(), 0.5) / np.exp(off).sum())
    R = ridge * np.eye(X.shape[1]); R[0, 0] = 0.0
    for _ in range(iters):
        eta = off + X @ beta
        mu = np.exp(eta)
        H = X.T @ (X * mu[:, None]) + R
        g = X.T @ (y - mu) - R @ beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    mu = np.exp(off + X @ beta)
    return beta, np.linalg.inv(X.T @ (X * mu[:, None]) + R)


def glm_logit(X, y, iters: int = 50, tol: float = 1e-9, ridge: float = 1e-3):
    """Логистическая GLM (IRLS) с небольшой гребневой поправкой (исходов «полная гибель» немного)."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    p0 = np.clip(y.mean() if len(y) else 0.5, 1e-4, 1 - 1e-4)
    beta[0] = np.log(p0 / (1 - p0))
    R = ridge * np.eye(X.shape[1]); R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        W = np.maximum(p * (1 - p), 1e-9)
        step = np.linalg.solve(X.T @ (X * W[:, None]) + R, X.T @ (y - p) - R @ beta)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


# ------------------------------------------------------------------ модель риска
class RiskModel:
    """Оценки агента на момент последнего обучения fit(exp, t)."""

    def __init__(self, exp, claimed: bool = True, credibility: bool = True, micro: bool = True):
        self.exp, self.claimed, self.credibility, self.micro = exp, claimed, credibility, micro
        reg = exp.reg
        self.reg = reg
        self.fitted = False
        self.lam_sil = np.zeros(exp.N)

    # ---- признаки
    def _cols(self, region, age, micro, hw, device):
        r = self.reg
        cols = [np.ones(len(region))]
        cols += [(region == j).astype(float) for j in range(1, r.n_region)]
        cols.append(np.log1p(np.asarray(age, float) / AGE_UNIT))
        cols.append(np.log(np.asarray(micro, float) / self.m0) if self.micro else np.zeros(len(region)))
        if self.claimed:
            cols += [(hw == j).astype(float) for j in range(1, r.n_hw)]
            cols.append(np.asarray(device, float))
        return np.column_stack(cols)

    def _phi_cols(self, region, age, hw, device):
        r = self.reg
        cols = [np.ones(len(region))]
        cols += [(region == j).astype(float) for j in range(1, r.n_region)]
        cols.append(np.log1p(np.asarray(age, float) / AGE_UNIT))
        if self.claimed:
            cols += [(hw == j).astype(float) for j in range(1, r.n_hw)]
            cols.append(np.asarray(device, float))
        return np.column_stack(cols)

    @staticmethod
    def _keep(X):
        """Столбцы без разброса (например, нет устройств в мире) выкидываются, чтобы матрица была невырожденной."""
        keep = np.ones(X.shape[1], bool)
        keep[1:] = X[:, 1:].std(0) > 0
        return keep

    # ---- обучение
    def fit(self, t: float):
        exp, r = self.exp, self.reg
        iv = exp.intervals(t)
        ctypes = ("region", "provider", "dc") if self.claimed else ("region", "provider")
        self.episodes, drop = exp.shock_episodes(t, iv, ctypes)
        tab = exp.table(t, iv, drop)
        tab = tab[tab["e"] > 0]
        if len(tab) == 0 or tab["N"].sum() < 5:
            return self
        self.t_fit = float(t)
        # А2: частота
        self.m0 = float(np.average(tab["micro_rate"], weights=tab["e"]))
        X = self._cols(tab["region"].to_numpy(), tab["age"].to_numpy(), tab["micro_rate"].to_numpy(),
                       tab["hw"].to_numpy(), tab["device"].to_numpy())
        self.keep = self._keep(X)
        X = X[:, self.keep]
        y, e = tab["N"].to_numpy(), tab["e"].to_numpy()
        self.beta, self.cov = glm_poisson(X, y, np.log(e))
        lam_hat = np.exp(X @ self.beta)
        mu = e * lam_hat
        self.pearson = float(((y - mu) ** 2 / mu).sum() / max(len(y) - X.shape[1], 1))
        # робастная ковариация с кластеризацией по узлам: случаи одного узла связаны его скрытым множителем
        G = np.zeros((exp.N, X.shape[1]))
        np.add.at(G, tab["node"].to_numpy(), X * (y - mu)[:, None])
        self.cov_robust = self.cov @ (G.T @ G) @ self.cov
        # А4: кредибилити — отношение «фактическое к ожидаемому» и вес опыта по узлам
        node = tab["node"].to_numpy()
        w = np.bincount(node, weights=mu, minlength=exp.N)
        n_act = np.bincount(node, weights=y, minlength=exp.N)
        has = w > 0
        Xi = np.where(has, n_act / np.where(has, w, 1.0), 1.0)
        I = int(has.sum())
        self.a_hat = max(0.0, float((w[has] * (Xi[has] - 1.0) ** 2).sum() - I) / float(w[has].sum()))
        if self.credibility and self.a_hat > 0:
            K = 1.0 / self.a_hat
            self.Z = w / (w + K)
        else:
            self.Z = np.zeros(exp.N)
        self.cred = self.Z * Xi + (1.0 - self.Z)
        self.w_exp = w
        # А3: тяжесть — phi по случаям с ясным исходом
        ev = exp.events(t, iv, drop)
        res = ev[np.isfinite(ev["total"])]
        # Среднее превышение франшизы для частичных случаев. Индивидуальные средние
        # стягиваются к сетевому среднему с той же кредибилити, что и частота.
        partial = res[res["total"] == 0]
        pn = partial["node"].to_numpy(dtype=int)
        px = np.maximum(partial["dur"].to_numpy(dtype=float) - exp.d, 0.0)
        pc = np.bincount(pn, minlength=exp.N).astype(float)
        ps = np.bincount(pn, weights=px, minlength=exp.N).astype(float)
        if len(px):
            m0_partial = float(px.mean())
            m_node = np.divide(ps, pc, out=np.full(exp.N, m0_partial), where=pc > 0)
            self.m_partial = self.Z * m_node + (1.0 - self.Z) * m0_partial
            self.m_partial_global = m0_partial
        else:
            # Без наблюдавшихся частичных исходов тариф этого компонента не определён.
            self.m_partial = np.full(exp.N, np.nan)
            self.m_partial_global = float("nan")
        nd = res["node"].to_numpy()
        age_ev = np.maximum(res["start"].to_numpy() - exp.join[nd], 0.0)
        Xp = self._phi_cols(r.region[nd], age_ev, r.hw[nd], r.is_device[nd].astype(int))
        self.keep_phi = self._keep(Xp)
        self.gamma = glm_logit(Xp[:, self.keep_phi], res["total"].to_numpy())
        # доля «уходов» среди полных гибелей: простой, не кончившийся за 7 суток (путь с задержкой T_rep, пункт 3.11)
        tot = res[res["total"] == 1]
        long_ = (~tot["done"].to_numpy()) & (tot["dur"].to_numpy() > 7.0)
        self.churn_share = float((long_.sum() + 0.5) / (len(tot) + 1.0))
        # P(L > T_rep | L > a): эмпирическая по сетке a (часы) — для RBNS
        self.grid_h = np.r_[np.arange(exp.d / HOUR, 6.0, 0.25), np.arange(6.0, exp.T_rep / HOUR + 1e-9, 1.0)]
        dur_h = res["dur"].to_numpy() / HOUR
        tot_ = res["total"].to_numpy()
        self.p_tot_grid = np.array([(tot_[dur_h > a].sum() + 0.5) / ((dur_h > a).sum() + 1.0) for a in self.grid_h])
        # А5: тихие потери — по узло-суткам с фрагментами (устройства отдельно)
        sil = np.array([n for _, n in exp.silent], int)
        cnt = np.bincount(sil, minlength=exp.N).astype(float) if len(sil) else np.zeros(exp.N)
        dev = r.is_device
        lam = np.zeros(exp.N)
        for grp in (dev, ~dev):
            if grp.any():
                lam[grp] = (cnt[grp].sum() + 0.5) / max(exp.node_days_stored[grp].sum(), 1.0)
        self.lam_sil = lam
        # А6: шоки кластеров — частота по типу (кластеро-сутки с узлами в реестре), к общей частоте 0,5 эпизода
        ep = self.episodes
        member = {"region": r.region, "provider": r.provider, "dc": r.dc}
        E = {}
        for ct in ctypes:
            m = member[ct]
            first = {}
            for cl in np.unique(m[m >= 0]):
                j = exp.join[m == cl]
                first[cl] = max(float(np.min(j)), 0.0) if np.isfinite(j).any() else np.inf
            E[ct] = float(sum(max(t - f, 0.0) for f in first.values()))
        n_all = len(ep)
        nu0 = (n_all + 0.5) / max(sum(E.values()), 1.0)
        self.nu = {ct: (float((ep["ctype"] == ct).sum()) + 0.5) / (E[ct] + 0.5 / nu0) for ct in ctypes}
        tot = ep["total"].dropna()
        self.p_shock_total = float((tot.sum() + 0.1) / (len(tot) + 1.0))
        self.eta = {ct: self.nu[ct] * self.p_shock_total for ct in ctypes}
        # Тарифная тяжесть шоков по отдельным публичным доменам. Частота использует
        # тот же пуассоновский приор, что и pooled-оценка; доля гибели и длительность
        # частичных исходов стягиваются к сетевым значениям при нехватке эпизодов.
        self.shock_types = tuple(ctypes)
        self.shock_nu, self.shock_phi, self.shock_m = {}, {}, {}
        member = {"region": r.region, "provider": r.provider, "dc": r.dc}
        shock_iv = iv.iloc[np.flatnonzero(drop)]
        shock_node = shock_iv["node"].to_numpy(dtype=int)
        shock_start = shock_iv["start"].to_numpy(dtype=float)
        shock_dur = shock_iv["dur_seen"].to_numpy(dtype=float)
        shock_end = shock_iv["end"].to_numpy(dtype=float)
        prior_exposure = 0.5 / max(nu0, 1e-12)
        for ct in ctypes:
            ids = member[ct]
            for cl in np.unique(ids[ids >= 0]):
                key = (ct, int(cl))
                cluster_ep = ep[(ep["ctype"] == ct) & (ep["cluster"] == cl)]
                joins = exp.join[ids == cl]
                joins = joins[np.isfinite(joins)]
                exposure = max(float(t - max(float(joins.min()), 0.0)), 0.0) if len(joins) else 0.0
                self.shock_nu[key] = (len(cluster_ep) + 0.5) / (exposure + prior_exposure)
                resolved = cluster_ep["total"].dropna().to_numpy(dtype=float)
                self.shock_phi[key] = float((resolved.sum() + self.p_shock_total) / (len(resolved) + 1.0))
                partial_excess = []
                cluster_nodes = ids[shock_node] == cl
                for start in cluster_ep["start"].to_numpy(dtype=float):
                    # Эпизод объединяет простои, начавшиеся в пределах десяти минут.
                    in_episode = cluster_nodes & (shock_start >= start - 1e-9) & (shock_start <= start + 10.0 / 1440.0 + 1e-9)
                    partial = in_episode & np.isfinite(shock_end) & (shock_dur <= exp.T_rep)
                    if partial.any():
                        partial_excess.extend(np.maximum(shock_dur[partial] - exp.d, 0.0).tolist())
                if partial_excess and np.isfinite(self.m_partial_global):
                    self.shock_m[key] = float((sum(partial_excess) + self.m_partial_global) / (len(partial_excess) + 1.0))
                elif partial_excess:
                    self.shock_m[key] = float(np.mean(partial_excess))
                else:
                    self.shock_m[key] = self.m_partial_global
        self.fitted = True
        return self

    # ---- прогноз
    def lam_class(self, nodes, t: float) -> np.ndarray:
        """Интенсивность страховых случаев по классу (GLM, без кредибилити) в момент t."""
        r, nodes = self.reg, np.asarray(nodes, int)
        age = np.maximum(t - self.exp.join[nodes], 0.0)
        micro = self.micro_now[nodes]
        X = self._cols(r.region[nodes], age, micro, r.hw[nodes], r.is_device[nodes].astype(int))[:, self.keep]
        return np.exp(X @ self.beta)

    def lam(self, nodes, t: float) -> np.ndarray:
        """Интенсивность после кредибилити (пункт 2.3)."""
        nodes = np.asarray(nodes, int)
        return self.lam_class(nodes, t) * self.cred[nodes]

    def phi(self, nodes, t: float) -> np.ndarray:
        r, nodes = self.reg, np.asarray(nodes, int)
        age = np.maximum(t - self.exp.join[nodes], 0.0)
        X = self._phi_cols(r.region[nodes], age, r.hw[nodes], r.is_device[nodes].astype(int))[:, self.keep_phi]
        return 1.0 / (1.0 + np.exp(-(X @ self.gamma)))

    def partial_excess(self, nodes) -> np.ndarray:
        """Ожидаемое превышение франшизы для частичного страхового случая, в сутках.

        Сетевое среднее стягивает узловые средние с весом кредибилити Z. Возвращает NaN,
        если в истории ещё нет ни одного частичного случая: тарифный компонент тогда
        нельзя честно оценить по данным.
        """
        return self.m_partial[np.asarray(nodes, int)]

    def refresh(self, t: float):
        """Признаки, меняющиеся каждую эпоху без переобучения: опыт микропрерываний."""
        self.micro_now = self.exp.micro_rate_now(t)


class EstimatedRates:
    """Оценки агента в интерфейсе reserve.TrueRates (подшаг А7)."""

    destroyed_share = 1.0         # гибель данных при шоке сеть не отличает от долгого простоя: консервативно — сразу

    def __init__(self, world, model: RiskModel):
        self.w = world               # только конфигурация среды (T_rep, задержки): истина узлов не читается
        self.m = model

    @property
    def join(self):
        return np.where(np.isfinite(self.m.exp.join), self.m.exp.join, 0.0)

    @property
    def lam_sil(self):
        return self.m.lam_sil

    def parts(self, nodes, t: float) -> dict:
        nodes = np.asarray(nodes, int)
        tot = self.m.lam(nodes, t) * self.m.phi(nodes, t)
        s = self.m.churn_share
        return {"own": tot * (1.0 - s), "churn": tot * s, "silent": self.m.lam_sil[nodes]}

    def fragment_loss(self, nodes, t: float) -> np.ndarray:
        p = self.parts(nodes, t)
        return p["own"] + p["churn"] + p["silent"]

    def shock_clusters(self, nodes, t: float = 0.0):
        """Кластеры узлов nodes с оценённой интенсивностью шока, уносящего фрагменты: (тип, номер, eta, маска)."""
        nodes = np.asarray(nodes, int)
        r = self.m.reg
        member = {"region": r.region, "provider": r.provider, "dc": r.dc}
        out = []
        for ct in ("dc", "provider", "region"):
            eta = self.m.eta.get(ct, 0.0)
            if eta <= 0:
                continue
            ids = member[ct][nodes]
            for cl in np.unique(ids):
                if cl >= 0:
                    out.append((ct, int(cl), float(eta), ids == cl))
        return out

    def p_total_given(self, nodes, a_hours) -> np.ndarray:
        """P(L > T_rep | L > a) для текущих простоев (RBNS)."""
        return np.interp(np.asarray(a_hours, float), self.m.grid_h, self.m.p_tot_grid)
