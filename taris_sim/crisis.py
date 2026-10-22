"""Кризисы и предвестники (шаг 3, пункт 3.9).

Всё строится из потока rng.CRISIS по базовой жизни узлов, поэтому не зависит ни от решений политики,
ни от остальных потоков мира (общие случайные числа). Ключ каждого подпотока — тип кризиса, кластер
и номер эпизода, а не сквозной номер: изменение частоты одного типа не сдвигает другие типы.

Построение в две фазы:
  1. Crises(world).build() — шкала эпизодов (кризисы и ложные тревоги), версии ПО, предвестники,
     принудительные уходы и гибели узлов, отменённые регистрации;
  2. outage_rows(rows) — простои от исходов кризисов; вызывается из world, когда итоговая жизнь
     узлов уже известна.
Флаг «это предвестник» и шкала кризисов — только «взгляд истины».
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn

from .config import HOUR, YEAR
from . import rng as R

TYPES = ("storm", "cascade", "wear", "exodus", "software", "fire")
T_ID = {t: k + 1 for k, t in enumerate(TYPES)}
SCHED, FALSE, PRE, OUT, VER, CASC = range(1, 7)          # подпотоки rng.CRISIS
CH_XI, CH_MICRO, CH_SHORT, CH_SILENT, CH_CHURN, CH_JOIN = range(1, 7)

# каналы предвестника по типам кризиса
CHANNELS = {
    "storm": ("micro", "short"),
    "cascade": ("micro", "short"),
    "software": ("micro", "short"),
    "wear": ("silent", "audit"),
    "exodus": ("churn", "join"),
    "fire": (),
}
OUTAGE_TAG = 100000            # метка cluster у простоев от кризисов: OUTAGE_TAG + номер эпизода


def type_params(cfg, typ: str) -> dict:
    """Параметры типа кризиса из config (поля cr_<тип>_*) с поправками несогласованного мира."""
    pre = f"cr_{typ}_"
    p = {f[len(pre):]: getattr(cfg, f) for f in cfg.__dataclass_fields__ if f.startswith(pre)}
    if cfg.world_mode == "mismatched":
        if "s" in p:
            p["s"] *= cfg.mism_s_factor
        if "q" in p:
            p["q"] *= cfg.mism_q_factor
    return p


class Crises:
    def __init__(self, world):
        self.w = world
        self.cfg = world.cfg
        nd = world.nodes
        self.N = len(nd)
        self.H = float(self.cfg.horizon_days)
        self.a0 = nd["alive_from"].to_numpy()
        self.a1 = nd["alive_to_base"].to_numpy()
        self.join = nd["join_t"].to_numpy()
        self.episodes: list[dict] = []
        self.pre_rows, self.silent_rows, self.audit_rows = [], [], []
        self.forced_rows, self.cancel_rows = [], []

    # ------------------------------------------------------------- общее
    def g_shape(self, u):
        """Форма роста предвестника g(u): плавная в согласованном мире, ступенька — в несогласованном."""
        u = np.clip(u, 0.0, 1.0)
        if self.cfg.world_mode == "matched":
            return u ** self.cfg.pre_gamma
        return (u >= self.cfg.pre_u0).astype(float)

    def draw_W(self, g, mean: float) -> float:
        c = self.cfg
        if c.world_mode == "matched":
            return float(g.gamma(c.pre_k_W, mean / c.pre_k_W))
        s = c.pre_sigma_W
        return float(np.exp(g.normal(np.log(mean) - s * s / 2, s)))

    def stream(self, *labels):
        return R.stream(self.cfg.seed, R.CRISIS, *labels)

    def alive_base(self, idx, t):
        idx = np.asarray(idx, dtype=int)
        return idx[(self.a0[idx] <= t) & (self.a1[idx] > t)]

    def _add(self, typ, ctype, cluster, key, tau, W, predictable, false_alarm, targets, parent=-1, w0=None):
        ep = dict(id=len(self.episodes), type=typ, ctype=ctype, cluster=cluster, key=key, tau=float(tau),
                  W=float(W), w0=float(tau - W if w0 is None else w0), predictable=bool(predictable),
                  false_alarm=bool(false_alarm), parent=parent, targets=np.asarray(targets, dtype=int),
                  end=float(tau), n_hit=0, destroyed=False, n_xi=0)
        self.episodes.append(ep)
        return ep

    # ------------------------------------------------------------- построение
    def build(self):
        c, nd = self.cfg, self.w.nodes
        self._versions()
        by_dc = {k: np.asarray(v) for k, v in nd.groupby("dc").groups.items()}
        by_region = {k: np.asarray(v) for k, v in nd.groupby("region").groups.items()}
        by_batch = {k: np.asarray(v) for k, v in nd.groupby(["dc", "hw_true"]).groups.items()}

        for dc in range(c.n_dc):
            self._schedule("storm", "dc", dc, dc, by_dc.get(dc, []))
        for (dc, hw), idx in sorted(by_batch.items()):
            self._schedule("wear", "batch", f"{dc}/{c.hw_classes[hw]}", dc * 10 + int(hw), idx)
        for r in range(c.n_regions):
            self._schedule("exodus", "region", r, r, by_region.get(r, []))
        self._schedule_software()
        for dc in range(c.n_dc):
            self._schedule("fire", "dc", dc, dc, by_dc.get(dc, []))
        self._cascades(by_dc)

        for ep in self.episodes:
            if ep["predictable"] or ep["false_alarm"]:
                self._precursor(ep, by_region)
            if not ep["false_alarm"]:
                self._outcome_lives(ep, by_region)
        return self

    def _schedule(self, typ, ctype, cluster, ckey, idx):
        """Кризисы типа typ в одном кластере (пуассоновский поток) и ложные тревоги (отдельный поток)."""
        p = type_params(self.cfg, typ)
        rate = p["rate"] / YEAR
        pi, q, Wm = p.get("pi", 0.0), p.get("q", 0.0), p.get("W_days")
        g = self.stream(SCHED, T_ID[typ], ckey)
        taus = np.sort(g.uniform(0, self.H, g.poisson(rate * self.H)))
        for k, tau in enumerate(taus):
            P = g.random() < pi
            W = self.draw_W(g, Wm) if Wm else 0.0
            self._add(typ, ctype, cluster, (T_ID[typ], ckey, k, 0), tau, W, P, False, idx)
        if pi > 0 and q > 0:
            gf = self.stream(FALSE, T_ID[typ], ckey)
            taus = np.sort(gf.uniform(0, self.H, gf.poisson(q * pi * rate * self.H)))
            for k, tau in enumerate(taus):
                W = self.draw_W(gf, Wm)
                self._add(typ, ctype, cluster, (T_ID[typ], ckey, k, 1), tau, W, True, True, idx)

    def _schedule_software(self):
        """Неудачное обновление: срабатывает вскоре после выпуска версии (задержка ~ Exp со средним trigger_days)
        и бьёт только по узлам, уже обновившимся к этому моменту. Вероятность для одной версии —
        rate * (интервал выпуска) / год, чтобы средняя частота в год осталась rate."""
        c = self.cfg
        p = type_params(c, "software")
        p_bad = min(1.0, p["rate"] * c.sw_release_every_days / YEAR)
        for v in range(1, len(self.release)):
            g = self.stream(SCHED, T_ID["software"], v)
            # все числа тянутся всегда, чтобы версии не сдвигали друг друга
            r, delay, P, W = g.random(), g.exponential(p["trigger_days"]), g.random() < p["pi"], self.draw_W(g, p["W_days"])
            rf, delay_f, Wf = g.random(), g.exponential(p["trigger_days"]), self.draw_W(g, p["W_days"])
            for is_fa, hit, dl, Wx, Px in ((False, r < p_bad, delay, W, P),
                                           (True, rf < p["q"] * p["pi"] * p_bad, delay_f, Wf, True)):
                tau = self.release[v] + dl
                if hit and tau < self.H:
                    tg = self.alive_base(np.flatnonzero(self.version_at(tau) == v), tau)
                    self._add("software", "version", v, (T_ID["software"], v, 0, int(is_fa)), tau, Wx, Px, is_fa, tg,
                              w0=max(tau - Wx, self.release[v]))

    def _cascades(self, by_dc):
        """Каскад: после бури или пожара в дата-центре — вторичная буря в других дата-центрах провайдера."""
        c = self.cfg
        p = type_params(c, "cascade")
        prov = self.w.dc_provider
        primaries = [e for e in self.episodes if e["type"] in ("storm", "fire") and not e["false_alarm"]]
        for ep in primaries:
            dc = ep["cluster"]
            g = self.stream(CASC, *ep["key"])
            for sib in range(c.n_dc):
                if sib == dc or prov[sib] != prov[dc]:
                    continue
                r, delay = g.random(), g.exponential(p["delay_hours"] * HOUR)
                tau = ep["tau"] + delay
                if r < p["p"] and tau < self.H:
                    self._add("cascade", "dc", sib, (T_ID["cascade"],) + ep["key"] + (sib,), tau, delay,
                              True, False, by_dc.get(sib, []), parent=ep["id"], w0=ep["tau"])

    # ------------------------------------------------------------- версии ПО
    def _versions(self):
        c = self.cfg
        g = self.stream(VER, 0)
        n = g.poisson(self.H / c.sw_release_every_days)
        self.release = np.r_[-np.inf, np.sort(g.uniform(0, self.H, n))]        # версия 0 — у всех с начала
        upd = np.full((self.N, n + 1), -np.inf)
        for i in range(self.N):
            delay = self.stream(VER, 1, i).exponential(c.sw_update_delay_days, n)
            t = self.release[1:] + delay
            # узел, пришедший после выпуска версии, ставит её при регистрации
            upd[i, 1:] = np.where(self.release[1:] <= self.a0[i], self.a0[i], t)
        self.upd = upd
        rows = [(i, v, upd[i, v]) for i in range(self.N) for v in range(1, n + 1)
                if self.a0[i] <= upd[i, v] < self.a1[i]]
        self.node_updates = pd.DataFrame(rows, columns=["node", "version", "t"])
        self.versions = pd.DataFrame({"version": np.arange(n + 1), "t": self.release})

    def version_at(self, t) -> np.ndarray:
        """Версия клиента каждого узла в момент t."""
        ok = self.upd <= t
        return np.where(ok, np.arange(self.upd.shape[1]), -1).max(axis=1)

    # ------------------------------------------------------------- предвестники
    def _precursor(self, ep, by_region):
        c, w = self.cfg, self.w
        typ, key, tau, W = ep["type"], ep["key"], ep["tau"], ep["W"]
        p = type_params(c, typ)
        rho, s = p["rho"], p["s"]
        w0 = ep["w0"]
        u = lambda t: (t - w0) / max(tau - w0, 1e-12)
        targets = ep["targets"]
        xi = self.stream(PRE, *key, CH_XI).random(len(targets)) < s
        ep["n_xi"] = int(xi.sum())
        nd = w.nodes
        for i in targets[xi]:
            lo = max(w0, self.a0[i])
            if typ == "software":
                lo = max(lo, self.upd[i, ep["cluster"]])         # предвестник — только у обновившихся
            hi = min(tau, self.a1[i])
            if hi <= lo:
                continue
            for ch in CHANNELS[typ]:
                if ch == "micro":
                    lam = float(nd["lam_micro"].iat[i]) * (rho - 1.0)
                    g = self.stream(PRE, *key, int(i), CH_MICRO)
                    ts = w._thin(g, lambda t: lam * self.g_shape(u(t)), lo, hi, lam)
                    d, mean = c.d, c.micro_mean_minutes / 1440.0
                    dur = -mean * np.log(1 - g.random(len(ts)) * (1 - np.exp(-d / mean)))
                    self.pre_rows += [(int(i), t, t + l, "micro", ep["id"]) for t, l in zip(ts, dur)]
                elif ch == "short":
                    grid = np.linspace(lo, hi, 50)
                    lam_max = float(w.rate(np.full(50, i), grid).max()) * 1.05 * (rho - 1.0)
                    g = self.stream(PRE, *key, int(i), CH_SHORT)
                    rate_fn = lambda t: w.rate(np.full(len(t), i), t) * (rho - 1.0) * self.g_shape(u(t))
                    ts = w._thin(g, rate_fn, lo, hi, lam_max)
                    dur = self._short_durations(g, int(i), len(ts))
                    self.pre_rows += [(int(i), t, t + l, "short", ep["id"]) for t, l in zip(ts, dur)]
                elif ch == "silent":
                    lam = float(nd["lam_silent"].iat[i]) * (rho - 1.0)
                    g = self.stream(PRE, *key, int(i), CH_SILENT)
                    ts = w._thin(g, lambda t: lam * self.g_shape(u(t)), lo, hi, lam)
                    self.silent_rows += [(int(i), t, ep["id"]) for t in ts]
                elif ch == "audit":
                    self.audit_rows.append((int(i), lo, hi, w0, tau, rho, ep["id"]))
                elif ch == "churn":
                    j0 = self.join[i]
                    lam_max = float(w.churn_rate(lo - j0)) * (rho - 1.0) * 1.0001
                    g = self.stream(PRE, *key, int(i), CH_CHURN)
                    ts = w._thin(g, lambda t: w.churn_rate(t - j0) * (rho - 1.0) * self.g_shape(u(t)),
                                 lo, hi, lam_max, first_only=True)
                    if len(ts):
                        self.forced_rows.append((int(i), float(ts[0]), "pre_churn", ep["id"]))
        if "join" in CHANNELS[typ]:
            # регистрации в регионе падают до доли rho_join: часть новых узлов не приходит
            region = by_region.get(ep["cluster"], np.empty(0, int))
            r = self.stream(PRE, *key, CH_JOIN).random(len(region))
            for i, ri in zip(region, r):
                j = self.join[i]
                if max(w0, 0.0) <= j < tau and ri < (1.0 - p["rho_join"]) * float(self.g_shape(u(j))):
                    self.cancel_rows.append((int(i), float(j), ep["id"]))

    def _short_durations(self, g, i, m):
        """Короткие простои предвестника: длиннее франшизы d, короче порога T_rep."""
        out = np.empty(0)
        while len(out) < m:
            L = self.w.sample_duration(g, i, m)
            out = np.r_[out, L[L < self.cfg.T_rep]]
        return out[:m]

    # ------------------------------------------------------------- исходы
    def _outcome_lives(self, ep, by_region):
        """Исходы, меняющие жизнь узлов: гибель партии при износе, волна ухода при массовом уходе."""
        c, w = self.cfg, self.w
        typ, key, tau = ep["type"], ep["key"], ep["tau"]
        p = type_params(c, typ)
        if typ == "wear":
            T = p["period_days"]
            g = self.stream(OUT, *key)
            tg = ep["targets"]
            r, tdie = g.random(len(tg)), tau + g.uniform(0, T, len(tg))
            alive = (self.a0[tg] <= tau) & (self.a1[tg] > tau)
            for i, t in zip(tg[(r < p["p_death"]) & alive], tdie[(r < p["p_death"]) & alive]):
                self.forced_rows.append((int(i), float(t), "wear_death", ep["id"]))
            ep["end"] = tau + T
        elif typ == "exodus":
            T = p["period_days"]
            mult = p["multiplier"]
            region = ep["targets"]
            for i in region:
                lo, hi = max(tau, self.a0[i]), min(tau + T, self.a1[i])
                if hi <= lo:
                    continue
                j0 = self.join[i]
                lam_max = float(w.churn_rate(lo - j0)) * (mult - 1.0) * 1.0001
                g = self.stream(OUT, *key, int(i))
                ts = w._thin(g, lambda t: w.churn_rate(t - j0) * (mult - 1.0), lo, hi, lam_max, first_only=True)
                if len(ts):
                    self.forced_rows.append((int(i), float(ts[0]), "exodus", ep["id"]))
            # во время кризиса регистрации в регионе подавлены до доли rho_join
            r = self.stream(OUT, *key, CH_JOIN).random(len(region))
            for i, ri in zip(region, r):
                if tau <= self.join[i] < tau + T and ri < 1.0 - p["rho_join"]:
                    self.cancel_rows.append((int(i), float(self.join[i]), ep["id"]))
            ep["end"] = tau + T

    def outage_rows(self, rows: list):
        """Простои от исходов кризисов (вызывается из world при итоговой жизни узлов)."""
        c, w = self.cfg, self.w
        for ep in self.episodes:
            typ = ep["type"]
            if ep["false_alarm"] or typ not in ("storm", "cascade", "fire", "software"):
                continue
            g = self.stream(OUT, *ep["key"])
            tau, tag, tg = ep["tau"], OUTAGE_TAG + ep["id"], ep["targets"]
            if typ in ("storm", "cascade") and c.world_mode == "mismatched":
                Dl = c.storm_hours * HOUR
                ep["n_hit"] = w.storm_rows(rows, g, tg, tau, Dl, c.storm_multiplier, "crisis", tag)
                ep["end"] = tau + Dl
                continue
            if typ in ("storm", "cascade"):
                L, destroyed = w.shock_draw(g, c.shock_scale_hours[0], c.shock_total_share)
            elif typ == "fire":
                L, destroyed = w.shock_draw(g, c.shock_scale_hours[0], 1.0)
            else:
                p = type_params(c, "software")
                scale = p["L_mean_hours"] / gamma_fn(1.0 + 1.0 / c.shock_shape)
                L, destroyed = w.shock_draw(g, scale, p["total_share"])
            ep["n_hit"] = w.shock_rows(rows, tg, tau, L, destroyed, "crisis", tag)
            ep["end"], ep["destroyed"] = tau + L, destroyed

    # ------------------------------------------------------------- таблицы истины
    def table(self) -> pd.DataFrame:
        cols = ["id", "type", "ctype", "cluster", "tau", "W", "w0", "predictable", "false_alarm",
                "parent", "end", "n_targets", "n_hit", "destroyed", "n_xi"]
        rows = [{**{k: e[k] for k in cols if k in e}, "n_targets": len(e["targets"])} for e in self.episodes]
        return pd.DataFrame(rows, columns=cols)

    def frames(self) -> dict:
        return {
            "pre_events": pd.DataFrame(self.pre_rows, columns=["node", "start", "end", "channel", "episode"]),
            "pre_silent": pd.DataFrame(self.silent_rows, columns=["node", "t", "episode"]),
            "pre_audit": pd.DataFrame(self.audit_rows, columns=["node", "start", "end", "w0", "tau", "rho", "episode"]),
            "forced": pd.DataFrame(self.forced_rows, columns=["node", "t", "reason", "episode"]),
            "cancelled": pd.DataFrame(self.cancel_rows, columns=["node", "join_t", "episode"]),
        }
