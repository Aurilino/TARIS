"""Интерфейс политики и две граничные политики (шаг 3; шаг 4, пункт 4.1).

UniformPolicy — равномерное размещение со статическим кодом (нижняя граница).
OraclePolicy — знает истинный риск узлов и скрытые кластеры (верхняя граница).
Политика видит мир только через NetworkView; оракул дополнительно получает объект World.
"""
from __future__ import annotations

import numpy as np


class Policy:
    name = "base"

    def place(self, obj: int, t: float, view) -> list[int]:
        """Узлы для фрагментов нового объекта (их число = n_j)."""
        raise NotImplementedError

    def on_epoch(self, t: float, view) -> list[tuple[int, list[int], list[int]]]:
        """Решения о восстановлении: (объект, номера фрагментов, узлы-получатели)."""
        return []

    def migrate(self, t: float, view) -> list[tuple[int, int, int]]:
        """Переносы заранее (пункт 3.9): (объект, номер фрагмента, узел-получатель)."""
        return []

    def alarm(self, t: float, view) -> list[dict]:
        """Тревоги о кризисах (пункт 3.9): словари с ключами type (тип кризиса), ctype и cluster
        (кластер-цель в тех же обозначениях, что world.crises: dc — номер, region — номер,
        batch — «dc/класс», version — номер версии), prob (вероятность) и horizon (горизонт, сутки)."""
        return []

    def predict(self, t: float, view):
        return None

    def explain(self, obj: int, t: float, view):
        return None

    @staticmethod
    def _rng(view):
        return np.random.default_rng(int.from_bytes(view.seed[:8], "big"))


class UniformPolicy(Policy):
    """Равномерный выбор среди доступных узлов, статический код (n, k), немедленное восстановление."""

    def __init__(self, n: int = 15):
        self.n = n
        self.name = f"Равномерное, n={n}"

    @staticmethod
    def _choose(rng, cand, m, exclude=()):
        """m различных узлов из cand, исключая exclude (быстрый выбор с отбраковкой)."""
        out, seen = [], set(exclude)
        if len(cand) - len(seen) < m:
            return []
        while len(out) < m:
            for j in rng.integers(0, len(cand), 2 * (m - len(out)) + 2):
                c = int(cand[j])
                if c not in seen:
                    seen.add(c); out.append(c)
                    if len(out) == m:
                        break
        return out

    def place(self, obj, t, view):
        return self._choose(self._rng(view) if not hasattr(self, "_g") else self._g, view.candidates(), self.n)

    def on_epoch(self, t, view):
        self._g = self._rng(view)
        cand = view.candidates()
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose(self._g, cand, len(idx), exclude=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class OraclePolicy(UniformPolicy):
    """Знает истинную интенсивность полной гибели узлов и скрытые кластеры.
    Вес узла обратно пропорционален истинному риску; в одном дата-центре — не больше одного фрагмента объекта,
    у одного провайдера, в одном регионе и у одного владельца — не больше min(n-k, k-1)."""

    def __init__(self, world, n: int = 15, gamma: float = 1.0):
        super().__init__(n)
        self.name = f"Оракул, n={n}"
        nd = world.nodes
        phi = np.array([world.true_phi(i) for i in range(len(nd))])
        q = nd["lam_avg"].to_numpy() * phi
        # устройства пользователей (3.10) ещё и уходят чаще: добавляется избыточный уход старожилов (у узлов в ДЦ — 0)
        q = q + (nd["m_churn"].to_numpy() - 1.0) * world.cfg.churn_base
        self.w = (1.0 / np.maximum(q, 1e-9)) ** gamma
        self.dc, self.prov, self.reg, self.own = (nd[c].to_numpy() for c in ("dc", "provider", "region", "owner"))
        # у устройства нет дата-центра: оно — свой отдельный домен отказа (номера после дата-центров)
        dev = self.dc < 0
        self.dc = np.where(dev, world.cfg.n_dc + np.arange(len(nd)), self.dc)
        self.k = world.cfg.k
        self.n_dc, self.n_prov, self.n_reg = int(self.dc.max()) + 1, int(self.prov.max()) + 1, world.cfg.n_regions

    def _choose_spread(self, rng, cand, m, held=()):
        """Выбор с разнесением: в дата-центре — один фрагмент объекта, у провайдера, в регионе и у владельца —
        не больше r = min(n - k, k - 1). Если узлов не набрать (например, кластер-цель кризиса исключён),
        лимиты ослабляются по одному, начиная с самого безобидного: сначала регион (+1, +2, ...), потом
        дата-центр (2, 3, ... фрагмента), в крайнем случае — провайдер. Дата-центр — меньший домен отказа, чем
        провайдер: при немногих живых провайдерах лишний фрагмент лучше положить вторым в дата-центр, чем
        четвёртым к провайдеру, — простой провайдера дольше T_rep иначе губит объекты разом.
        Лимит на владельца не ослабляется (общий владелец — это риск раскрытия)."""
        r0, n = min(self.n - self.k, self.k - 1), self.n
        steps = ([(r0, r0 + rr, 1) for rr in range(n + 1)]                 # регион
                 + [(r0, r0 + n, dc) for dc in range(2, n + 1)]            # дата-центр
                 + [(r0 + rp, r0 + n, n) for rp in range(1, n + 1)])       # провайдер
        for r_prov, r_reg, dc_cap in steps:
            out = self._choose_spread_r(rng, cand, m, held, r_prov, r_reg, r0, dc_cap)
            if out:
                return out
        return []

    def _choose_spread_r(self, rng, cand, m, held, r_prov, r_reg, r_own, dc_cap=1):
        held = np.fromiter(held, int) if len(held) else np.empty(0, int)
        cd = np.bincount(self.dc[held], minlength=self.n_dc)
        cp = np.bincount(self.prov[held], minlength=self.n_prov)
        cr = np.bincount(self.reg[held], minlength=self.n_reg)
        co = {}
        for h in held:
            co[self.own[h]] = co.get(self.own[h], 0) + 1
        cand = cand[~np.isin(cand, held)] if len(held) else cand
        w = self.w[cand].astype(float)
        out = []
        for _ in range(m):
            ok = (cd[self.dc[cand]] < dc_cap) & (cp[self.prov[cand]] < r_prov) & (cr[self.reg[cand]] < r_reg)
            if co:
                ok &= np.array([co.get(o, 0) < r_own for o in self.own[cand]]) if len(co) else ok
            ww = w * ok
            tot = ww.sum()
            if tot <= 0:
                return []
            j = int(np.searchsorted(np.cumsum(ww), rng.random() * tot, side="right"))
            j = min(j, len(cand) - 1)
            c = int(cand[j]); out.append(c)
            cd[self.dc[c]] += 1; cp[self.prov[c]] += 1; cr[self.reg[c]] += 1
            co[self.own[c]] = co.get(self.own[c], 0) + 1
            w[j] = 0.0
        return out

    def _cand(self, t, view):
        return view.candidates()

    def place(self, obj, t, view):
        return self._choose_spread(self._g if hasattr(self, "_g") else self._rng(view), self._cand(t, view), self.n)

    def on_epoch(self, t, view):
        self._g = self._rng(view)
        cand = self._cand(t, view)
        reqs = []
        for obj in view.needs_repair():
            idx = sorted(view.missing(obj))
            nodes = self._choose_spread(self._g, cand, len(idx), held=view.holders(obj))
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs


class CrisisOraclePolicy(OraclePolicy):
    """Оракул кризисов (пункт 3.9): оракул, который с начала окна предвестника знает тип, кластер-цель
    и момент каждого предсказуемого кризиса. Пока кризис известен и не закончился, он поднимает тревогу,
    не кладёт новые фрагменты на узлы-цели, а до момента кризиса переносит с них фрагменты.
    О внезапных кризисах не знает, ложные тревоги отличает от кризисов. Верхняя граница для H8."""

    def __init__(self, world, n: int = 15, gamma: float = 1.0):
        super().__init__(world, n, gamma)
        self.name = f"Оракул кризисов, n={n}"
        cr = world.crises
        self.known = cr[cr["predictable"] & ~cr["false_alarm"]].reset_index(drop=True)
        self.targets = {e["id"]: e["targets"] for e in world.crisis.episodes}

    def _active(self, t):
        k = self.known
        return k[(k["w0"] <= t) & (k["end"] > t)]

    def _avoid(self, t) -> np.ndarray:
        a = self._active(t)
        if a.empty:
            return np.empty(0, int)
        return np.unique(np.concatenate([self.targets[i] for i in a["id"]]))

    def _cand(self, t, view):
        cand = view.candidates()
        avoid = self._avoid(t)
        return cand[~np.isin(cand, avoid)] if len(avoid) else cand

    def alarm(self, t, view):
        a = self._active(t)
        a = a[a["tau"] > t]
        return [dict(type=r.type, ctype=r.ctype, cluster=r.cluster, prob=1.0, horizon=r.tau - t + 1e-6)
                for r in a.itertuples()]

    def migrate(self, t, view):
        a = self._active(t)
        a = a[a["tau"] > t]
        if a.empty:
            return []
        cand = self._cand(t, view)
        out = []
        for node in np.unique(np.concatenate([self.targets[i] for i in a["id"]])):
            for obj, idx in view.on_node(int(node)):
                if idx in view.migrating(obj):
                    continue
                held = view.holders(obj) | {to for o, _, to in out if o == obj}
                to = self._choose_spread(self._g, cand, 1, held=held - {int(node)})
                if to:
                    out.append((obj, idx, to[0]))
        return out


class ReservePolicy(OraclePolicy):
    """Оракул с резервом информации (шаг 3, пункт 3.11): размещение как у оракула, восстановление — по порогу запаса.

    Восстановление объекта (всех признанных утраченными фрагментов разом) запускается, когда запас
    не выше порога R_min. Три варианта для H7:
    - mode="thiele" — TARIS-резерв: порог свой у каждого объекта, из уравнения Тиле по меняющимся во времени
      истинным интенсивностям его узлов (минимум приведённой стоимости при P_loss <= eps за горизонт H);
      запас — фактический, с поправками IBNR и RBNS; порог пересчитывается при смене набора узлов и раз в 30 суток;
    - mode="markov" — марковский конкурент: порог на объект при размещении по постоянной интенсивности
      (минимум трафика при P_loss <= eps), запас — по карте, без поправок;
    - mode="fixed" — эвристический порог R_fixed для всех объектов (R_fixed = n - k - 1 — немедленное восстановление).
    Если eps недостижим, берётся немедленное восстановление.
    """

    def __init__(self, world, n: int = 15, mode: str = "thiele", eps: float = 1e-3, H: int = 365,
                 R_fixed: int | None = None, gamma: float = 1.0, block: int = 30):
        from . import reserve as RS
        super().__init__(world, n, gamma)
        self.RS, self.mode, self.eps, self.H, self.block = RS, mode, eps, H, block
        self.rates = RS.TrueRates(world)
        self.R_fixed = n - self.k - 1 if R_fixed is None else R_fixed
        self.costs = RS.Costs()
        self.thr = {}          # объект -> (ключ набора узлов, порог)
        self.memo = {}         # округлённые интенсивности -> порог
        self.name = {"thiele": f"Резерв (Тиле), n={n}", "markov": f"Марковский порог, n={n}",
                     "fixed": f"Порог R_min={self.R_fixed}, n={n}"}[mode]

    def _threshold(self, obj, nodes, t, view):
        if self.mode == "fixed":
            return self.R_fixed
        T_end = view.L.T
        key = frozenset(nodes) if self.mode == "markov" else (frozenset(nodes), int(t // self.block))
        old = self.thr.get(obj)
        if old is not None and (self.mode == "markov" or old[0] == key):
            return old[1]
        RS, k, n = self.RS, self.k, self.n
        Hrem = int(min(self.H, max(T_end - t, self.block)))
        nb = max(1, int(np.ceil(Hrem / self.block)))
        nodes = np.asarray(sorted(nodes), int)
        cl = self.rates.shock_clusters(nodes, t)
        shocks = [(eta, int(m.sum())) for _, _, eta, m in cl]
        mu = RS.object_mu(self.rates, nodes, t, shocks)
        if self.mode == "markov":
            lam = np.full(nb, self.rates.fragment_loss(nodes, t).mean())
        else:
            lam = np.array([self.rates.fragment_loss(nodes, t + (b + 0.5) * self.block).mean() for b in range(nb)])
        # кэш порогов: интенсивности округляются ступенями 5% (логарифмическая сетка), задержка — до 0,05 суток
        q = lambda x: float(np.exp(np.round(np.log(max(x, 1e-12)) * 20) / 20))
        lam = np.array([q(x) for x in lam])
        shocks = sorted((q(e), m) for e, m in shocks)
        mu = 1.0 / (round(20.0 / mu) / 20.0)
        mkey = (self.mode, tuple(lam), tuple(shocks), mu)
        if mkey not in self.memo:
            best, R_best = None, n - k - 1
            for R_min in range(0, n - k):
                m = RS.object_metrics(n, k, lam, shocks, R_min, mu, self.costs, dt=self.block)
                if m["P_loss"] <= self.eps:
                    v = m["cost"] if self.mode == "thiele" else m["traffic"]
                    if best is None or v < best:
                        best, R_best = v, R_min
            self.memo[mkey] = R_best
        R = self.memo[mkey]
        self.thr[obj] = (key, R)
        return R

    def on_epoch(self, t, view):
        self._g = self._rng(view)
        cand = self._cand(t, view)
        reqs = []
        need = view.needs_repair()
        if self.mode == "thiele" and need:
            last_ok, down = view.last_audit_ok(), view.down_since()
        for obj in need:
            holders = view.holders(obj)
            if len(holders) < self.k:
                continue
            thr = self._threshold(obj, holders, t, view)
            if self.mode == "thiele":
                R_eff, _ = self.RS.effective_reserve(self.rates, view, obj, t, self.k, last_ok, down)
            else:
                R_eff = len(holders) - self.k
            if R_eff > thr:
                continue
            idx = sorted(view.missing(obj))
            nodes = self._choose_spread(self._g, cand, len(idx), held=holders)
            if nodes:
                reqs.append((obj, idx, nodes))
        return reqs
