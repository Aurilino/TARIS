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
        self.n_own = int(self.own.max()) + 1

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
        co = np.bincount(self.own[held], minlength=self.n_own)          # фрагменты объекта у каждого владельца
        cand = cand[~np.isin(cand, held)] if len(held) else cand
        w = self.w[cand].astype(float)
        dc_c, prov_c, reg_c, own_c = self.dc[cand], self.prov[cand], self.reg[cand], self.own[cand]
        out = []
        for _ in range(m):
            ok = (cd[dc_c] < dc_cap) & (cp[prov_c] < r_prov) & (cr[reg_c] < r_reg) & (co[own_c] < r_own)
            ww = w * ok
            tot = ww.sum()
            if tot <= 0:
                return []
            j = int(np.searchsorted(np.cumsum(ww), rng.random() * tot, side="right"))
            j = min(j, len(cand) - 1)
            c = int(cand[j]); out.append(c)
            cd[self.dc[c]] += 1; cp[self.prov[c]] += 1; cr[self.reg[c]] += 1
            co[self.own[c]] += 1
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

    def __init__(self, world, n: int = 15, gamma: float = 1.0, fast: bool = False):
        super().__init__(world, n, gamma)
        self.name = f"Оракул кризисов, n={n}" + (", тревоги каждый подшаг" if fast else "")
        self.fast = fast              # тревоги и перенос с шагом alarm_step_hours (ledger, Ч2)
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


class AgentCrisisPolicy(OraclePolicy):
    """Агент с прогнозом кризисов (шаг 5, Ч5): размещение как у оракула, тревоги — по счёту аномалий агента
    (forecast.Forecaster) только по данным сети. Пока тревога по группе узлов действует, новые фрагменты в группу
    не кладутся, а с её узлов фрагменты переносятся заранее — как у оракула кризисов, но без знания истины.

    Тревога поднимается, когда счёт группы выше порога своего типа, и держится hold суток после последнего
    превышения. Порог агент учит сам (П3): каждый всплеск счёта выше floor становится эпизодом; через
    горизонт разметки агент смотрит, случился ли в группе кризис (массовый простой дата-центра — для бури
    и каскада; не меньше двух узлов партии, ушедших в простой дольше суток, — для износа; уходы в регионе
    втрое выше фона — для массового ухода). Порог — максимум (верные - kappa * ложные) по эпизодам с пиком
    не ниже порога; пока верных эпизодов меньше min_pos — начальный порог c_init. Для сбоя ПО (эпизодов
    мало) порог постоянный — c_init.
    fast=True — тревоги и перенос каждый подшаг (ledger, alarm_step_hours); forecast=False — та же политика
    без прогноза (нижняя граница для H8). За один вызов — не больше max_moves переносов; фрагмент, которому
    некуда переехать из-за лимитов на кластеры, до конца тревог больше не пробуется. Тревога держится не меньше
    шага решений; группу больше max_share узлов в реестре (например, версию ПО, на которую обновилась почти вся
    сеть) агент отмечает тревогой, но не избегает; если без групп в тревоге кандидатов меньше 3n, размещение и
    перенос идут по всем кандидатам. fixed_thr — постоянные пороги (без обучения): показывает, что даёт сам прогноз.

    kappa="econ" — экономический критерий (Э1): эпизод весит не 1 и kappa, а деньгами по ценам резерва
    (reserve.Costs: c_tr за единицу трафика, W за объект). В начале всплеска агент снимает по своей карте
    размещения, сколько объектов держат в группе f фрагментов. Перенос стоит C = c_tr * F / k (F — фрагменты в
    группе) и в верном, и в ложном эпизоде; верный эпизод ещё и приносит B = c_tr * R + W * dL — избегнутый трафик
    восстановления и избегнутые потери при доле q узлов группы, выбитых кризисом (q агент оценивает по своей
    разметке). Порог — максимум суммы (y * B - C) по эпизодам с пиком не ниже порога; «не тревожить» (порог inf)
    тоже допустим."""

    GRID = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)
    HOLD = {"storm": 0.25, "cascade": 0.25, "software": 0.25, "wear": 7.0, "exodus": 7.0}   # сутки
    LABEL = {"storm": 1.0, "cascade": 1.0, "software": 1.0, "wear": 30.0, "exodus": 14.0}  # горизонт разметки
    LEARN = {"storm": "dc", "cascade": "dc", "wear": "wear", "exodus": "exodus", "software": None}

    def __init__(self, world, n: int = 13, gamma: float = 1.0, fast: bool = False, forecast: bool = True,
                 c_init: float = 10.0, floor: float = 2.0, kappa: float = 0.5, min_pos: int = 3,
                 max_moves: int = 2000, fixed_thr: dict | None = None):
        from .experience import Experience
        from .forecast import Forecaster
        super().__init__(world, n, gamma)
        self.fast, self.forecast = fast, forecast
        hours = world.cfg.alarm_step_hours if fast else 24
        self.fc = Forecaster(world, Experience(world), step_hours=hours)
        self.c_init, self.floor, self.kappa, self.min_pos = c_init, floor, kappa, min_pos
        self.thr = {k: c_init for k in ("dc", "wear", "exodus", "software")}
        self.learn = fixed_thr is None          # fixed_thr — постоянные пороги по группам, без обучения
        if fixed_thr:
            self.thr.update(fixed_thr)
        self.active = {}          # (тип, ctype, кластер) -> [до какого момента, узлы]
        self.runs = {}            # группа -> открытый всплеск: [начало, пик, последнее превышение floor, узлы, тип]
        self.pending = []         # закрытые всплески, ждущие разметки
        self.labels = {k: [] for k in ("dc", "wear", "exodus")}      # (пик, был ли кризис)
        self.thr_log = []
        self.max_moves = max_moves  # переносов за один вызов (пропускная способность сети)
        self.step = hours / 24.0    # шаг решений о тревогах, сутки: тревога держится не меньше шага
        self.max_share = 0.25       # группу больше этой доли узлов в реестре агент не избегает (только тревога)
        self.nomove = set()         # фрагменты, которым при текущих тревогах некуда переехать
        self.econ = kappa == "econ"
        if self.econ:
            from .reserve import Costs
            from .config import HOUR
            self.costs = Costs()
            self.w_loss = world.cfg.T_rep_hours * HOUR + world.cfg.repair_hours * HOUR   # окно до конца восстановления
            self.q_hit = {k: [0.0, 0.0] for k in self.labels}     # выбито узлов и всего узлов в верных эпизодах
            self.econ_log = []                                    # (t, группа, порог, kappa_экв, B, C)
            self._coef_memo, self._econ_mark = {}, {}
        self.name = ("Агент кризисов" if forecast else "Оракул без прогноза") + f", n={n}" + \
            (", шаг тревог 1 ч" if fast and hours < 24 else "") + (", экономический порог" if self.econ else "")

    # ---- разметка своих всплесков (П3)
    def _crisis_seen(self, lab, nodes, t0, t1):
        fc = self.fc
        d0, d1 = int(max(t0, 0)), int(min(t1, fc.T - 1))
        if lab == "dc":
            return bool(fc.mass[nodes, d0:d1 + 1].any())
        c0, c1 = int(t0 * fc.cpd), int(min(t1, fc.T) * fc.cpd)
        n_long = float(fc.longst[nodes, c0:c1].sum())
        if lab == "wear":
            return n_long >= 2
        cut = max(int(t0) - 30, 1)
        rate = (fc.cum["longst"][nodes, min(cut, fc.cum_day)].sum() + 0.5) / max(fc.cum["up"][nodes, min(cut, fc.cum_day)].sum() / fc.cpd, 1.0)
        return n_long >= max(3.0, 3.0 * rate * len(nodes) * (t1 - t0))

    def _hit_share(self, lab, nodes, t0, t1):
        """Сколько узлов группы кризис выбил надолго (по данным сети): для дата-центра — узлы массового простоя,
        для партии и региона — узлы, чей простой стал длиннее суток."""
        fc = self.fc
        if lab == "dc":
            d0, d1 = int(max(t0, 0)), int(min(t1, fc.T - 1))
            return int(fc.mass[nodes, d0:d1 + 1].any(1).sum())
        c0, c1 = int(t0 * fc.cpd), int(min(t1, fc.T) * fc.cpd)
        return int((fc.longst[nodes, c0:c1].sum(1) > 0).sum())

    def _snapshot(self, view, g):
        """Число объектов по числу f своих фрагментов в группе g (по карте размещения): {f: объектов}."""
        cnt = {}
        for node in g:
            for obj, _ in view.on_node(int(node)):
                cnt[obj] = cnt.get(obj, 0) + 1
        out = {}
        for f in cnt.values():
            out[f] = out.get(f, 0) + 1
        return out

    def _p_loss(self, m, p):
        """Вероятность потерять объект с m фрагментами за окно восстановления: больше m - k гибелей из m."""
        from scipy.stats import binom
        return float(binom.sf(m - self.k, m, p)) if m >= self.k else 1.0

    def _coef(self, lab, f):
        """Выгода b_f и цена c_f на один объект с f фрагментами в группе (Э1); кэш по (q, p) текущей оценки."""
        from scipy.stats import binom
        c, k, n, fc = self.costs, self.k, self.n, self.fc
        m, N = self.q_hit[lab]
        q = (m + 1.0) / (N + 2.0)
        day = min(fc.cum_day, fc.T)
        lam = (fc.cum["longst"][:, day].sum() + 0.5) / max(fc.cum["up"][:, day].sum() / fc.cpd, 1.0)
        p = 1.0 - np.exp(-lam * self.w_loss)              # фрагмент гибнет за окно восстановления (фон)
        key = (lab, q, p, f)
        if key not in self._coef_memo:
            base = self._p_loss(n, p)
            R = c.size * ((1.0 - (1.0 - q) ** f) + q * f / k)
            dL = sum(binom.pmf(r, f, q) * self._p_loss(n - r, p) for r in range(f + 1)) - base
            self._coef_memo[key] = (c.c_tr * R + c.W * dL, c.c_tr * c.size * f / k)
        return self._coef_memo[key]

    def _value(self, lab, snap):
        """Выгода верной тревоги B и цена переноса C для эпизода со снимком snap (Э1)."""
        B = C = 0.0
        for f, cnt in snap.items():
            b, cc = self._coef(lab, f)
            B += cnt * b; C += cnt * cc
        return B, C

    def _learn(self, t):
        keep = []
        for r in self.pending:
            start, peak, last, nodes, typ, lab = r[:6]
            due = last + self.LABEL[typ]
            if t < due:
                keep.append(r)
                continue
            y = self._crisis_seen(lab, nodes, start, due)
            if self.econ:
                if y:
                    self.q_hit[lab][0] += self._hit_share(lab, nodes, start, due)
                    self.q_hit[lab][1] += len(nodes)
                self.labels[lab].append((peak, y, r[6]))
            else:
                self.labels[lab].append((peak, y))
        self.pending = keep
        for lab, L in self.labels.items():
            if not self.learn or sum(v[1] for v in L) < self.min_pos:
                continue
            pk = np.array([v[0] for v in L]); y = np.array([v[1] for v in L], bool)
            if self.econ:
                mark = (len(L), int(t))
                if self._econ_mark.get(lab) == mark:     # пересчёт — при новых метках или раз в сутки (фон p)
                    continue
                self._econ_mark[lab] = mark
                self._coef_memo = {}
                BC = np.array([self._value(lab, v[2]) for v in L])
                gv = y * BC[:, 0] - BC[:, 1]
                grid = self.GRID + (np.inf,)
                gain = np.array([gv[pk >= c].sum() for c in grid])
                c = grid[int(np.flatnonzero(gain >= gain.max() - 1e-12)[-1])]   # при равенстве — больший порог
                Bm, Cm = float(BC[y, 0].mean()), float(BC[:, 1].mean())
                kap = Cm / (Bm - Cm) if Bm > Cm else np.inf
                if not self.econ_log or self.econ_log[-1][1:3] != (lab, c) or t - self.econ_log[-1][0] >= 30:
                    self.econ_log.append((t, lab, c, kap, Bm, Cm))
            else:
                gain = [(y & (pk >= c)).sum() - self.kappa * (~y & (pk >= c)).sum() for c in self.GRID]
                c = self.GRID[int(np.argmax(gain))]
            if c != self.thr[lab]:
                self.thr[lab] = c
                self.thr_log.append((t, lab, c, len(L), int(y.sum())))

    # ---- интерфейс политики
    def _avoid(self, t):
        nodes = [v[1] for v in self.active.values() if v[0] > t and v[2]]
        return np.unique(np.concatenate(nodes)) if nodes else np.empty(0, int)

    def _cand(self, t, view):
        cand = view.candidates()
        avoid = self._avoid(t)
        if not len(avoid):
            return cand
        left = cand[~np.isin(cand, avoid)]
        # без группы в тревоге узлов не хватает на разнесённое размещение — тревога не должна губить объекты
        return left if len(left) >= 3 * self.n else cand

    def alarm(self, t, view):
        if not self.forecast:
            return []
        self.fc.update(t, view)
        self._learn(t)
        for typ, ct, cl, S, O, E, g in self.fc.scores():
            lab = self.LEARN[typ] or "software"
            gk = (lab, ct, str(cl))
            run = self.runs.get(gk)
            if S > self.floor:
                if run is None:
                    self.runs[gk] = [t, S, t, g, typ, lab]
                    if self.econ and lab != "software":
                        self.runs[gk].append(self._snapshot(view, g))     # до переносов этого вызова
                else:
                    run[1] = max(run[1], S); run[2] = t; run[4] = typ
            elif run is not None and t - run[2] > self.HOLD[typ]:
                if lab != "software":
                    self.pending.append(tuple(run))
                del self.runs[gk]
            if S > self.thr[lab]:
                n_reg = max(int((view.join_times() <= t).sum()), 1)
                act = len(g) <= self.max_share * n_reg       # слишком большую группу не избегать
                self.active[(typ, ct, cl)] = [t + max(self.HOLD[typ], self.step), g, act]
        self.active = {k: v for k, v in self.active.items() if v[0] > t}
        return [dict(type=k[0], ctype=k[1], cluster=k[2], prob=1.0, horizon=v[0] - t) for k, v in self.active.items()]

    def migrate(self, t, view):
        if not self.active:
            self.nomove = set()
            return []
        cand = self._cand(t, view)
        out = []
        for node in self._avoid(t):
            for obj, idx in view.on_node(int(node)):
                if len(out) >= self.max_moves:
                    return out
                if (obj, idx) in self.nomove or idx in view.migrating(obj):
                    continue
                held = view.holders(obj) | {to for o, _, to in out if o == obj}
                to = self._choose_spread(self._g, cand, 1, held=held - {int(node)})
                if to:
                    out.append((obj, idx, to[0]))
                else:
                    self.nomove.add((obj, idx))      # места нет (лимиты на кластеры) — до конца тревог не пробовать
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
    Абляции TARIS-резерва (H7): corr=False — запас по карте, без IBNR и RBNS; time_varying=False — интенсивности
    на горизонте постоянны (берутся на момент расчёта порога, без старения узлов).
    """

    def __init__(self, world, n: int = 15, mode: str = "thiele", eps: float = 1e-3, H: int = 365,
                 R_fixed: int | None = None, gamma: float = 1.0, block: int = 30, corr: bool = True,
                 time_varying: bool = True, rates: str = "true", claimed: bool = True):
        from . import reserve as RS
        super().__init__(world, n, gamma)
        self.RS, self.mode, self.eps, self.H, self.block = RS, mode, eps, H, block
        self.corr, self.time_varying = corr and mode == "thiele", time_varying
        self.agent = rates == "agent"
        if self.agent:
            # оценки агента (шаг 5, пункт 5.1): обучение раз в block эпох; до первого обучения — немедленное восстановление
            from .experience import Experience
            from .risk import RiskModel, EstimatedRates
            self.exp = Experience(world, block)
            self.model = RiskModel(self.exp, claimed=claimed)
            self.rates = EstimatedRates(world, self.model)
            self.fits = []
        else:
            self.rates = RS.TrueRates(world)
        self.R_fixed = n - self.k - 1 if R_fixed is None else R_fixed
        self.costs = RS.Costs()
        self.thr = {}          # объект -> (ключ набора узлов, порог)
        self.memo = {}         # округлённые интенсивности -> порог
        self.name = {"thiele": f"Резерв (Тиле), n={n}", "markov": f"Марковский порог, n={n}",
                     "fixed": f"Порог R_min={self.R_fixed}, n={n}"}[mode]
        if mode == "thiele" and not corr:
            self.name += ", без IBNR/RBNS"
        if mode == "thiele" and not time_varying:
            self.name += ", постоянные интенсивности"
        if self.agent:
            self.name += ", оценки агента" + ("" if claimed else " (только проверяемые признаки)")

    def _threshold(self, obj, nodes, t, view):
        if self.mode == "fixed":
            return self.R_fixed
        if self.agent and not self.model.fitted:
            return self.n - self.k - 1
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
        if self.mode == "markov" or not self.time_varying:
            lam = np.full(nb, self.rates.fragment_loss(nodes, t).mean())
        else:
            lam = np.array([self.rates.fragment_loss(nodes, t + (b + 0.5) * self.block).mean() for b in range(nb)])
        # кэш порогов: интенсивности округляются ступенями 5% (логарифмическая сетка), задержка — до 0,05 суток
        q = lambda x: float(np.exp(np.round(np.log(max(x, 1e-12)) * 20) / 20))
        lam = np.array([q(x) for x in lam])
        shocks = sorted((q(e), m) for e, m in shocks)
        mu = 1.0 / (round(20.0 / mu) / 20.0)
        mkey = (self.mode, tuple(lam), tuple(shocks), mu)       # порог зависит только от этих величин
        if mkey not in self.memo:
            # P_loss убывает с ростом порога (восстановление раньше): допустимые пороги — от некоторого R и выше.
            # Перебор сверху вниз до первого недопустимого проверяет все допустимые — выбор тот же, что при полном.
            best, R_best = None, n - k - 1
            for R_min in range(n - k - 1, -1, -1):
                m = RS.object_metrics(n, k, lam, shocks, R_min, mu, self.costs, dt=self.block)
                if m["P_loss"] > self.eps:
                    break
                v = m["cost"] if self.mode == "thiele" else m["traffic"]
                if best is None or v < best:
                    best, R_best = v, R_min
            self.memo[mkey] = R_best
        R = self.memo[mkey]
        self.thr[obj] = (key, R)
        return R

    def on_epoch(self, t, view):
        self._g = self._rng(view)
        if self.agent:
            self.exp.update(t, view)
            if t > 0 and int(t) % self.block == 0:
                self.model.fit(t)
                if self.model.fitted:
                    self.model.refresh(t)
                    self.fits.append((t, self.model.a_hat, self.model.pearson, self.model.churn_share))
        cand = self._cand(t, view)
        reqs = []
        need = view.needs_repair()
        if self.corr and need and (not self.agent or self.model.fitted):
            last_ok, down = view.last_audit_ok(), view.down_since()
        for obj in need:
            holders = view.holders(obj)
            if len(holders) < self.k:
                continue
            thr = self._threshold(obj, holders, t, view)
            if self.corr and (not self.agent or self.model.fitted):
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
