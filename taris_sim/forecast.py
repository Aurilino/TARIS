"""Прогноз кризисов агентом TARIS (шаг 5, подшаги П1–П4, для H8).

Агент видит только сеть: начала простоев по пингам, проваленные аудиты, признанные тихие потери, реестр и
заявленные признаки узлов, версии клиента. Для групп узлов, в которых случаются кризисы пункта 3.9, он
считает, во сколько раз частота событий канала выше фона узлов, и пуассоновское отношение правдоподобия:

    S = O * ln(O / E) - (O - E)   (0 при O <= E),

где O — наблюдаемое число событий группы в окне, E — ожидаемое по фону.

Группы и каналы (П1):
- буря и каскад — дата-центр (заявленный), начала простоев за последние сутки; тревога «каскад», если в другом
  дата-центре того же провайдера за последние 2 суток был массовый простой (сеть видит первичный кризис);
- износ партии — дата-центр × класс оборудования (заявленные), проваленные аудиты и тихие потери за 7 суток;
- уязвимость ПО — узлы одной версии клиента, начала простоев за сутки;
- массовый уход — регион, простои без конца дольше суток, начавшиеся за последние 7 суток.
Фон узла — его собственная частота до начала окна исключения (сглаженная к средней по сети на c0 суток).

Порог тревоги своего типа (П3) агент учит сам по своей истории: кризис в группе он видит как массовый простой
(не меньше половины узлов группы начинают простой длиннее франшизы в одни сутки) или волну полных гибелей.
"""
from __future__ import annotations

import numpy as np

from .config import HOUR

TYPES = ("storm", "cascade", "wear", "software", "exodus")
WINDOW = {"storm": 1, "cascade": 1, "software": 1, "wear": 7, "exodus": 7}       # окно счёта, сутки
GAP = {"storm": 7, "cascade": 7, "software": 7, "wear": 60, "exodus": 30}        # исключение из фона перед окном


def lr_score(O, E):
    O, E = np.asarray(O, float), np.maximum(np.asarray(E, float), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(O > E, O * np.log(np.where(O > 0, O / E, 1.0)) - (O - E), 0.0)
    return s


class Forecaster:
    """Счёт аномалий по группам узлов на каждую эпоху; ведёт историю для обучения порогов."""

    def __init__(self, world, exp, c0: float = 30.0):
        c = world.cfg
        self.exp, self.reg = exp, exp.reg
        self.T, self.N = c.horizon_days, exp.N
        self.d = c.franchise_d_hours * HOUR
        self.hw_names = c.hw_classes
        self.c0 = c0
        self.starts = np.zeros((self.N, self.T))        # начала простоев по эпохам
        self.longst = np.zeros((self.N, self.T))        # начала простоев, которые длятся дольше суток (уходы)
        self.audf = np.zeros((self.N, self.T))          # проваленные аудиты и признанные тихие потери
        self.up = np.zeros((self.N, self.T))            # экспозиция: узел в реестре и не в долгом простое
        self.mass = np.zeros((self.N, self.T), bool)    # узел начал простой длиннее d в эпоху массового простоя группы
        self.version = np.full(self.N, -1)
        self.n_silent = 0
        r = self.reg
        self.groups = {}                                 # тип -> список (ctype, кластер, метка, массив узлов)
        dcs = np.unique(r.dc[r.dc >= 0])
        self.groups["storm"] = [("dc", int(d), int(d), np.flatnonzero(r.dc == d)) for d in dcs]
        self.groups["wear"] = [("batch", f"{int(d)}/{self.hw_names[h]}", (int(d), int(h)),
                                np.flatnonzero((r.dc == d) & (r.hw == h)))
                               for d in dcs for h in range(len(self.hw_names)) if ((r.dc == d) & (r.hw == h)).any()]
        self.groups["exodus"] = [("region", int(g), int(g), np.flatnonzero(r.region == g)) for g in np.unique(r.region)]
        self.dc_prov = {int(d): int(r.provider[r.dc == d][0]) for d in dcs}
        self.t = -1

    # ------------------------------------------------------------ накопление
    def update(self, t: float, view):
        """Забирает наблюдения эпохи e = t - 1 (view.step): в момент t известны события до t."""
        st = view.step
        if st is None:
            return
        e = int(st["epoch"])
        self.t = e
        reg = view.join_times() <= t
        on = np.zeros(self.N, bool)
        og = st["outages_ongoing"]
        if len(og):
            on[og["node"].to_numpy()] = (t - og["start"].to_numpy()) > 1.0
        self.up[:, e] = reg & ~on
        s = st["outages_started"]
        if len(s):
            np.add.at(self.starts[:, e], s["node"].to_numpy(), 1.0)
        a = st["audits"]
        if len(a):
            f = a[~a["passed_avail"]]
            np.add.at(self.audf[:, e], f["node"].to_numpy(), 1.0)
        sil = view.silent_declared()
        for _, n in sil[self.n_silent:]:
            self.audf[n, e] += 1.0
        self.n_silent = len(sil)
        # простои, идущие дольше суток: отмечаются в эпоху начала (видны через сутки)
        if len(og):
            el = t - og["start"].to_numpy()
            new_long = (el > 1.0) & (el <= 2.0)
            np.add.at(self.longst, (og["node"].to_numpy()[new_long], np.floor(og["start"].to_numpy()[new_long]).astype(int)), 1.0)
        self.version = view.node_versions()
        # массовый простой дата-центра в эту эпоху: не меньше половины узлов в сети начали простой длиннее d
        ended = st["outages_ended"]
        big = set(og["node"].to_numpy()[(t - og["start"].to_numpy()) > self.d].tolist()) if len(og) else set()
        if len(ended):
            big |= set(ended["node"].to_numpy()[ended["dur"].to_numpy() > self.d].tolist())
        started_now = set(s["node"].to_numpy().tolist()) if len(s) else set()
        hit = np.zeros(self.N, bool)
        hit[list(big & started_now)] = True
        for _, _, _, g in self.groups["storm"]:
            n_up = self.up[g, max(e - 1, 0)].sum()
            if n_up >= 2 and hit[g].sum() >= 0.5 * n_up:
                self.mass[g[hit[g]], e] = True

    # ------------------------------------------------------------ счёт
    def _bg(self, M, nodes, e, gap):
        """Фоновая частота канала у узлов nodes по истории до e - gap (сглаженная к средней по сети)."""
        h = max(e - gap, 0)
        if h == 0:
            return np.full(len(nodes), np.nan)
        cnt, ex = M[:, :h].sum(1), self.up[:, :h].sum(1)
        m = cnt.sum() / max(ex.sum(), 1e-9)
        return (cnt[nodes] + m * self.c0) / (ex[nodes] + self.c0)

    def scores(self) -> list:
        """Счёт по всем группам по данным до последней известной эпохи: (тип, ctype, кластер, S, O, E, узлы)."""
        e = self.t
        out = []
        if e < 30:
            return out
        r = self.reg
        # буря / каскад: дата-центр, начала простоев за сутки
        bg = self._bg(self.starts, np.arange(self.N), e, GAP["storm"])
        recent_mass = {}
        for ct, cl, key, g in self.groups["storm"]:
            if self.mass[g, max(e - 2, 0):e + 1].any():
                recent_mass[self.dc_prov[cl]] = recent_mass.get(self.dc_prov[cl], set()) | {cl}
        for ct, cl, key, g in self.groups["storm"]:
            # предвестник растёт к концу окна (пункт 3.9): лучший из счётов за 1, 2 и 3 последние эпохи
            best = (0.0, 0.0, 0.0)
            for h in (1, 2, 3):
                a = max(e - h + 1, 0)
                O = self.starts[g, a:e + 1].sum(); E = (bg[g][:, None] * self.up[g, a:e + 1]).sum()
                S = float(lr_score(O, E))
                if S > best[0] or h == 1:
                    best = max(best, (S, O, E))
            sibling = recent_mass.get(self.dc_prov[cl], set()) - {cl}
            out.append(("cascade" if sibling else "storm", ct, cl, best[0], best[1], best[2], g))
        # износ партии: аудиты и тихие потери за 7 суток
        h = WINDOW["wear"]
        bg = self._bg(self.audf, np.arange(self.N), e - h + 1, GAP["wear"])
        for ct, cl, key, g in self.groups["wear"]:
            O = self.audf[g, e - h + 1:e + 1].sum(); E = (bg[g][:, None] * self.up[g, e - h + 1:e + 1]).sum()
            out.append(("wear", ct, cl, float(lr_score(O, E)), O, E, g))
        # уязвимость ПО: версии клиента новее исходной
        bg = self._bg(self.starts, np.arange(self.N), e, GAP["software"])
        for v in np.unique(self.version[self.version > 0]):
            g = np.flatnonzero(self.version == v)
            O = self.starts[g, e].sum(); E = (bg[g] * self.up[g, e]).sum()
            out.append(("software", "version", int(v), float(lr_score(O, E)), O, E, g))
        # массовый уход: простои дольше суток, начавшиеся за 7 суток (последние сутки ещё не видны)
        h = WINDOW["exodus"]
        a, b = max(e - h, 0), max(e - 1, 0)
        if b > a:
            cnt = self.longst[:, :max(a - GAP["exodus"], 1)].sum(1)
            ex = self.up[:, :max(a - GAP["exodus"], 1)].sum(1)
            for ct, cl, key, g in self.groups["exodus"]:
                rate = (cnt[g].sum() + 0.5) / max(ex[g].sum(), 1.0)
                O = self.longst[g, a:b].sum(); E = rate * self.up[g, a:b].sum()
                out.append(("exodus", ct, cl, float(lr_score(O, E)), O, E, g))
        return out
