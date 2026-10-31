"""Прогноз кризисов агентом TARIS (шаг 5, подшаги П1–П4, для H8).

Агент видит только сеть: начала простоев по пингам, проваленные аудиты, признанные тихие потери, реестр и
заявленные признаки узлов, версии клиента. Для групп узлов, в которых случаются кризисы пункта 3.9, он
считает, во сколько раз частота событий канала выше фона узлов, и пуассоновское отношение правдоподобия:

    S = O * ln(O / E) - (O - E)   (0 при O <= E),

где O — наблюдаемое число событий группы в окне, E — ожидаемое по фону.

Группы и каналы (П1):
- буря и каскад — дата-центр (заявленный), начала простоев за последние 6, 12 или 24 ч (лучший из счётов; при
  суточном шаге — за 1, 2 или 3 суток); тревога «каскад», если в другом дата-центре того же провайдера за последние
  2 суток был массовый простой (сеть видит первичный кризис);
- износ партии — дата-центр × класс оборудования (заявленные), проваленные аудиты и тихие потери за 7 суток;
- уязвимость ПО — узлы одной версии клиента, начала простоев (окна те же, что у бури);
- массовый уход — регион, простои, ставшие длиннее суток за последние 7 суток.
Фон узла — его собственная частота до начала окна исключения (сглаженная к средней по сети на c0 суток).

Счётчики ведутся по ячейкам длины step_hours (Ч4): 24 — раз в сутки, 1 — раз в час для политики с подшагами
тревог (ledger, alarm_step_hours). Данные приходят через view.step за отрезок [t0, t1) с прошлого вызова.

Порог тревоги своего типа (П3) агент учит сам по своей истории: кризис в группе он видит как массовый простой
(не меньше половины узлов группы начинают простой длиннее франшизы за один отрезок) или волну полных гибелей.
"""
from __future__ import annotations

import numpy as np

from .config import HOUR

TYPES = ("storm", "cascade", "wear", "software", "exodus")
WINDOW = {"storm": 1, "cascade": 1, "software": 1, "wear": 7, "exodus": 7}       # окно счёта, сутки (для разметки)
GAP = {"storm": 7, "cascade": 7, "software": 7, "wear": 60, "exodus": 30}        # исключение из фона перед окном


def lr_score(O, E):
    O, E = np.asarray(O, float), np.maximum(np.asarray(E, float), 1e-9)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(O > E, O * np.log(np.where(O > 0, O / E, 1.0)) - (O - E), 0.0)
    return s


class Forecaster:
    """Счёт аномалий по группам узлов; данные копятся по ячейкам длины step_hours."""

    def __init__(self, world, exp, c0: float = 30.0, step_hours: int = 24):
        c = world.cfg
        self.exp, self.reg = exp, exp.reg
        self.T, self.N = c.horizon_days, exp.N
        self.d = c.franchise_d_hours * HOUR
        self.hw_names = c.hw_classes
        self.c0 = c0
        self.cpd = int(round(24 / step_hours))           # ячеек в сутках
        C = self.T * self.cpd
        f32 = np.float32
        self.starts = np.zeros((self.N, C), f32)         # начала простоев по ячейкам
        self.longst = np.zeros((self.N, C), f32)         # начала простоев, которые длятся дольше суток (уходы)
        self.audf = np.zeros((self.N, C), f32)           # проваленные аудиты и признанные тихие потери
        self.up = np.zeros((self.N, C), f32)             # экспозиция в ячейках: узел в реестре и не в долгом простое
        self.mass = np.zeros((self.N, self.T), bool)     # узел начал простой длиннее d при массовом простое дата-центра
        # накопленные по суткам суммы (для фона): индекс d — сумма по суткам < d
        self.cum = {k: np.zeros((self.N, self.T + 1)) for k in ("starts", "longst", "audf", "up")}
        self.cum_day = 0
        self.version = np.full(self.N, -1)
        self.n_silent = 0
        self.cell = -1                                   # последняя заполненная ячейка
        r = self.reg
        self.groups = {}
        dcs = np.unique(r.dc[r.dc >= 0])
        self.groups["storm"] = [("dc", int(d), int(d), np.flatnonzero(r.dc == d)) for d in dcs]
        self.groups["wear"] = [("batch", f"{int(d)}/{self.hw_names[h]}", (int(d), int(h)),
                                np.flatnonzero((r.dc == d) & (r.hw == h)))
                               for d in dcs for h in range(len(self.hw_names)) if ((r.dc == d) & (r.hw == h)).any()]
        self.groups["exodus"] = [("region", int(g), int(g), np.flatnonzero(r.region == g)) for g in np.unique(r.region)]
        self.dc_prov = {int(d): int(r.provider[r.dc == d][0]) for d in dcs}

    def _cells(self, x):
        return np.minimum(np.floor(np.asarray(x, float) * self.cpd + 1e-9).astype(int), self.T * self.cpd - 1)

    # ------------------------------------------------------------ накопление
    def update(self, t: float, view):
        """Забирает наблюдения view.step за [t0, t1) — то, что стало известно с прошлого вызова."""
        st = view.step
        if st is None:
            return
        t0, t1 = (st["t0"], st["t1"]) if "t0" in st else (float(st["epoch"]), float(st["epoch"]) + 1.0)
        c0 = int(self._cells(t0))
        c1 = min(int(np.ceil(t1 * self.cpd - 1e-6)) - 1, self.T * self.cpd - 1)     # последняя ячейка внутри [t0, t1)
        self.cell = c1
        reg = view.join_times() <= t1
        og = st["outages_ongoing"]
        on = np.zeros(self.N, bool)
        if len(og):
            on[og["node"].to_numpy()] = (t1 - og["start"].to_numpy()) > 1.0
        self.up[:, c0:c1 + 1] = (reg & ~on)[:, None]
        s = st["outages_started"]
        if len(s):
            np.add.at(self.starts, (s["node"].to_numpy(), self._cells(s["start"].to_numpy())), 1.0)
        a = st["audits"]
        if len(a):
            f = a[~a["passed_avail"]]
            np.add.at(self.audf, (f["node"].to_numpy(), self._cells(f["t"].to_numpy())), 1.0)
        sil = view.silent_declared()
        for ts, n in sil[self.n_silent:]:
            self.audf[n, self._cells(ts)] += 1.0
        self.n_silent = len(sil)
        if len(og):
            # простой стал длиннее суток за этот отрезок — отмечается в последней ячейке отрезка (когда это стало
            # известно): задним числом нельзя, накопленные суммы для фона по прошедшим суткам уже посчитаны
            st_og = og["start"].to_numpy()
            new_long = ((t1 - st_og) > 1.0) & ((t0 - st_og) <= 1.0)
            np.add.at(self.longst[:, c1], og["node"].to_numpy()[new_long], 1.0)
        self.version = view.node_versions()
        # массовый простой дата-центра: не меньше половины узлов в сети начали простой длиннее d за отрезок
        ended = st["outages_ended"]
        big = set(og["node"].to_numpy()[(t1 - og["start"].to_numpy()) > self.d].tolist()) if len(og) else set()
        if len(ended):
            big |= set(ended["node"].to_numpy()[ended["dur"].to_numpy() > self.d].tolist())
        started_now = set(s["node"].to_numpy().tolist()) if len(s) else set()
        hit = np.zeros(self.N, bool)
        hit[list(big & started_now)] = True
        e = min(int(t1 - 1e-9), self.T - 1)
        for _, _, _, g in self.groups["storm"]:
            n_hit = int(hit[g].sum())
            n_up = int((reg[g] & ~on[g]).sum()) + n_hit
            if n_hit >= 2 and n_hit >= 0.5 * n_up:
                self.mass[g[hit[g]], e] = True
        # накопленные суммы по завершённым суткам
        while self.cum_day < min(int(np.floor(t1 + 1e-9)), self.T):
            d0 = self.cum_day
            sl = slice(d0 * self.cpd, (d0 + 1) * self.cpd)
            for k in self.cum:
                self.cum[k][:, d0 + 1] = self.cum[k][:, d0] + getattr(self, k)[:, sl].sum(1)
            self.cum_day += 1

    # ------------------------------------------------------------ счёт
    def _bg(self, key, day):
        """Фоновая частота канала (в сутки) у всех узлов по истории до суток day (сглаженная к средней по сети)."""
        day = int(min(max(day, 0), self.cum_day))
        if day == 0:
            return None
        cnt = self.cum[key][:, day]
        ex = self.cum["up"][:, day] / self.cpd
        m = cnt.sum() / max(ex.sum(), 1e-9)
        return (cnt + m * self.c0) / (ex + self.c0)

    def _window(self, M, g, cells, bg):
        a = max(self.cell - cells + 1, 0)
        O = float(M[g, a:self.cell + 1].sum())
        E = float((bg[g] * self.up[g, a:self.cell + 1].sum(1)).sum() / self.cpd)
        return O, E

    def scores(self) -> list:
        """Счёт по всем группам по данным до последней заполненной ячейки: (тип, ctype, кластер, S, O, E, узлы)."""
        if self.cell < 0:
            return []
        day = (self.cell + 1) / self.cpd                   # момент конца данных, сутки
        if day < 30:
            return []
        out = []
        # буря / каскад / ПО: начала простоев; окна 6, 12, 24 ч (при суточных ячейках — 1, 2, 3 суток)
        wins = (1, 2, 3) if self.cpd == 1 else tuple(max(1, int(round(h * self.cpd / 24))) for h in (6, 12, 24))
        bg = self._bg("starts", int(day) - GAP["storm"])
        if bg is None:
            return []
        e = min(int(day - 1e-9), self.T - 1)
        recent_mass = {}
        for ct, cl, key, g in self.groups["storm"]:
            if self.mass[g, max(e - 2, 0):e + 1].any():
                recent_mass.setdefault(self.dc_prov[cl], set()).add(cl)

        def best(g):
            b = None
            for h in wins:
                O, E = self._window(self.starts, g, h, bg)
                S = float(lr_score(O, E))
                if b is None or S > b[0]:
                    b = (S, O, E)
            return b

        for ct, cl, key, g in self.groups["storm"]:
            S, O, E = best(g)
            sibling = recent_mass.get(self.dc_prov[cl], set()) - {cl}
            out.append(("cascade" if sibling else "storm", ct, cl, S, O, E, g))
        for v in np.unique(self.version[self.version > 0]):
            g = np.flatnonzero(self.version == v)
            S, O, E = best(g)
            out.append(("software", "version", int(v), S, O, E, g))
        # износ партии: аудиты и тихие потери за 7 суток
        bgw = self._bg("audf", int(day) - 7 - GAP["wear"])
        if bgw is not None:
            for ct, cl, key, g in self.groups["wear"]:
                O, E = self._window(self.audf, g, 7 * self.cpd, bgw)
                out.append(("wear", ct, cl, float(lr_score(O, E)), O, E, g))
        # массовый уход: простои, ставшие длиннее суток за последние 7 суток
        cut = min(int(day) - 7 - GAP["exodus"], self.cum_day)
        if cut > 0:
            cnt, ex = self.cum["longst"][:, cut], self.cum["up"][:, cut] / self.cpd
            a, b = max(self.cell - 7 * self.cpd + 1, 0), self.cell + 1
            for ct, cl, key, g in self.groups["exodus"]:
                rate = (cnt[g].sum() + 0.5) / max(ex[g].sum(), 1.0)
                O = float(self.longst[g, a:b].sum()); E = rate * float(self.up[g, a:b].sum()) / self.cpd
                out.append(("exodus", ct, cl, float(lr_score(O, E)), O, E, g))
        return out
