"""Таблица опыта агента TARIS (шаг 5, пункт 5.1, подшаг А1).

Агент копит то, что видит сеть: простои по пингам (вывод chain.step() по эпохам), тихие потери, признанные
по аудиту, число фрагментов на узлах и реестр узлов (момент регистрации, регион, провайдер — проверяемые
признаки; класс оборудования и дата-центр — заявленные). Истинных столбцов мира (Theta, источник простоя,
жизнь узла, владелец) здесь нет: реестр берёт из world.nodes только перечисленные столбцы.

В момент t используются только наблюдения до t: простой, который по данным эпохи закончился позже t,
считается ещё идущим. Время — сутки, длительности простоев — тоже в сутках (франшиза d и T_rep переводятся).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import HOUR

VISIBLE = ("region", "provider", "subnet")          # проверяемые сетью (по IP)
CLAIMED = ("hw_claimed", "dc")                      # заявленные узлом (подделываются в S5)


class Registry:
    """Реестр узлов: только публичные признаки. Момент регистрации берётся из view.join_times()."""

    def __init__(self, world):
        nd = world.nodes
        self.N = len(nd)
        self.region = nd["region"].to_numpy().astype(int)
        self.provider = nd["provider"].to_numpy().astype(int)
        self.wallet = pd.factorize(nd["wallet"], sort=True)[0].astype(int)
        self.subnet = pd.factorize(nd["subnet"], sort=True)[0].astype(int)
        self.hw = nd["hw_claimed"].to_numpy().astype(int)
        self.dc = nd["dc"].to_numpy().astype(int)
        self.is_device = self.dc < 0                 # заявленный класс: устройство пользователя (пункт 3.10)
        self.n_region = int(self.region.max()) + 1
        self.n_hw = len(world.cfg.hw_classes)


class Experience:
    """Накопленные наблюдения сети и построение таблицы «узел × период» для моделей частоты и тяжести."""

    def __init__(self, world, block: int = 30):
        c = world.cfg
        self.reg = Registry(world)
        self.d = c.franchise_d_hours * HOUR
        self.T_rep = c.T_rep_hours * HOUR
        self.block = block
        self.N = self.reg.N
        self._s, self._e, self._n = [], [], []          # закончившиеся простои (по данным эпох)
        self.ongoing = {}                                # узел -> начало идущего простоя (по последней эпохе)
        self.join = np.full(self.N, np.inf)
        self.node_days_stored = np.zeros(self.N)         # узло-сутки с фрагментами (экспозиция тихих потерь)
        self.silent = []                                 # (t, узел)
        self.t = 0.0

    # ------------------------------------------------------------ накопление
    def update(self, t: float, view):
        """Вызывается политикой раз в эпоху: забирает view.step и состояние реестра."""
        st = view.step
        self.t = float(t)
        self.join = view.join_times()
        if st is not None:
            e = st["outages_ended"]
            if len(e):
                self._n.append(e["node"].to_numpy()); self._s.append(e["start"].to_numpy()); self._e.append(e["end"].to_numpy())
            og = st["outages_ongoing"]
            self.ongoing = dict(zip(og["node"].to_numpy().tolist(), og["start"].to_numpy().tolist()))
        self.node_days_stored += view.stored() > 0
        self.silent = view.silent_declared()

    def intervals(self, now: float | None = None) -> pd.DataFrame:
        """Наблюдаемые простои к моменту now: node, start, end (inf — ещё идёт), dur_seen (известная часть)."""
        now = self.t if now is None else now
        if self._n:
            n, s, e = np.concatenate(self._n), np.concatenate(self._s), np.concatenate(self._e)
        else:
            n = s = e = np.empty(0)
        on = np.array(list(self.ongoing.items()), float).reshape(-1, 2)
        n = np.r_[n, on[:, 0]].astype(int); s = np.r_[s, on[:, 1]]; e = np.r_[e, np.full(len(on), np.inf)]
        df = pd.DataFrame({"node": n, "start": s, "end": e}).drop_duplicates(["node", "start"], keep="first")
        df = df[df["start"] < now]
        df.loc[df["end"] > now, "end"] = np.inf                   # конец после now политике ещё не известен
        df["dur_seen"] = np.minimum(df["end"], now) - df["start"]
        return df.sort_values(["node", "start"]).reset_index(drop=True)

    # ------------------------------------------------------------ шоки кластеров (А6)
    def shock_episodes(self, now: float | None = None, iv: pd.DataFrame | None = None,
                       ctypes=("region", "provider", "dc"), frac: float = 0.6, min_nodes: int = 3,
                       window_min: float = 10.0, purity: float = 0.9):
        """Эпизоды общих шоков в данных сети. Всплеск — не меньше min_nodes узлов, начавших простой длиннее d
        в окне window_min минут. Шок относится к наименьшему кластеру (по числу доступных узлов), в котором
        лежит не меньше purity узлов всплеска и который всплеск накрывает не меньше чем на frac.
        Возвращает (эпизоды: ctype, cluster, start, n_nodes, n_up, total — доля простоев эпизода длиннее T_rep,
        nan — исход ещё не ясен; маску простоев iv, отнесённых к шокам)."""
        now = self.t if now is None else now
        iv = self.intervals(now) if iv is None else iv
        r, win = self.reg, window_min / 1440.0
        node, st, en, dur = (iv[c].to_numpy() for c in ("node", "start", "end", "dur_seen"))
        member = {"region": r.region, "provider": r.provider, "dc": r.dc}
        join = np.where(np.isfinite(self.join), self.join, np.inf)
        used = np.zeros(len(iv), bool)
        sel = np.flatnonzero(dur > self.d)
        sel = sel[np.argsort(st[sel], kind="mergesort")]
        ts = st[sel]
        eps, j0 = [], 0
        while j0 < len(sel):
            j1 = np.searchsorted(ts, ts[j0] + win, side="right")
            if j1 - j0 < min_nodes:
                j0 += 1
                continue
            grp = sel[j0:j1]
            t0 = ts[j0]
            best = None
            for ct in ctypes:
                ids = member[ct][node[grp]]
                vals, cnts = np.unique(ids, return_counts=True)
                cl = vals[np.argmax(cnts)]
                if cl < 0:
                    continue
                inside = grp[ids == cl]
                nn = len(np.unique(node[inside]))
                if nn < min_nodes or nn < purity * len(np.unique(node[grp])):
                    continue
                nodes_cl = np.flatnonzero(member[ct] == cl)
                in_cl = np.isin(node, nodes_cl)
                down = np.unique(node[in_cl & (st < t0 - 1e-9) & (en > t0)])
                n_up = int((join[nodes_cl] <= t0).sum()) - len(down)
                if nn >= frac * max(n_up, 1) and (best is None or n_up < best[4]):
                    best = (ct, int(cl), inside, nn, n_up)
            if best is None:
                j0 += 1
                continue
            ct, cl, inside, nn, n_up = best
            used[inside] = True
            d_ = dur[inside]
            resolved = np.isfinite(en[inside]) | (d_ > self.T_rep)
            tot = float((d_ > self.T_rep).mean()) if resolved.all() else np.nan
            eps.append((ct, cl, float(t0), nn, n_up, tot))
            j0 = j1
        ep = pd.DataFrame(eps, columns=["ctype", "cluster", "start", "n_nodes", "n_up", "total"])
        return ep, used

    def events(self, now: float | None = None, iv: pd.DataFrame | None = None, drop=None) -> pd.DataFrame:
        """Страховые случаи (простои, известные как длиннее d) с исходом для тяжести:
        dur — известная длительность (у идущих — прошедшая), done — простой закончился,
        total — 1, если уже ясно, что длиннее T_rep; 0 — закончился короче; nan — исход ещё не ясен.
        drop — маска простоев iv, которые не считаются собственными (отнесены к шокам кластеров)."""
        iv = self.intervals(now) if iv is None else iv
        keep = iv["dur_seen"] > self.d
        if drop is not None:
            keep &= ~drop
        ev = iv[keep].copy()
        ev["done"] = np.isfinite(ev["end"])
        ev["total"] = np.where(ev["dur_seen"] > self.T_rep, 1.0, np.where(ev["done"], 0.0, np.nan))
        return ev.rename(columns={"dur_seen": "dur"})

    # ------------------------------------------------------------ таблица «узел × период»
    def table(self, now: float | None = None, iv: pd.DataFrame | None = None, drop=None) -> pd.DataFrame:
        """Строки (узел, период длины block): экспозиция e (сутки в сети и не в простое), число случаев N,
        признаки: возраст в середине периода, частота микропрерываний до начала периода (опыт узла)."""
        now = self.t if now is None else now
        B, N = self.block, self.N
        nb = int(np.ceil(now / B))
        if nb == 0:
            return pd.DataFrame()
        edges = np.minimum(np.arange(nb + 1) * B, now).astype(float)
        join = np.where(np.isfinite(self.join), self.join, np.inf)
        # время в реестре по периодам
        reg = np.clip(edges[None, 1:] - np.maximum(join[:, None], edges[None, :-1]), 0.0, None)
        iv = self.intervals(now) if iv is None else iv
        node, s, e = iv["node"].to_numpy(), np.maximum(iv["start"].to_numpy(), 0.0), np.minimum(iv["end"].to_numpy(), now)
        down = np.zeros((N, nb))
        b0 = np.floor(s / B).astype(int); b1 = np.minimum(np.floor(np.nextafter(e, -np.inf) / B).astype(int), nb - 1)
        one = b0 == b1
        np.add.at(down, (node[one], b0[one]), e[one] - s[one])
        for i, a, b, x, y in zip(node[~one], b0[~one], b1[~one], s[~one], e[~one]):
            for j in range(a, b + 1):
                down[i, j] += max(0.0, min(y, edges[j + 1]) - max(x, edges[j]))
        expo = np.clip(reg - down, 0.0, None)
        dur = iv["dur_seen"].to_numpy()
        big = dur > self.d
        own = big if drop is None else big & ~np.asarray(drop)     # простои шоков — не собственные случаи узла
        cnt = np.zeros((N, nb)); np.add.at(cnt, (node[own], b0[own]), 1.0)
        micro = np.zeros((N, nb)); np.add.at(micro, (node[~big], b0[~big]), 1.0)
        # частота микропрерываний до начала периода (сглаженная к средней по сети на c0 суток)
        cm, ce = np.cumsum(micro, 1) - micro, np.cumsum(expo, 1) - expo
        m0 = micro.sum() / max(expo.sum(), 1e-9)
        c0 = 5.0
        micro_rate = (cm + m0 * c0) / (ce + c0)
        ii, bb = np.nonzero(reg > 0)
        mid = 0.5 * (edges[bb] + edges[bb + 1])
        r = self.reg
        return pd.DataFrame({"node": ii, "block": bb, "e": expo[ii, bb], "N": cnt[ii, bb],
                             "age": np.maximum(mid - join[ii], 0.0), "micro_rate": micro_rate[ii, bb],
                             "region": r.region[ii], "provider": r.provider[ii], "hw": r.hw[ii],
                             "device": r.is_device[ii].astype(int)})

    def micro_rate_now(self, now: float | None = None) -> np.ndarray:
        """Сглаженная частота микропрерываний каждого узла за всю историю до now (признак для прогноза)."""
        now = self.t if now is None else now
        iv = self.intervals(now)
        small = iv[iv["dur_seen"] <= self.d]
        cnt = np.bincount(small["node"].to_numpy(), minlength=self.N).astype(float)
        join = np.where(np.isfinite(self.join), self.join, now)
        down = np.bincount(iv["node"].to_numpy(), weights=(np.minimum(iv["end"], now) - iv["start"]).to_numpy(), minlength=self.N)
        expo = np.clip(now - np.maximum(join, 0.0) - down, 0.0, None)
        m0 = cnt.sum() / max(expo.sum(), 1e-9)
        c0 = 5.0
        return (cnt + m0 * c0) / (expo + c0)
