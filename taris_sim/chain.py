"""Эпохи, общая случайность, пинги, аудиты и микропрерывания (шаг 3, пункт 3.3).

Истинное состояние узла («взгляд истины») — интервалы недоступности из world плюс микропрерывания.
Сеть видит только результаты пингов (сетка с шагом ping, своя фаза у каждого узла) и аудитов.
Всё здесь зависит только от мира, а не от политики, поэтому наблюдения строятся заранее,
а step() выдаёт их по эпохам. Целостность фрагментов при аудите проверяет ledger (там известны фрагменты).
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from .config import Config, MINUTE
from .world import World
from . import rng as R

SOURCE_PRIORITY = {"noise": 0, "micro": 1, "pre_micro": 2, "own": 3, "pre_short": 4, "storm": 5, "shock": 6, "crisis": 7, "leave": 8}
REAL_SOURCES = ["own", "pre_short", "shock", "storm", "crisis", "leave"]     # простои, а не шум и микропрерывания


def merge_intervals(df: pd.DataFrame, extra_cols=("source",)) -> pd.DataFrame:
    """Слияние перекрывающихся и смежных интервалов по каждому узлу (векторно)."""
    if df.empty:
        return df
    df = df.sort_values(["node", "start"], kind="mergesort").reset_index(drop=True)
    node = df["node"].to_numpy()
    end_cmax = df.groupby("node", sort=False)["end"].cummax().to_numpy()
    prev = np.r_[-np.inf, end_cmax[:-1]]
    new_node = np.r_[True, node[1:] != node[:-1]]
    new_grp = new_node | (df["start"].to_numpy() > prev)
    gid = np.cumsum(new_grp)
    agg = {"node": "first", "start": "min", "end": "max"}
    if "source" in df:
        df["prio"] = df["source"].map(SOURCE_PRIORITY)
        agg["prio"] = "max"
    out = df.groupby(gid).agg(agg).reset_index(drop=True)
    if "prio" in out:
        inv = {v: k for k, v in SOURCE_PRIORITY.items()}
        out["source"] = out.pop("prio").map(inv)
    return out


class Chain:
    def __init__(self, world: World):
        self.world = world
        self.cfg: Config = world.cfg
        c = self.cfg
        nd = world.nodes
        self.phase = np.array([R.stream(c.seed, R.PING, i).uniform(0, c.ping) for i in range(len(nd))])
        self.micro = self._gen_micro()
        self.true_down = self._true_down()
        self.observed = self._observe()
        self.audits = self._audits()
        self.reset()

    # ---------------------------------------------------- истинное состояние
    def _gen_micro(self) -> pd.DataFrame:
        c, nd = self.cfg, self.world.nodes
        d, mean = c.d, c.micro_mean_minutes * MINUTE
        parts = []
        for i in range(len(nd)):
            # генерация по базовой жизни, обрезка по итоговой (общие случайные числа, пункт 3.9)
            a0, a1 = nd["alive_from"].iat[i], nd["alive_to_base"].iat[i]
            if a1 <= a0:
                continue
            g = R.stream(c.seed, R.MICRO, i)
            m = g.poisson(nd["lam_micro"].iat[i] * (a1 - a0))
            st = g.uniform(a0, a1, m)
            # экспоненциальная длительность, усечённая до d (микропрерывание короче франшизы)
            u = g.random(m)
            dur = -mean * np.log(1 - u * (1 - np.exp(-d / mean)))
            keep = st < nd["alive_to"].iat[i]
            parts.append(pd.DataFrame({"node": i, "start": st[keep], "end": (st + dur)[keep]}))
        return pd.concat(parts, ignore_index=True)

    def _true_down(self) -> pd.DataFrame:
        w, c = self.world, self.cfg
        real = w.outages[["node", "start", "end", "kind"]].rename(columns={"kind": "source"})
        micro = self.micro.assign(source="micro")
        nd = w.nodes
        lv = nd[np.isfinite(nd["leave_t"])]
        leave = pd.DataFrame({"node": lv.index, "start": lv["leave_t"].to_numpy(), "end": float(c.horizon_days), "source": "leave"})
        parts = [real, micro, leave]
        if w.crisis is not None:
            # микропрерывания-предвестники кризисов (пункт 3.9): сеть видит их как обычные микропрерывания
            pm = w.pre_events[w.pre_events["channel"] == "micro"]
            parts.append(pd.DataFrame({"node": pm["node"].to_numpy(), "start": pm["start"].to_numpy(),
                                       "end": pm["end"].to_numpy(), "source": "pre_micro"}))
        return pd.concat(parts, ignore_index=True)

    # ------------------------------------------------------------ пинги
    def _observe(self) -> pd.DataFrame:
        """Сеть видит простой как серию проваленных пингов. Наблюдаемое начало — первый проваленный пинг,
        наблюдаемый конец — первый успешный пинг после простоя."""
        c, nd = self.cfg, self.world.nodes
        P = c.ping
        td = self.true_down
        ph = self.phase[td["node"].to_numpy()]
        m0 = np.ceil((td["start"].to_numpy() - ph) / P)
        m1 = np.ceil((td["end"].to_numpy() - ph) / P) - 1
        seen = m1 >= m0
        obs = pd.DataFrame({
            "node": td["node"].to_numpy()[seen],
            "start": (ph + m0 * P)[seen],
            "end": (ph + (m1 + 1) * P)[seen],
            "source": td["source"].to_numpy()[seen],
        })
        # для проверок: как выглядел бы каждый истинный простой, если бы он был единственным
        self.obs_unmerged = obs.assign(true_start=td["start"].to_numpy()[seen], true_end=td["end"].to_numpy()[seen])
        # ложные провалы пингов: шум сети — одиночные проваленные пинги
        parts = []
        for i in range(len(nd)):
            a0, a1 = nd["alive_from"].iat[i], nd["alive_to_base"].iat[i]
            if a1 <= a0:
                continue
            g = R.stream(c.seed, R.NOISE, i)
            n_p = int((a1 - a0) / P)
            m = g.binomial(n_p, c.ping_noise)
            idx = np.floor((a0 - self.phase[i]) / P) + 1 + g.integers(0, n_p, m)
            st = self.phase[i] + idx * P
            st = st[st < nd["alive_to"].iat[i]]
            parts.append(pd.DataFrame({"node": i, "start": st, "end": st + P, "source": "noise"}))
        self.n_noise_pings = int(sum(len(p_) for p_ in parts))
        obs = pd.concat([obs] + parts, ignore_index=True)
        obs = merge_intervals(obs)
        obs["dur"] = obs["end"] - obs["start"]
        return obs

    def classify(self, d: float | None = None) -> pd.DataFrame:
        """Классы наблюдаемых простоев при франшизе d (по умолчанию — из конфигурации)."""
        c = self.cfg
        d = c.d if d is None else d
        o = self.observed.copy()
        o["cls"] = np.select([o["dur"] <= d, o["dur"] <= c.T_rep], ["micro", "partial"], "total")
        o["is_real"] = o["source"].isin(REAL_SOURCES)
        return o

    # ------------------------------------------------------------ аудиты
    def _audits(self) -> pd.DataFrame:
        """Аудиты PDP: узел должен быть доступен в момент аудита; плюс ложные провалы (шум аудитора)."""
        c, nd = self.cfg, self.world.nodes
        merged = merge_intervals(self.true_down.drop(columns="source").assign(source="own")).drop(columns="source")
        by_node = {i: (g["start"].to_numpy(), g["end"].to_numpy()) for i, g in merged.groupby("node")}
        parts = []
        for i in range(len(nd)):
            a0, a1 = nd["alive_from"].iat[i], nd["alive_to_base"].iat[i]
            if a1 <= a0:
                continue
            g = R.stream(c.seed, R.AUDIT, i)
            e0, e1 = int(np.floor(a0)), int(np.ceil(a1))
            ep = np.arange(e0, e1)
            if c.audit_interval_days > 1:                          # редкие аудиты: сутки со сдвигом по номеру узла
                ep = ep[(ep - i) % c.audit_interval_days == 0]
            ep = np.repeat(ep, c.audits_per_epoch)
            t = ep + g.random(len(ep))
            t = t[(t >= a0) & (t < a1)]
            noise = g.random(len(t)) < c.audit_noise
            keep = t < nd["alive_to"].iat[i]
            t, noise = t[keep], noise[keep]
            s, e = by_node.get(i, (np.empty(0), np.empty(0)))
            if len(s):
                j = np.searchsorted(s, t, side="right") - 1
                down = (j >= 0) & (t < e[np.maximum(j, 0)])
            else:                                                  # у узла не было ни одного простоя
                down = np.zeros(len(t), bool)
            parts.append(pd.DataFrame({"node": i, "t": t, "available": ~down, "noise_fail": noise}))
        a = pd.concat(parts, ignore_index=True)
        a["epoch"] = np.floor(a["t"]).astype(int)
        a["passed_avail"] = a["available"] & ~a["noise_fail"]     # итог без учёта целостности фрагмента
        if self.world.crisis is not None:
            a["pre_fail"] = self._pre_audit_fails(a)
            a["passed_avail"] &= ~a["pre_fail"]
        return a

    def _pre_audit_fails(self, a: pd.DataFrame) -> np.ndarray:
        """Предвестник износа (пункт 3.9): в окне вероятность ложного провала аудита растёт до
        p_noise * (1 + (rho - 1) * g(u)). Шумовой провал уже разыгран, поэтому здесь добавка
        p_noise * (rho - 1) * g(u) / (1 - p_noise) — среди аудитов без шумового провала."""
        c, w = self.cfg, self.world
        pre = np.zeros(len(a), bool)
        win = w.pre_audit
        if win.empty:
            return pre
        node = a["node"].to_numpy()
        t = a["t"].to_numpy()
        noise = a["noise_fail"].to_numpy()
        for i, g in win.groupby("node"):
            rows = np.flatnonzero(node == i)
            r = R.stream(c.seed, R.CRISIS, 20, int(i)).random(len(rows))   # одно число на каждый аудит узла
            extra = np.zeros(len(rows))
            for wr in g.itertuples(index=False):
                inside = (t[rows] >= wr.start) & (t[rows] < wr.end)
                u = (t[rows][inside] - wr.w0) / max(wr.tau - wr.w0, 1e-12)
                extra[inside] += c.audit_noise * (wr.rho - 1.0) * w.crisis.g_shape(u)
            pre[rows] = ~noise[rows] & (r < extra / (1.0 - c.audit_noise))
        return pre

    # ------------------------------------------------- пошаговый интерфейс
    def reset(self):
        self.t = 0
        self.seed_t = hashlib.sha256(f"genesis:{self.cfg.seed}".encode()).digest()

    def step(self) -> dict:
        """Одна эпоха: общая случайность и наблюдения, ставшие известными к её концу.
        Только «взгляд сети» (пункт 3.9, проверка Д3): ни источника простоя, ни признаков шума и
        предвестника, а у ещё идущих простоев — ни конца, ни длительности (это было бы будущее)."""
        t = self.t
        self.seed_t = hashlib.sha256(self.seed_t + t.to_bytes(4, "big")).digest()
        out = self.observe(float(t), float(t + 1))
        self.t += 1
        return {"epoch": t, "seed": self.seed_t.hex(), **out}

    def observe(self, a: float, b: float) -> dict:
        """Наблюдения сети за отрезок [a, b): начавшиеся и закончившиеся в нём простои, простои, идущие
        в момент b (без конца и длительности), и аудиты. Для подшагов политики внутри эпохи (тревоги и
        перенос раз в час) — те же данные, что у step(), только за более короткий отрезок."""
        o = self.observed
        started = o.loc[(o["start"] >= a) & (o["start"] < b), ["node", "start"]]
        ended = o.loc[(o["end"] >= a) & (o["end"] < b), ["node", "start", "end", "dur"]]
        ongoing = o.loc[(o["start"] < b) & (o["end"] >= b), ["node", "start"]]
        au = self.audits
        aud = au.loc[(au["t"] >= a) & (au["t"] < b), ["node", "t", "epoch", "passed_avail"]]
        return {"t0": a, "t1": b, "outages_started": started, "outages_ended": ended,
                "outages_ongoing": ongoing, "audits": aud}
