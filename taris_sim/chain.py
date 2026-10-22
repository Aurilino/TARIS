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

SOURCE_PRIORITY = {"noise": 0, "micro": 1, "own": 2, "storm": 3, "shock": 4, "leave": 5}


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
        return pd.concat([real, micro, leave], ignore_index=True)

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
        o["is_real"] = o["source"].isin(["own", "shock", "storm", "leave"])
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
            ep = np.repeat(np.arange(e0, e1), c.audits_per_epoch)
            t = ep + g.random(len(ep))
            t = t[(t >= a0) & (t < a1)]
            noise = g.random(len(t)) < c.audit_noise
            keep = t < nd["alive_to"].iat[i]
            t, noise = t[keep], noise[keep]
            s, e = by_node.get(i, (np.empty(0), np.empty(0)))
            j = np.searchsorted(s, t, side="right") - 1
            down = (j >= 0) & (t < e[np.maximum(j, 0)])
            parts.append(pd.DataFrame({"node": i, "t": t, "available": ~down, "noise_fail": noise}))
        a = pd.concat(parts, ignore_index=True)
        a["epoch"] = np.floor(a["t"]).astype(int)
        a["passed_avail"] = a["available"] & ~a["noise_fail"]     # итог без учёта целостности фрагмента
        return a

    # ------------------------------------------------- пошаговый интерфейс
    def reset(self):
        self.t = 0
        self.seed_t = hashlib.sha256(f"genesis:{self.cfg.seed}".encode()).digest()

    def step(self) -> dict:
        """Одна эпоха: общая случайность и наблюдения, ставшие известными к её концу."""
        t = self.t
        self.seed_t = hashlib.sha256(self.seed_t + t.to_bytes(4, "big")).digest()
        o = self.observed
        started = o[(o["start"] >= t) & (o["start"] < t + 1)]
        ended = o[(o["end"] >= t) & (o["end"] < t + 1)]
        ongoing = o[(o["start"] < t + 1) & (o["end"] >= t + 1)]
        aud = self.audits[self.audits["epoch"] == t]
        self.t += 1
        return {"epoch": t, "seed": self.seed_t.hex(),
                "outages_started": started, "outages_ended": ended, "outages_ongoing": ongoing,
                "audits": aud}
