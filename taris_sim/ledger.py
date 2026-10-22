"""Учёт исходов и сквозной прогон (шаг 3, пункт 3.6).

Ledger исполняет решения политики в мире: размещает фрагменты, применяет события мира
(уничтожение данных, признание фрагмента утраченным, тихие потери и их обнаружение аудитом),
выполняет восстановления и ведёт журнал: размещения, потери, восстановления, раскрытия.

Правила (согласованы со спецификацией и шагом 2):
- фрагмент существует, пока его данные целы и сеть не признала его утраченным;
- простой дольше T_rep: через T_rep после начала сеть признаёт фрагменты узла утраченными;
- шок с полной гибелью и уход узла уничтожают данные сразу, признание — через T_rep;
- тихая потеря уничтожает один фрагмент сразу, признание — на первом аудите при доступном узле;
- объект потерян, когда существующих фрагментов меньше k;
- восстановление читает k доступных фрагментов и пишет r новых: трафик (k + r) * size / k.
"""
from __future__ import annotations

import heapq
import numpy as np
import pandas as pd

from .config import HOUR
from .world import World
from .chain import Chain, merge_intervals
from .data import DataStream
from . import rng as R


class NetworkView:
    """То, что видит политика в момент принятия решения."""

    def __init__(self, ledger: "Ledger"):
        self.L = ledger
        self.t = 0.0
        self.seed = b"\0" * 32

    def candidates(self) -> np.ndarray:
        """Узлы, зарегистрированные к моменту t, не находящиеся в наблюдаемом простое и не заполненные."""
        L = self.L
        e = int(self.t)
        ok = L.registered_by(self.t) & ~L.obs_down[:, min(e, L.T - 1)] & (L.used < L.capacity)
        return np.flatnonzero(ok)

    def needs_repair(self):
        return [o for o in self.L.need_repair if self.L.obj_lost_t[o] == np.inf]

    def missing(self, obj: int) -> set:
        return self.L.missing[obj] - self.L.pending[obj]

    def holders(self, obj: int) -> set:
        return {self.L.f_node[f] for f in self.L.obj_frags[obj]}

    # ---- реестр узлов и версии клиента (пункт 3.9): то, что узлы публикуют о себе
    def registered(self) -> np.ndarray:
        """Узлы, зарегистрированные к моменту t (отменённые регистрации сюда не попадают никогда)."""
        return np.flatnonzero(self.L.registered_by(self.t))

    def join_times(self) -> np.ndarray:
        """Моменты регистрации узлов; у ещё не пришедших — inf (будущее политике не видно)."""
        j = self.L.join
        return np.where(j <= self.t, j, np.inf)

    def releases(self) -> np.ndarray:
        """Моменты выпуска версий клиента к моменту t; версия 0 — исходная (момент -inf)."""
        cr = self.L.w.crisis
        if cr is None:
            return np.array([-np.inf])
        return cr.release[cr.release <= self.t]

    def node_versions(self) -> np.ndarray:
        """Заявленная версия клиента каждого узла в момент t; -1 у ещё не зарегистрированных."""
        L = self.L
        reg = L.registered_by(self.t)
        if L.w.crisis is None:
            return np.where(reg, 0, -1)
        return np.where(reg, L.w.crisis.version_at(self.t), -1)


class Ledger:
    def __init__(self, world: World, chain: Chain, data: DataStream):
        self.w, self.ch, self.data, self.cfg = world, chain, data, world.cfg
        c = self.cfg
        self.T = c.horizon_days
        nd = world.nodes
        self.N = len(nd)
        self.capacity = data.capacity
        self.join = nd["join_t"].to_numpy()
        self.owner = nd["owner"].to_numpy()
        # наблюдаемый простой на границах эпох (что видит политика)
        self.obs_down = np.zeros((self.N, self.T), bool)
        o = chain.observed
        e0 = np.ceil(o["start"].to_numpy()).astype(int)
        e1 = np.ceil(o["end"].to_numpy()).astype(int) - 1
        for i, a, b in zip(o["node"].to_numpy(), e0, np.minimum(e1, self.T - 1)):
            if b >= a:
                self.obs_down[i, a:b + 1] = True
        # истинные интервалы недоступности (для чтения при восстановлении и учёта доступности)
        td = merge_intervals(chain.true_down.drop(columns="source").assign(source="own")).drop(columns="source")
        self.down = {i: (g["start"].to_numpy(), g["end"].to_numpy()) for i, g in td.groupby("node")}
        self.true_down_at = np.zeros((self.N, self.T), bool)       # истинный простой в момент начала эпохи
        e0 = np.ceil(td["start"].to_numpy()).astype(int)
        e1 = np.ceil(td["end"].to_numpy()).astype(int) - 1
        for i, a, b in zip(td["node"].to_numpy(), e0, np.minimum(e1, self.T - 1)):
            if b >= a:
                self.true_down_at[i, a:b + 1] = True
        self.alive_to = nd["alive_to"].to_numpy()
        self.size = data.objects["size"].to_numpy()
        # время обнаружения тихой потери: первый аудит после события при доступном узле без провала
        # (шум аудитора или предвестник износа, пункт 3.9)
        a = chain.audits[chain.audits["passed_avail"]]
        self.audit_ok = {i: g["t"].to_numpy() for i, g in a.groupby("node")}

    # ------------------------------------------------------------ сервис
    def registered_by(self, t):
        return self.join <= t

    def is_up(self, node, t) -> bool:
        s, e = self.down.get(node, (None, None))
        if s is None:
            return True
        j = np.searchsorted(s, t, side="right") - 1
        return not (j >= 0 and t < e[j])

    # ------------------------------------------------------------ состояние
    def _reset_state(self):
        self.f_obj, self.f_idx, self.f_node, self.f_from, self.f_to, self.f_end = [], [], [], [], [], []
        self.f_exist, self.f_dead = [], []
        self.node_frags = [set() for _ in range(self.N)]
        self.used = np.zeros(self.N, int)
        n_obj = len(self.data.objects)
        self.obj_frags = [set() for _ in range(n_obj)]
        self.obj_n = np.zeros(n_obj, int)
        self.obj_exist = np.zeros(n_obj, int)
        self.obj_lost_t = np.full(n_obj, np.inf)
        self.missing = [set() for _ in range(n_obj)]
        self.pending = [set() for _ in range(n_obj)]
        self.need_repair = set()
        self.owner_idx = {}
        self.log_loss, self.log_repair, self.log_disc = [], [], []
        self.heap, self.seq = [], 0

    def _push(self, t, kind, payload):
        heapq.heappush(self.heap, (t, self.seq, kind, payload)); self.seq += 1

    def _world_events(self):
        c, w = self.cfg, self.w
        T_rep = c.T_rep
        out = w.outages
        for r in out[out["cls"] == "total"].itertuples():
            if r.destroyed:
                self._push(r.start, "destroy_node", r.node)
            if r.start + T_rep < self.T:
                self._push(r.start + T_rep, "declare_node", r.node)
        nd = w.nodes
        for i, lt in zip(nd.index, nd["leave_t"].to_numpy()):
            if np.isfinite(lt):
                self._push(lt, "destroy_node", i)
                if lt + T_rep < self.T:
                    self._push(lt + T_rep, "declare_node", i)
        for k, r in enumerate(w.silent.itertuples()):
            self._push(r.t, "silent", (r.node, k))

    # ------------------------------------------------------------ операции
    def _add_fragment(self, obj, idx, node, t):
        f = len(self.f_obj)
        self.f_obj.append(obj); self.f_idx.append(idx); self.f_node.append(node)
        self.f_from.append(t); self.f_to.append(np.inf); self.f_end.append(""); self.f_exist.append(True); self.f_dead.append(np.inf)
        self.node_frags[node].add(f); self.obj_frags[obj].add(f); self.used[node] += 1
        self.obj_exist[obj] += 1
        key = (self.owner[node], obj)
        s = self.owner_idx.setdefault(key, set()); s.add(idx)
        if len(s) == self.cfg.k:
            self.log_disc.append((obj, int(self.owner[node]), t))

    def _destroy(self, f, t):
        if self.f_exist[f]:
            self.f_exist[f] = False
            self.f_dead[f] = t
            obj = self.f_obj[f]
            self.obj_exist[obj] -= 1
            if self.obj_exist[obj] < self.cfg.k and self.obj_lost_t[obj] == np.inf:
                self.obj_lost_t[obj] = t
                self.log_loss.append((obj, t))

    def _declare(self, f, t, reason):
        """Сеть признаёт фрагмент утраченным: он удаляется из карты размещения и ждёт восстановления."""
        if self.f_to[f] != np.inf:
            return
        self._destroy(f, t)
        node, obj = self.f_node[f], self.f_obj[f]
        self.f_to[f] = t; self.f_end[f] = reason
        self.node_frags[node].discard(f); self.obj_frags[obj].discard(f); self.used[node] -= 1
        if self.obj_lost_t[obj] == np.inf:
            self.missing[obj].add(self.f_idx[f]); self.need_repair.add(obj)

    def _process(self, t_end):
        while self.heap and self.heap[0][0] < t_end:
            t, _, kind, p = heapq.heappop(self.heap)
            if kind == "destroy_node":
                for f in list(self.node_frags[p]):
                    self._destroy(f, t)
            elif kind == "declare_node":
                for f in list(self.node_frags[p]):
                    self._declare(f, t, "простой > T_rep")
            elif kind == "silent":
                node, k = p
                fr = [f for f in self.node_frags[node] if self.f_exist[f]]
                if fr:
                    g = R.stream(self.cfg.seed, R.SILENT, 10**6 + k)
                    f = fr[int(g.integers(len(fr)))]
                    self._destroy(f, t)
                    at = self.audit_ok.get(node, np.empty(0))
                    j = np.searchsorted(at, t, side="right")
                    if j < len(at):
                        self._push(at[j], "declare_frag", f)
            elif kind == "declare_frag":
                self._declare(p, t, "тихая потеря (аудит)")
            elif kind == "repair_done":
                obj, idx, nodes = p
                for i, n in zip(idx, nodes):
                    self.pending[obj].discard(i)
                    if self.obj_lost_t[obj] != np.inf:
                        continue
                    if self.alive_to[n] > t:
                        self._add_fragment(obj, i, n, t)
                        self.missing[obj].discard(i)
                if not (self.missing[obj] - self.pending[obj]):
                    self.need_repair.discard(obj)

    def _repair(self, t, reqs):
        c = self.cfg
        for obj, idx, nodes in reqs:
            if self.obj_lost_t[obj] != np.inf or not idx:
                continue
            e = min(int(t), self.T - 1)
            readable = sum(1 for f in self.obj_frags[obj] if self.f_exist[f] and not self.true_down_at[self.f_node[f], e])
            if readable < c.k:
                continue                         # прочитать нечего — повтор в следующую эпоху
            idx, nodes = list(idx)[:len(nodes)], list(nodes)[:len(idx)]
            r = len(idx)
            traffic = (c.k + r) * self.size[obj] / c.k
            self.log_repair.append((obj, t, r, traffic))
            self.pending[obj].update(idx)
            self._push(t + c.repair_hours * HOUR, "repair_done", (obj, idx, nodes))

    # ------------------------------------------------------------ прогон
    def run(self, policy, epochs: int | None = None, progress: bool = False):
        self._reset_state()
        self._world_events()
        self.ch.reset()
        view = NetworkView(self)
        T = self.T if epochs is None else epochs
        daily = []
        created = self.data.objects["created_epoch"].to_numpy()
        n_created = 0
        for t in range(T):
            step = self.ch.step()
            view.t, view.seed = float(t), bytes.fromhex(step["seed"])
            self._repair(float(t), policy.on_epoch(float(t), view))
            for obj in self.data.new_objects(t):
                nodes = policy.place(int(obj), float(t), view)
                self.obj_n[obj] = len(nodes)
                for i, n in enumerate(nodes):
                    self._add_fragment(int(obj), i, int(n), float(t))
            self._process(t + 1.0)
            n_created += int((created == t).sum())
            alive_objs = n_created - len(self.log_loss)
            daily.append((t, int(self.used.sum()), alive_objs))
            if progress and t % 100 == 0:
                print(f"  эпоха {t}: фрагментов {self.used.sum()}, потеряно объектов {len(self.log_loss)}")
        return RunResult(self, policy, pd.DataFrame(daily, columns=["epoch", "fragments", "objects_alive"]))


class RunResult:
    def __init__(self, L: Ledger, policy, daily: pd.DataFrame):
        c = L.cfg
        self.cfg, self.policy_name = c, policy.name
        self.daily = daily
        self.fragments = pd.DataFrame({"obj": L.f_obj, "idx": L.f_idx, "node": L.f_node, "t_from": L.f_from,
                                       "t_to": L.f_to, "end": L.f_end, "exist": L.f_exist, "t_dead": L.f_dead})
        self.losses = pd.DataFrame(L.log_loss, columns=["obj", "t"])
        self.repairs = pd.DataFrame(L.log_repair, columns=["obj", "t", "r", "traffic"])
        self.disclosures = pd.DataFrame(L.log_disc, columns=["obj", "owner", "t"])
        self.objects = L.data.objects.assign(n=L.obj_n, lost_t=L.obj_lost_t)
        self._L = L

    def metrics(self) -> dict:
        c, ob, d = self.cfg, self.objects, self.daily
        T = c.horizon_days
        size = ob["size"].to_numpy()
        years = int(T // 365)
        eafdl = []
        for y in range(years):
            a, b = 365 * y, 365 * (y + 1)
            lost = size[(ob["lost_t"] >= a) & (ob["lost_t"] < b)].sum()
            stored = d[(d.epoch >= a) & (d.epoch < b)]["objects_alive"].mean() * float(np.mean(size))
            eafdl.append(lost / stored if stored > 0 else np.nan)
        n_loss = len(self.losses)
        obj_days = d["objects_alive"].sum()
        return {
            "Политика": self.policy_name,
            "Объектов": len(ob),
            "Потеряно объектов": n_loss,
            "EAFDL (среднее по годам)": float(np.nanmean(eafdl)),
            "MTTDL объекта, лет": (obj_days / 365 / n_loss) if n_loss else np.inf,
            "Средняя избыточность n/k": float(d["fragments"].iloc[-1] / max(1, d["objects_alive"].iloc[-1]) / c.k),
            "Трафик восстановления в год": float(self.repairs["traffic"].sum() / (T / 365)),
            "Операций восстановления": len(self.repairs),
            "Раскрытий": len(self.disclosures),
        }

    def unavailability(self, n_objects: int | None = None) -> pd.DataFrame:
        """Доля времени, когда объект нельзя прочитать (доступно меньше k фрагментов), для выборки объектов."""
        L, c = self._L, self.cfg
        n_objects = n_objects or c.track_objects
        rng = np.random.default_rng(0)
        objs = rng.choice(len(self.objects), min(n_objects, len(self.objects)), replace=False)
        fr = self.fragments
        rows = []
        for obj in objs:
            t0 = float(self.objects.at[obj, "created_epoch"])
            t1 = min(float(self.objects.at[obj, "lost_t"]), float(c.horizon_days))
            if t1 <= t0:
                continue
            ev = []
            for r in fr[fr.obj == obj].itertuples():
                a, b = max(r.t_from, t0), min(r.t_dead, r.t_to, t1)
                if b <= a:
                    continue
                ev += [(a, 1), (b, -1)]
                s, e = L.down.get(r.node, (np.empty(0), np.empty(0)))
                m = (e > a) & (s < b)
                for ss, ee in zip(s[m], e[m]):
                    ev += [(max(ss, a), -1), (min(ee, b), 1)]
            if not ev:
                continue
            ev.sort()
            tt = np.array([e[0] for e in ev]); dd = np.cumsum([e[1] for e in ev])
            dur = np.diff(np.r_[tt, t1])
            un = dur[(dd < c.k)].sum()
            rows.append((obj, un / (t1 - t0)))
        return pd.DataFrame(rows, columns=["obj", "unavail_share"])
