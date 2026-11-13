"""Отдельный исполнитель I4.1: тайм-аут не уничтожает физическую копию."""
import heapq
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from taris_sim.ledger import Ledger, NetworkView, RunResult
from taris_sim import rng as R


class PhysicalRunResult(RunResult):
    """Не позволяет случайно применить старые метрики конца карты к физическим копиям."""
    def metrics(self):
        return {"schema": "taris-physical-metrics/1", "policy": self.policy_name,
                **self._L.physical_report(self._L.executed_until)}

    def unavailability(self, n_objects=None):
        raise NotImplementedError("Для I4.1 используйте metrics(): недоступность на всём физическом горизонте")

    def security_report(self, scenario, queries):
        raise NotImplementedError("SecurityReplay требует отдельной адаптации к физическим копиям")

    def adversary_report(self):
        raise NotImplementedError("Модель противника ещё не адаптирована к физическим копиям")

    def crisis_report(self):
        raise NotImplementedError("Кризисный отчёт ещё не адаптирован к физическим копиям")


class PhysicalLedger(Ledger):
    def _reset_state(self):
        super()._reset_state()
        self.physical_nodes = [set() for _ in range(self.N)]
        self.allocated = []
        self.references = [Counter() for _ in self.obj_frags]
        self.reserved = np.zeros(self.N, int)
        self.physical_events = []
        self.declarations = []
        self.returns = []
        self.operation_outcomes = []
        self.repair_rejections = Counter()
        self.cleanup = []
        self.silent_announced = set()
        self.initial_writes = []
        self.initial_write_rejections = []

    def _world_events(self):
        super()._world_events()
        for node, (_, ends) in self.obs_iv.items():
            for t in ends:
                if t < self.T:
                    self._push(float(t), "observed_return", int(node))

    def _add_fragment(self, obj, idx, node, t):
        if self.used[node] >= self.capacity[node]:
            raise ValueError("Запись сверх ёмкости")
        if not self.registered_by(t)[node] or self.alive_to[node] <= t:
            raise ValueError("Запись на незарегистрированный или ушедший узел")
        if not self.is_up(node, t):
            raise ValueError("Запись на недоступный узел")
        super()._add_fragment(obj, idx, node, t)
        f = len(self.f_obj) - 1
        self.allocated.append(True)
        self.physical_nodes[node].add(f)
        self.references[obj][idx] += 1
        self.obj_exist[obj] = len(self.references[obj])
        self.physical_events.append((float(t), 1))

    def _destroy(self, f, t):
        if not self.f_exist[f]:
            return
        self.f_exist[f] = False
        self.f_dead[f] = float(t)
        obj, idx = self.f_obj[f], self.f_idx[f]
        self.references[obj][idx] -= 1
        if self.references[obj][idx] == 0:
            del self.references[obj][idx]
        self.obj_exist[obj] = len(self.references[obj])
        self.physical_events.append((float(t), -1))
        if self.obj_exist[obj] < self.cfg.k and self.obj_lost_t[obj] == np.inf:
            self.obj_lost_t[obj] = float(t)
            self.log_loss.append((obj, float(t)))

    def _free_slot(self, f, t):
        if self.allocated[f]:
            self._destroy(f, t)
            self.allocated[f] = False
            self.used[self.f_node[f]] -= 1
            self.physical_nodes[self.f_node[f]].discard(f)

    def _declare(self, f, t, reason):
        if reason.startswith("тихая") and f not in self.silent_announced:
            self.log_silent.append((t, self.f_node[f]))
            self.silent_announced.add(f)
        if self.f_to[f] != np.inf:
            if not self.f_exist[f]:
                self._free_slot(f, t)
            return
        node, obj, idx = self.f_node[f], self.f_obj[f], self.f_idx[f]
        self.f_to[f], self.f_end[f] = float(t), reason
        self.node_frags[node].discard(f)
        self.obj_frags[obj].discard(f)
        self.declarations.append((float(t), int(obj), int(idx), int(node), bool(self.f_exist[f])))
        if not self.f_exist[f]:
            self._free_slot(f, t)
        if self.obj_lost_t[obj] == np.inf and not any(
                self.f_idx[g] == idx for g in self.obj_frags[obj]):
            self.missing[obj].add(idx)
            self.need_repair.add(obj)

    def _observed_up(self, node, t):
        starts, ends = self.obs_iv.get(node, (np.empty(0), np.empty(0)))
        j = np.searchsorted(starts, t, side="right") - 1
        return not (j >= 0 and t < ends[j])

    def _return_node(self, node, t):
        if (not self._observed_up(node, t) or not self.is_up(node, t)
                or self.alive_to[node] <= t):
            return
        for f in sorted(self.physical_nodes[node]):
            if self.f_to[f] == np.inf:
                continue
            obj, idx = self.f_obj[f], self.f_idx[f]
            if not self.f_exist[f]:
                self._free_slot(f, t)
                continue
            active = [g for g in self.obj_frags[obj] if self.f_idx[g] == idx and self.f_exist[g]]
            if active:
                self._free_slot(f, t)
                self.cleanup.append((float(t), int(f), "замена уже существует"))
            elif self.obj_lost_t[obj] == np.inf:
                # Положительная проверка вернувшейся копии; истина не идёт в Experience.
                if not self.is_up(node, t) or self.alive_to[node] <= t:
                    continue
                self.f_to[f], self.f_end[f] = np.inf, ""
                self.node_frags[node].add(f)
                self.obj_frags[obj].add(f)
                self.missing[obj].discard(idx)
                self.returns.append((float(t), int(obj), int(idx), int(node)))
                if not (self.missing[obj] - self.pending[obj]):
                    self.need_repair.discard(obj)

    def _readable(self, f, t):
        return self.f_exist[f] and self.alive_to[self.f_node[f]] > t and self.is_up(self.f_node[f], t)

    def _repair(self, t, reqs):
        for obj, indices, targets in reqs:
            if self.obj_lost_t[obj] != np.inf:
                self.repair_rejections["object_lost"] += 1
                continue
            by_index = {}
            for f in sorted(self.obj_frags[obj]):
                if self._readable(f, t):
                    by_index.setdefault(self.f_idx[f], f)
            if len(by_index) < self.cfg.k:
                self.repair_rejections["sources_unavailable"] += 1
                continue
            selected, nodes = [], []
            held = {self.f_node[f] for f in self.obj_frags[obj]}
            for idx, node in zip(indices, targets):
                idx, node = int(idx), int(node)
                if (idx not in self.missing[obj] or idx in self.pending[obj] or idx in selected
                        or node in held or node in nodes or node < 0 or node >= self.N
                        or not self.registered_by(t)[node] or self.alive_to[node] <= t
                        or not self.is_up(node, t) or self.used[node] >= self.capacity[node]):
                    self.repair_rejections["invalid_or_full_target"] += 1
                    continue
                selected.append(idx)
                nodes.append(node)
                self.used[node] += 1
                self.reserved[node] += 1
            if not selected:
                continue
            sources = list(by_index.values())[:self.cfg.k]
            traffic = (self.cfg.k + len(selected)) * self.size[obj] / self.cfg.k
            self.log_repair.append((obj, t, len(selected), traffic))
            self.pending[obj].update(selected)
            self._push(t + self._operation_delay(traffic), "physical_repair_done", (obj, selected, nodes, sources))

    def _repair_done(self, t, obj, indices, nodes, sources):
        sources_ok = (self.obj_lost_t[obj] == np.inf
                      and len({self.f_idx[f] for f in sources if self._readable(f, t)}) >= self.cfg.k)
        for idx, node in zip(indices, nodes):
            self.pending[obj].discard(idx)
            self.reserved[node] -= 1
            self.used[node] -= 1
            already = any(self.f_idx[f] == idx and self.f_exist[f] for f in self.obj_frags[obj])
            ok = (sources_ok and not already and self.alive_to[node] > t and self.is_up(node, t)
                  and self.used[node] < self.capacity[node])
            status = "written" if ok else "already_returned" if already else "failed"
            if ok:
                self._add_fragment(obj, idx, node, t)
                self.missing[obj].discard(idx)
            self.operation_outcomes.append((float(t), int(obj), int(idx), int(node), status))
        if not (self.missing[obj] - self.pending[obj]):
            self.need_repair.discard(obj)
        # Удалять старые копии на недоступных узлах нельзя.
        for node in range(self.N):
            if self._observed_up(node, t) and self.is_up(node, t):
                self._return_node(node, t)

    def _process(self, t_end):
        while self.heap and self.heap[0][0] < t_end:
            t, _, kind, p = heapq.heappop(self.heap)
            if kind == "destroy_node":
                for f in sorted(self.physical_nodes[p]):
                    self._destroy(f, t)
            elif kind == "declare_node":
                for f in sorted(self.node_frags[p]):
                    self._declare(f, t, "простой > T_rep")
            elif kind == "observed_return":
                self._return_node(p, t)
            elif kind == "silent":
                node, key = p
                fr = sorted(f for f in self.physical_nodes[node] if self.f_exist[f])
                if fr:
                    g = R.stream(self.cfg.seed, R.SILENT, 10**6 + key)
                    f = fr[int(g.integers(len(fr)))]
                    self._destroy(f, t)
                    at = self.audit_ok.get(node, np.empty(0))
                    j = np.searchsorted(at, t, side="right")
                    if j < len(at):
                        self._push(at[j], "declare_frag", f)
            elif kind == "declare_frag":
                self._declare(p, t, "тихая потеря (аудит)")
            elif kind == "physical_repair_done":
                self._repair_done(t, *p)
            else:
                raise ValueError("Неподдерживаемое событие PhysicalLedger: " + kind)

    def _migrate(self, t, reqs):
        if reqs:
            raise NotImplementedError("Миграции будут отдельным этапом PhysicalLedger")

    def _release(self, t, reqs):
        if reqs:
            raise NotImplementedError("Высвобождение будет отдельным этапом PhysicalLedger")

    def run(self, policy, epochs=None, progress=False):
        if getattr(policy, "fast", False):
            raise NotImplementedError("Подшаги тревог не включены в I4.1")
        self._reset_state()
        self._world_events()
        self.ch.reset()
        view = NetworkView(self)
        end = self.T if epochs is None else epochs
        if not isinstance(end, int) or not 0 < end <= self.T:
            raise ValueError("Некорректный горизонт исполнения")
        daily, previous, created = [], None, 0
        for epoch in range(end):
            t = float(epoch)
            step = self.ch.step()
            view.t, view.seed, view.step = t, bytes.fromhex(step["seed"]), previous
            previous = step
            for alarm in policy.alarm(t, view):
                self.log_alarm.append((t, alarm["type"], alarm["ctype"], alarm["cluster"],
                    float(alarm.get("prob", 1.)), float(alarm.get("horizon", 1.))))
            requests = policy.on_epoch(t, view)
            asked = {r[0] for r in requests if r[1] and r[2]}
            self.stalled += sum(1 for obj in view.needs_repair() if obj not in asked and view.missing(obj))
            self._repair(t, requests)
            self._migrate(t, policy.migrate(t, view))
            self._release(t, policy.release(t, view))
            for obj in self.data.new_objects(epoch):
                obj = int(obj)
                target = getattr(policy, "target_fragments", None)
                want = target(obj, t, view) if callable(target) else getattr(policy, "n", None)
                nodes = list(map(int, policy.place(obj, t, view)))
                self.obj_n[obj] = len(nodes)
                written, written_indices = [], set()
                for idx, node in enumerate(nodes):
                    if (node < 0 or node >= self.N or node in written
                            or not self.registered_by(t)[node] or self.alive_to[node] <= t
                            or not self.is_up(node, t) or self.used[node] >= self.capacity[node]):
                        self.initial_write_rejections.append((obj, idx, node, t))
                        continue
                    self._add_fragment(obj, idx, node, t)
                    written.append(node)
                    written_indices.add(idx)
                self.initial_writes.append((obj, t, nodes, written))
                if want is not None and len(written) < want:
                    self.log_short.append((obj, t, len(written), int(want)))
                if len(written) < self.cfg.k:
                    self.obj_lost_t[obj] = t
                    self.log_loss.append((obj, t))
                else:
                    self.missing[obj].update(set(range(len(nodes))) - written_indices)
                    if self.missing[obj]:
                        self.need_repair.add(obj)
                created += 1
            self._process(t + 1.)
            daily.append((epoch, int(self.used.sum()), created - len(self.log_loss)))
            if progress and epoch % 100 == 0:
                print(f"эпоха {epoch}: физических потерь {len(self.log_loss)}", flush=True)
        self.executed_until = float(end)
        return PhysicalRunResult(self, policy, pd.DataFrame(daily, columns=["epoch", "fragments", "objects_alive"]))

    def physical_report(self, end):
        """Истинная метрика после прогона; не вход политики и не старый RunResult.metrics."""
        exposure = unavailable = 0.0
        for obj, row in self.data.objects.iterrows():
            start = float(row.created_epoch)
            if start >= end:
                continue
            exposure += end - start
            changes = defaultdict(Counter)
            changes[start]  # объект без единого фрагмента тоже недоступен
            for f, owner in enumerate(self.f_obj):
                if owner != obj:
                    continue
                a, b = max(start, self.f_from[f]), min(end, self.f_dead[f])
                if a >= b:
                    continue
                intervals = [(a, b)]
                ds, de = self.down.get(self.f_node[f], ([], []))
                for s, e in zip(ds, de):
                    intervals = [(x, min(y, s)) for x, y in intervals if x < min(y, s)] + [
                        (max(x, e), y) for x, y in intervals if max(x, e) < y]
                for x, y in intervals:
                    changes[x][self.f_idx[f]] += 1
                    changes[y][self.f_idx[f]] -= 1
            counts, previous = Counter(), start
            for t in sorted(changes):
                if sum(v > 0 for v in counts.values()) < self.cfg.k:
                    unavailable += t - previous
                counts.update(changes[t])
                previous = t
            if sum(v > 0 for v in counts.values()) < self.cfg.k:
                unavailable += end - previous
        fragment_days = sum(delta * max(0., end - t) for t, delta in self.physical_events if t < end)
        incomplete = {r[0] for r in self.initial_writes if len(r[3]) < self.cfg.k and r[1] < end}
        return {"objects": int((self.data.objects.created_epoch < end).sum()),
                "physical_lost_objects": int((self.obj_lost_t < end).sum()),
                "initial_incomplete_objects": len(incomplete),
                "lost_admitted_objects": sum(1 for obj, t in self.log_loss if t < end and obj not in incomplete),
                "initial_failed_writes": len(self.initial_write_rejections),
                "object_days": exposure, "unavailable_object_days": unavailable,
                "unavailability_share": unavailable / exposure if exposure else None,
                "physical_fragment_days": fragment_days,
                "declarations": len(self.declarations), "live_at_declaration": sum(r[4] for r in self.declarations),
                "returns": len(self.returns), "redundant_deleted": len(self.cleanup),
                "repair_batches_started": len(self.log_repair),
                "repair_traffic": float(sum(r[3] for r in self.log_repair)),
                "repair_outcomes": dict(Counter(r[4] for r in self.operation_outcomes)),
                "repair_rejections": dict(self.repair_rejections),
                "pending_writes": int(self.reserved.sum()),
                "allocated_slots": int(self.used.sum()),
                "physical_copies": int(sum(self.f_exist)),
                "active_map_records": int(sum(map(len, self.obj_frags)))}
