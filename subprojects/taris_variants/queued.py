"""I4.2: один FIFO-канал, полный временный буфер, платные proof-проверки."""
from collections import Counter, deque, defaultdict
from dataclasses import dataclass
import heapq
import numpy as np
from .physical import PhysicalLedger, PhysicalRunResult


@dataclass(frozen=True)
class QueueSettings:
    rate_size_per_hour: float = 2.
    max_jobs: int = 8
    buffer_capacity_size: float = 1.
    proof_fraction: float = .01
    proof_setup_hours: float = .05

    def validate(self):
        if (not np.isfinite(self.rate_size_per_hour) or self.rate_size_per_hour <= 0
                or type(self.max_jobs) is not int or self.max_jobs < 1
                or not np.isfinite(self.buffer_capacity_size) or self.buffer_capacity_size <= 0
                or not np.isfinite(self.proof_fraction) or not 0 < self.proof_fraction <= 1
                or not np.isfinite(self.proof_setup_hours) or self.proof_setup_hours < 0):
            raise ValueError("Некорректные параметры общей очереди")


class QueuedRunResult(PhysicalRunResult):
    def metrics(self):
        return {"schema": "taris-queued-metrics/1", "policy": self.policy_name,
                **self._L.queue_report(self._L.executed_until)}


class QueuedLedger(PhysicalLedger):
    def __init__(self, world, chain, data, queue_settings=None):
        self.queue_settings = queue_settings or QueueSettings()
        self.queue_settings.validate()
        super().__init__(world, chain, data)

    def _reset_state(self):
        super()._reset_state()
        self.jobs = []
        self.waiting = deque()
        self.active_job = None
        self.verification_pending = set()
        self.buffers = {}  # obj -> индекс интервала полного буфера
        self.buffer_intervals = []
        self.segments = []
        self.queue_rejections = Counter()
        self.queue_events = []
        self.handling_world_event = False

    def _destroy(self, f, t):
        obj = self.f_obj[f]
        was_lost = self.obj_lost_t[obj]
        super()._destroy(f, t)
        if was_lost == np.inf and self.obj_lost_t[obj] != np.inf and obj in self.buffers:
            assert self.log_loss[-1] == (obj, float(t))
            self.log_loss.pop()
            self.obj_lost_t[obj] = np.inf

    def _close_buffer(self, obj, t, reason):
        if obj not in self.buffers:
            return
        interval = self.buffer_intervals[self.buffers.pop(obj)]
        interval["end"], interval["reason"] = float(t), reason
        if self.obj_exist[obj] < self.cfg.k and self.obj_lost_t[obj] == np.inf:
            self.obj_lost_t[obj] = float(t)
            self.log_loss.append((obj, float(t)))

    def _continuous_up(self, node, a, b):
        starts, ends = self.down.get(node, (np.empty(0), np.empty(0)))
        return (self.alive_to[node] > b and self.is_up(node, a) and self.is_up(node, b)
                and not bool(np.any((starts < b) & (ends > a))))

    def _outstanding(self):
        return len(self.waiting) + int(self.active_job is not None)

    def _enqueue(self, job, t):
        job.update(id=len(self.jobs), submitted=float(t), started=None, finished=None, status="queued")
        self.jobs.append(job)
        self.waiting.append(job["id"])
        self.queue_events.append((float(t), "enqueue", job["id"], self._outstanding()))

    def _return_node(self, node, t):
        if node in self.verification_pending or not self._observed_up(node, t):
            return
        if not any(self.f_to[f] != np.inf for f in self.physical_nodes[node]):
            return
        if self._outstanding() >= self.queue_settings.max_jobs:
            self.queue_rejections["proof_queue_full"] += 1
            return
        self.verification_pending.add(node)
        self._enqueue({"kind": "proof", "node": int(node)}, t)
        if not self.handling_world_event:
            self._pump(t)

    def _repair(self, t, reqs):
        # Повтор проверок, не принятых в полную очередь или неуспешных ранее.
        for node in range(self.N):
            if self._observed_up(node, t):
                self._return_node(node, t)
        for obj, indices, targets in reqs:
            if self.obj_lost_t[obj] != np.inf:
                self.queue_rejections["object_lost"] += 1
                continue
            if self.size[obj] > self.queue_settings.buffer_capacity_size:
                self.queue_rejections["buffer_too_small"] += 1
                continue
            if self._outstanding() >= self.queue_settings.max_jobs:
                self.queue_rejections["repair_queue_full"] += 1
                continue
            selected, nodes = [], []
            held = {self.f_node[f] for f in self.obj_frags[obj]}
            for idx, node in zip(indices, targets):
                idx, node = int(idx), int(node)
                if (idx not in self.missing[obj] or idx in self.pending[obj] or idx in selected
                        or node in held or node in nodes or not 0 <= node < self.N
                        or not self.registered_by(t)[node] or not self.is_up(node, t)
                        or self.alive_to[node] <= t or self.used[node] >= self.capacity[node]):
                    self.queue_rejections["invalid_or_pending_target"] += 1
                    continue
                selected.append(idx)
                nodes.append(node)
                self.used[node] += 1
                self.reserved[node] += 1
            if selected:
                self.pending[obj].update(selected)
                self._enqueue({"kind": "repair", "obj": int(obj), "indices": selected, "nodes": nodes}, t)
        self._pump(t)

    def _pump(self, t):
        if self.active_job is not None or not self.waiting:
            return
        key = self.waiting.popleft()
        job = self.jobs[key]
        self.active_job, job["started"], job["status"] = key, float(t), "setup"
        hours = self.cfg.repair_hours if job["kind"] == "repair" else self.queue_settings.proof_setup_hours
        self._push(t + hours/24., "queue_setup_done", key)
        self.queue_events.append((float(t), "start", key, self._outstanding()))

    def _segment(self, job, t, phase, volume):
        until = t + volume / self.queue_settings.rate_size_per_hour / 24.
        if not np.isfinite(until) or until <= t:
            raise ArithmeticError("Длительность фазы не представима выбранным шагом времени")
        self.segments.append({"job": job["id"], "phase": phase, "start": float(t),
                              "end": float(until), "volume": float(volume)})
        job["phase_started"], job["status"] = float(t), phase
        self._push(until, "queue_phase_done", job["id"])

    def _setup_done(self, key, t):
        job = self.jobs[key]
        assert self.active_job == key
        if job["kind"] == "proof":
            node = job["node"]
            copies = [f for f in self.physical_nodes[node] if self.f_to[f] != np.inf]
            if not copies or not self.is_up(node, t) or not self._observed_up(node, t):
                self._finish(job, t, "proof_unavailable")
                return
            job["copies"] = sorted(copies)
            volume = self.queue_settings.proof_fraction * sum(self.size[self.f_obj[f]]/self.cfg.k for f in copies)
            self._segment(job, t, "proof", volume)
            return
        obj = job["obj"]
        by_index = {}
        for f in sorted(self.obj_frags[obj]):
            if self._readable(f, t):
                by_index.setdefault(self.f_idx[f], f)
        if self.obj_lost_t[obj] != np.inf or len(by_index) < self.cfg.k:
            self._finish(job, t, "sources_unavailable")
            return
        job["sources"] = list(by_index.values())[:self.cfg.k]
        traffic = self.size[obj] * (1 + len(job["indices"])/self.cfg.k)
        self.log_repair.append((obj, t, len(job["indices"]), traffic))
        self._segment(job, t, "read", self.size[obj])

    def _phase_done(self, key, t):
        job = self.jobs[key]
        assert self.active_job == key
        a = job["phase_started"]
        if job["status"] == "proof":
            node = job["node"]
            if self._continuous_up(node, a, t) and self._observed_up(node, t):
                # Проверяются только копии, за которые назначен proof-бюджет.
                current = self.physical_nodes[node]
                checked = current.intersection(job["copies"])
                other = current - checked
                self.physical_nodes[node] = checked
                PhysicalLedger._return_node(self, node, t)
                self.physical_nodes[node].update(other)
                self._finish(job, t, "proof_completed")
            else:
                self._finish(job, t, "proof_failed")
            return
        obj = job["obj"]
        if job["status"] == "read":
            ok = self.obj_lost_t[obj] == np.inf and all(
                self.f_exist[f] and self._continuous_up(self.f_node[f], a, t) for f in job["sources"])
            if not ok:
                self._finish(job, t, "read_failed")
                return
            assert not self.buffers  # один сервер, один полный буфер
            self.buffers[obj] = len(self.buffer_intervals)
            self.buffer_intervals.append({"obj": obj, "job": key, "start": float(t),
                                          "end": None, "size": float(self.size[obj]), "reason": None})
            self._segment(job, t, "write", len(job["indices"])*self.size[obj]/self.cfg.k)
            return
        assert job["status"] == "write"
        buffer_ok = obj in self.buffers and self.obj_lost_t[obj] == np.inf
        written = 0
        for idx, node in zip(job["indices"], job["nodes"]):
            self.reserved[node] -= 1
            self.used[node] -= 1
            self.pending[obj].discard(idx)
            already = any(self.f_idx[f] == idx and self.f_exist[f] for f in self.obj_frags[obj])
            ok = buffer_ok and not already and self._continuous_up(node, a, t)
            if ok:
                self._add_fragment(obj, idx, node, t)
                self.missing[obj].discard(idx)
                written += 1
            self.operation_outcomes.append((float(t), obj, idx, node,
                                            "written" if ok else "already_returned" if already else "failed"))
        job["reservations_released"] = True
        self._close_buffer(obj, t, "write_attempt_finished")
        self._finish(job, t, "write_completed" if written else "write_without_new_copy")

    def _finish(self, job, t, status):
        if job["kind"] == "repair":
            obj = job["obj"]
            if not job.get("reservations_released"):
                for idx, node in zip(job["indices"], job["nodes"]):
                    self.used[node] -= 1
                    self.reserved[node] -= 1
                    self.pending[obj].discard(idx)
                job["reservations_released"] = True
            if not (self.missing[obj] - self.pending[obj]):
                self.need_repair.discard(obj)
        else:
            self.verification_pending.discard(job["node"])
        job["finished"], job["status"] = float(t), status
        self.active_job = None
        self.queue_events.append((float(t), "finish", job["id"], self._outstanding()))
        self._pump(t)

    def _process(self, t_end):
        while self.heap and self.heap[0][0] < t_end:
            event = heapq.heappop(self.heap)
            t, _, kind, payload = event
            if kind == "queue_setup_done":
                self._setup_done(payload, t)
            elif kind == "queue_phase_done":
                self._phase_done(payload, t)
            elif kind == "queue_buffer_fail":
                self._close_buffer(int(payload), t, "buffer_failure")
            else:
                # Старый обработчик получает ровно одно событие; созданные им события
                # возвращаются в общую кучу с исходными номерами последовательности.
                remaining, self.heap = self.heap, [event]
                self.handling_world_event = True
                super()._process(np.nextafter(float(t), np.inf))
                self.handling_world_event = False
                generated, self.heap = self.heap, remaining
                for item in generated:
                    heapq.heappush(self.heap, item)
                self._pump(t)

    def run(self, policy, epochs=None, progress=False):
        result = super().run(policy, epochs, progress)
        return QueuedRunResult(self, policy, result.daily)

    def queue_report(self, end):
        physical = self.physical_report(end)
        losses = physical.pop("physical_lost_objects")
        scheduled = physical.pop("repair_traffic")
        volumes = Counter()
        for s in self.segments:
            fraction = min(1., max(0., (end-s["start"])/(s["end"]-s["start"])))
            volumes[s["phase"]] += s["volume"] * fraction
        buffer_days = sum(max(0., min(end, r["end"] if r["end"] is not None else end)-r["start"])
                          for r in self.buffer_intervals)
        waiting = [j for j in self.jobs if j["started"] is None]
        waits = [j["started"]-j["submitted"] for j in self.jobs if j["started"] is not None]
        deficit = buffer_only = 0.
        for obj, row in self.data.objects.iterrows():
            created = float(row.created_epoch)
            if created >= end:
                continue
            changes = defaultdict(list)
            changes[created], changes[end]
            for f, owner in enumerate(self.f_obj):
                a, b = self.f_from[f], min(end, self.f_dead[f])
                if owner == obj and a < b:
                    changes[a].append((self.f_idx[f], 1))
                    changes[b].append((self.f_idx[f], -1))
            for interval in self.buffer_intervals:
                if interval["obj"] == obj:
                    a = interval["start"]
                    b = min(end, interval["end"] if interval["end"] is not None else end)
                    if a < b:
                        changes[a].append((None, 1))
                        changes[b].append((None, -1))
            counts, buffers, previous = Counter(), 0, created
            for t in sorted(changes):
                if sum(v > 0 for v in counts.values()) < self.cfg.k:
                    deficit += t-previous
                    buffer_only += (t-previous) * bool(buffers)
                for idx, delta in changes[t]:
                    if idx is None:
                        buffers += delta
                    else:
                        counts[idx] += delta
                previous = t
        return {**physical, "irrecoverable_objects": losses,
            "scheduled_repair_traffic": scheduled,
            "serviced_volume": dict(volumes), "buffer_object_days": buffer_days,
            "disk_deficit_object_days": deficit, "buffer_only_object_days": buffer_only,
            "unfinished_jobs": self._outstanding(), "queued_jobs": len(waiting),
            "open_buffers": len(self.buffers), "queue_rejections": dict(self.queue_rejections),
            "job_statuses": dict(Counter(j["status"] for j in self.jobs)),
            "started_wait_days_sum": float(sum(waits)), "started_jobs": len(waits),
            "pending_wait_days_sum": float(sum(end-j["submitted"] for j in waiting)),
            "max_outstanding_jobs": max((r[3] for r in self.queue_events), default=0)}
