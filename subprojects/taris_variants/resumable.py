"""I4.3: последовательные проверенные префиксы и удержание полного буфера."""
from dataclasses import dataclass
from collections import defaultdict
import heapq
import numpy as np
from .queued import QueuedLedger, QueuedRunResult
from .physical import PhysicalLedger


@dataclass(frozen=True)
class TransportSettings:
    stripes: int = 16
    retry_hours: float = 1.

    def validate(self):
        if (type(self.stripes) is not int or self.stripes < 1
                or not np.isfinite(self.retry_hours) or self.retry_hours <= 0):
            raise ValueError("Некорректные параметры передачи префиксов")


class ResumableResult(QueuedRunResult):
    def metrics(self):
        return {"schema": "taris-resumable-metrics/1", "policy": self.policy_name,
                **self._L.queue_report(self._L.executed_until)}


class ResumableLedger(QueuedLedger):
    def __init__(self, world, chain, data, queue_settings=None, transport_settings=None):
        self.transport = transport_settings or TransportSettings()
        self.transport.validate()
        super().__init__(world, chain, data, queue_settings)

    def _reset_state(self):
        super()._reset_state()
        self.disk_generation = np.zeros(self.N, int)
        self.memory_generation = np.zeros(len(self.obj_frags), int)
        self.decoded_prefixes = []
        self.receiver_prefixes = []
        self.block_results = []
        self.close_checks = []
        self.target_additions = []

    def _schedule_retry(self, job, t, reason):
        job["status"] = reason
        self._push(t + self.transport.retry_hours/24., "resume_poll", job["id"])

    def _close_prefixes(self, rows, key, t):
        for row in rows:
            if row["job"] == key and row["end"] is None:
                row["end"] = float(t)

    def _close_buffer(self, obj, t, reason):
        if reason == "buffer_failure":
            self.memory_generation[obj] += 1
            if self.active_job is not None:
                job = self.jobs[self.active_job]
                if job["kind"] == "repair" and job["obj"] == obj:
                    self._close_prefixes(self.decoded_prefixes, job["id"], t)
                    job["read_progress"] = 0
        super()._close_buffer(obj, t, reason)

    def _setup_done(self, key, t):
        job = self.jobs[key]
        if job["kind"] == "proof":
            return super()._setup_done(key, t)
        job.update(read_progress=0, cursor=0, write_progress=0, write_generation=None)
        self._next(job, t)

    def _segment_with(self, job, t, phase, volume, **metadata):
        self._segment(job, t, phase, volume)
        self.segments[-1].update(metadata)

    def _next(self, job, t):
        obj, count = job["obj"], self.transport.stripes
        if self.obj_lost_t[obj] != np.inf:
            self._finish(job, t, "object_lost")
            return
        if obj not in self.buffers:
            by_index = {}
            for f in sorted(self.obj_frags[obj]):
                if self._readable(f, t):
                    by_index.setdefault(self.f_idx[f], f)
            if len(by_index) < self.cfg.k:
                self._schedule_retry(job, t, "waiting_read_sources")
                return
            sources = list(by_index.values())[:self.cfg.k]
            if not job.get("read_started"):
                self.log_repair.append((obj, t, len(job["indices"]), self.size[obj]*(1+len(job["indices"])/self.cfg.k)))
                job["read_started"] = True
            self._segment_with(job, t, "read_block", self.size[obj]/count,
                stripe=job["read_progress"], sources=sources, memory_generation=int(self.memory_generation[obj]))
            return
        while job["cursor"] < len(job["indices"]):
            pos = job["cursor"]
            idx, node = job["indices"][pos], job["nodes"][pos]
            if any(self.f_idx[f] == idx and self.f_exist[f] for f in self.obj_frags[obj]):
                self.reserved[node] -= 1
                self.used[node] -= 1
                self.pending[obj].discard(idx)
                self.missing[obj].discard(idx)
                self.operation_outcomes.append((float(t), obj, idx, node, "already_returned"))
                job["cursor"] += 1
                job["write_progress"] = 0
                continue
            if job["write_generation"] != int(self.disk_generation[node]):
                self._close_prefixes(self.receiver_prefixes, job["id"], t)
                job["write_progress"] = 0
                job["write_generation"] = int(self.disk_generation[node])
            if not self._readable_target(node, t):
                since = job.setdefault("target_wait_since", float(t))
                if t-since < self.cfg.T_rep:
                    self._schedule_retry(job, t, "waiting_write_target")
                else:
                    # Длительно недоступную цель можно отменить лишь после проверки кода.
                    self._request_close(job, t)
                return
            job.pop("target_wait_since", None)
            self._segment_with(job, t, "write_block", self.size[obj]/self.cfg.k/count,
                stripe=job["write_progress"], idx=idx, node=node,
                disk_generation=int(self.disk_generation[node]), memory_generation=int(self.memory_generation[obj]))
            return
        self._request_close(job, t)

    def _readable_target(self, node, t):
        return self.alive_to[node] > t and self.is_up(node, t) and self._observed_up(node, t)

    def _request_close(self, job, t):
        job["status"] = "close_setup"
        self._push(t+self.queue_settings.proof_setup_hours/24., "resume_close_ready", job["id"])

    def _close_ready(self, job, t):
        obj = job["obj"]
        if obj not in self.buffers:
            self._next(job, t)
            return
        by_index = {}
        for f in sorted(self.obj_frags[obj]):
            if self._observed_up(self.f_node[f], t):
                by_index.setdefault(self.f_idx[f], f)
        if len(by_index) < self.cfg.k:
            self._schedule_retry(job, t, "retained_buffer")
            return
        self._segment_with(job, t, "close_proof", self.queue_settings.proof_fraction*self.size[obj],
                           sources=list(by_index.values())[:self.cfg.k])

    def _phase_done(self, key, t):
        job = self.jobs[key]
        if job["kind"] == "proof":
            return super()._phase_done(key, t)
        segment = self.segments[-1]
        assert segment["job"] == key
        obj, phase, a = job["obj"], segment["phase"], segment["start"]
        if phase == "close_proof":
            ok = obj in self.buffers and all(self.f_exist[f] and self._continuous_up(self.f_node[f], a, t)
                                            for f in segment["sources"])
            self.close_checks.append({"job": key, "time": float(t), "ok": bool(ok), "sources": segment["sources"]})
            if ok:
                self._close_buffer(obj, t, "confirmed_disk_code")
                status = "write_completed" if job["cursor"] == len(job["indices"]) else "closed_with_unfinished_targets"
                self._finish(job, t, status)
            else:
                # Отрицательная проверка целостности на доступном узле — наблюдаемое событие.
                for f in segment["sources"]:
                    if not self.f_exist[f] and self._readable_target(self.f_node[f], t):
                        self._declare(f, t, "тихая потеря (аудит)")
                self._schedule_retry(job, t, "retained_buffer")
            return
        memory_ok = segment["memory_generation"] == int(self.memory_generation[obj])
        if phase == "read_block":
            ok = memory_ok and self.obj_lost_t[obj] == np.inf and all(
                self.f_exist[f] and self._continuous_up(self.f_node[f], a, t) for f in segment["sources"])
            if ok:
                assert job["read_progress"] == segment["stripe"]
                job["read_progress"] += 1
                self.decoded_prefixes.append({"job": key, "obj": obj, "stripe": segment["stripe"],
                    "start": float(t), "end": None, "size": float(self.size[obj]/self.transport.stripes)})
                if job["read_progress"] == self.transport.stripes:
                    self.buffers[obj] = len(self.buffer_intervals)
                    self.buffer_intervals.append({"obj": obj, "job": key, "start": float(t), "end": None,
                                                  "size": float(self.size[obj]), "reason": None})
                    self._close_prefixes(self.decoded_prefixes, key, t)
            self.block_results.append({"job": key, "segment": len(self.segments)-1, "ok": bool(ok)})
        elif phase == "write_block":
            ok = (memory_ok and obj in self.buffers and segment["disk_generation"] == int(self.disk_generation[segment["node"]])
                  and self._continuous_up(segment["node"], a, t))
            if ok:
                assert job["write_progress"] == segment["stripe"]
                job["write_progress"] += 1
                self.receiver_prefixes.append({"job": key, "obj": obj, "idx": segment["idx"], "node": segment["node"],
                    "stripe": segment["stripe"], "start": float(t), "end": None,
                    "size": float(self.size[obj]/self.cfg.k/self.transport.stripes)})
                if job["write_progress"] == self.transport.stripes:
                    node, idx = segment["node"], segment["idx"]
                    self.reserved[node] -= 1
                    self.used[node] -= 1
                    self.pending[obj].discard(idx)
                    self._add_fragment(obj, idx, node, t)
                    self.missing[obj].discard(idx)
                    self.operation_outcomes.append((float(t), obj, idx, node, "written"))
                    self._close_prefixes(self.receiver_prefixes, key, t)
                    job["cursor"] += 1
                    job["write_progress"], job["write_generation"] = 0, None
            self.block_results.append({"job": key, "segment": len(self.segments)-1, "ok": bool(ok)})
        else:
            raise ValueError("Неизвестная фаза " + phase)
        if ok:
            self._next(job, t)
        else:
            self._schedule_retry(job, t, "retry_block")

    def _repair(self, t, reqs):
        remaining = []
        active = self.jobs[self.active_job] if self.active_job is not None else None
        for obj, indices, nodes in reqs:
            if active is None or active["kind"] != "repair" or active["obj"] != obj or obj not in self.buffers:
                remaining.append((obj, indices, nodes))
                continue
            held = {self.f_node[f] for f in self.obj_frags[obj]}
            for idx, node in zip(indices, nodes):
                idx, node = int(idx), int(node)
                if (idx not in self.missing[obj] or idx in self.pending[obj] or node in held
                        or not 0 <= node < self.N or not self.registered_by(t)[node]
                        or not self._readable_target(node, t) or self.used[node] >= self.capacity[node]
                        or node in active["nodes"][active.get("cursor", 0):]):
                    continue
                active["indices"].append(idx)
                active["nodes"].append(node)
                self.target_additions.append({"time": float(t), "job": active["id"],
                    "position": len(active["indices"])-1, "idx": idx, "node": node})
                self.pending[obj].add(idx)
                self.used[node] += 1
                self.reserved[node] += 1
        super()._repair(t, remaining)

    def _finish(self, job, t, status):
        if job["kind"] == "repair":
            assert job["obj"] not in self.buffers, "Нельзя выбросить последний полный буфер при завершении задания"
            self._close_prefixes(self.decoded_prefixes, job["id"], t)
            self._close_prefixes(self.receiver_prefixes, job["id"], t)
            # Уже записанные и пропущенные цели освободили резервации раньше.
            if not job.get("reservations_released"):
                for idx, node in zip(job["indices"][job.get("cursor", 0):], job["nodes"][job.get("cursor", 0):]):
                    self.used[node] -= 1
                    self.reserved[node] -= 1
                    self.pending[job["obj"]].discard(idx)
                job["reservations_released"] = True
        super()._finish(job, t, status)

    def _process(self, t_end):
        while self.heap and self.heap[0][0] < t_end:
            event = heapq.heappop(self.heap)
            t, _, kind, payload = event
            if kind in ("queue_setup_done", "queue_phase_done", "resume_poll", "resume_close_ready"):
                job = self.jobs[payload]
                if job["finished"] is not None:
                    continue
                assert self.active_job == payload
                if kind == "queue_setup_done":
                    self._setup_done(payload, t)
                elif kind == "queue_phase_done":
                    self._phase_done(payload, t)
                elif kind == "resume_close_ready":
                    self._close_ready(job, t)
                else:
                    self._next(job, t)
            elif kind == "queue_buffer_fail":
                self._close_buffer(int(payload), t, "buffer_failure")
            else:
                if kind == "destroy_node":
                    self.disk_generation[payload] += 1
                    if self.active_job is not None:
                        job = self.jobs[self.active_job]
                        if job["kind"] == "repair" and job.get("cursor", 0) < len(job["nodes"]):
                            if job["nodes"][job.get("cursor", 0)] == payload:
                                self._close_prefixes(self.receiver_prefixes, job["id"], t)
                                job["write_progress"] = 0
                remaining, self.heap = self.heap, [event]
                self.handling_world_event = True
                PhysicalLedger._process(self, np.nextafter(float(t), np.inf))
                self.handling_world_event = False
                generated, self.heap = self.heap, remaining
                for item in generated:
                    heapq.heappush(self.heap, item)
                self._pump(t)

    def run(self, policy, epochs=None, progress=False):
        result = super().run(policy, epochs, progress)
        return ResumableResult(self, policy, result.daily)

    def queue_report(self, end):
        report = super().queue_report(end)
        for rows, name in ((self.decoded_prefixes, "decoded_prefix_size_days"),
                           (self.receiver_prefixes, "receiver_prefix_size_days")):
            report[name] = sum(r["size"]*max(0., min(end, r["end"] if r["end"] is not None else end)-r["start"]) for r in rows)
        report["block_attempts"] = len(self.block_results)
        report["failed_blocks"] = sum(not r["ok"] for r in self.block_results)
        report["successful_close_checks"] = sum(r["ok"] for r in self.close_checks)
        report["full_buffers_created"] = len(self.buffer_intervals)
        changes = defaultdict(float)
        for row in self.decoded_prefixes + self.buffer_intervals:
            a, b = row["start"], min(end, row["end"] if row["end"] is not None else end)
            if a < b:
                changes[a] += row["size"]
                changes[b] -= row["size"]
        used = peak = 0.
        for t in sorted(changes):
            used += changes[t]
            peak = max(peak, used)
        report["peak_central_size"] = peak
        report["open_decoded_prefix_size"] = sum(r["size"] for r in self.decoded_prefixes if r["end"] is None)
        report["open_receiver_prefix_size"] = sum(r["size"] for r in self.receiver_prefixes if r["end"] is None)
        return report
