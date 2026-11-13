"""I4.4b: одна proof-проверка в вынужденную паузу владельца буфера."""
import heapq
import numpy as np
from .cancel_obsolete import CancellationLedger, CancellationResult


class SharedChannelResult(CancellationResult):
    def metrics(self):
        return {"schema": "taris-shared-channel-metrics/1", "policy": self.policy_name,
                **self._L.queue_report(self._L.executed_until)}


class SharedChannelLedger(CancellationLedger):
    def _reset_state(self):
        super()._reset_state()
        self.parked_job = None
        self.routing_active_job = None
        self.wake_ready = False
        self.proof_quota_used = False
        self.parked_intervals = []
        self.scheduler_events = []

    def _outstanding(self):
        owners = {j for j in (self.active_job, self.parked_job, self.routing_active_job) if j is not None}
        return len(self.waiting)+len(owners)

    def _schedule_retry(self, job, t, reason):
        if job["obj"] not in self.buffers:
            return super()._schedule_retry(job, t, reason)
        assert self.parked_job is None and self.active_job == job["id"]
        self.parked_job, self.active_job = job["id"], None
        self.wake_ready, self.proof_quota_used = False, False
        wake = t+self.transport.retry_hours/24.
        job["status"] = reason
        self.parked_intervals.append({"job": job["id"], "start": float(t), "end": None, "wake": float(wake)})
        self.scheduler_events.append({"kind": "park", "job": job["id"], "time": float(t), "wake": float(wake)})
        self._push(wake, "shared_wakeup", (job["id"], len(self.parked_intervals)-1))
        self._pump(t)

    def _pump(self, t):
        if self.active_job is not None:
            return
        if self.parked_job is None:
            return super()._pump(t)
        if self.wake_ready or t >= self.parked_intervals[-1]["wake"]:
            key = self.parked_job
            self.parked_job, self.active_job = None, key
            self.parked_intervals[-1]["end"] = float(t)
            self.scheduler_events.append({"kind": "resume", "job": key, "time": float(t)})
            self._next(self.jobs[key], t)
            return
        if self.proof_quota_used:
            return
        key = next((key for key in self.waiting if self.jobs[key]["kind"] == "proof"), None)
        if key is None:
            return
        self.waiting.remove(key)
        self.waiting.appendleft(key)
        self.proof_quota_used = True
        self.scheduler_events.append({"kind": "proof_bypass", "job": key, "owner": self.parked_job, "time": float(t)})
        super()._pump(t)

    def _repair(self, t, reqs):
        if self.parked_job is None:
            return super()._repair(t, reqs)
        # Маршрутизация запроса владельцу памяти не меняет реальную занятость канала.
        real_active = self.active_job
        self.routing_active_job = real_active
        self.active_job = self.parked_job
        try:
            super()._repair(t, reqs)
        finally:
            self.active_job = real_active
            self.routing_active_job = None
        self._pump(t)

    def _close_buffer(self, obj, t, reason):
        if reason == "buffer_failure" and self.parked_job is not None:
            job = self.jobs[self.parked_job]
            if job["obj"] == obj:
                self._close_prefixes(self.decoded_prefixes, job["id"], t)
                job["read_progress"] = 0
        super()._close_buffer(obj, t, reason)

    def _process(self, t_end):
        while self.heap and self.heap[0][0] < t_end:
            event = heapq.heappop(self.heap)
            t, _, kind, payload = event
            if kind == "shared_wakeup":
                key, pause = payload
                if (pause == len(self.parked_intervals)-1 and self.parked_job == key
                        and self.jobs[key]["finished"] is None):
                    self.wake_ready = True
                    self.scheduler_events.append({"kind": "wake_ready", "job": key, "time": float(t)})
                    self._pump(t)
                continue
            if kind == "destroy_node" and self.parked_job is not None:
                job = self.jobs[self.parked_job]
                pos = job.get("cursor", 0)
                if pos < len(job["nodes"]) and job["nodes"][pos] == payload:
                    self._close_prefixes(self.receiver_prefixes, job["id"], t)
                    job["write_progress"] = 0
            remaining, self.heap = self.heap, [event]
            super()._process(np.nextafter(float(t), np.inf))
            generated, self.heap = self.heap, remaining
            for item in generated:
                heapq.heappush(self.heap, item)

    def run(self, policy, epochs=None, progress=False):
        result = super().run(policy, epochs, progress)
        return SharedChannelResult(self, policy, result.daily)

    def queue_report(self, end):
        report = super().queue_report(end)
        report["parked_jobs"] = int(self.parked_job is not None)
        report["proof_bypass_starts"] = sum(r["kind"] == "proof_bypass" for r in self.scheduler_events)
        report["parked_buffer_days"] = sum(max(0., min(end, r["end"] if r["end"] is not None else end)-r["start"])
                                           for r in self.parked_intervals)
        report["resume_delay_days"] = sum(max(0., r["end"]-r["wake"]) for r in self.parked_intervals if r["end"] is not None)
        return report
