"""I6: отдельные правила доступа служебных proof к очереди и каналу."""
from dataclasses import dataclass
from .disk_stress import DiskStressLedger
from .targeted_proof import TargetedProofResult


@dataclass(frozen=True)
class QueueControlSettings:
    reserve_proof_slot: bool = False
    bypass_read_wait: bool = False


class QueueControlResult(TargetedProofResult):
    def metrics(self):
        return {"schema": "taris-queue-control-metrics/1", "policy": self.policy_name,
                **self._L.queue_report(self._L.executed_until)}


class QueueControlLedger(DiskStressLedger):
    def __init__(self, world, chain, data, queue_settings=None, transport_settings=None, waves=(), control=None):
        self.control = control or QueueControlSettings()
        if type(self.control.reserve_proof_slot) is not bool or type(self.control.bypass_read_wait) is not bool:
            raise ValueError("Флаги правил должны быть bool")
        super().__init__(world, chain, data, queue_settings, transport_settings, waves)
        if self.control.reserve_proof_slot and self.queue_settings.max_jobs < 2:
            raise ValueError("Резерв proof требует хотя бы двух мест")

    def _reset_state(self):
        super()._reset_state()
        self.repair_admissions = []
        self.read_wait_pauses = []
        self.proof_scan_seen = None

    def _return_node(self, node, t):
        # Один проход проверки возвратов на внешнее решение, как у контроля.
        seen = getattr(self, "proof_scan_seen", None)
        if seen is not None:
            if node in seen:
                return
            seen.add(node)
        return super()._return_node(node, t)

    def _enqueue(self, job, t):
        if self.control.reserve_proof_slot and job["kind"] == "repair":
            repairs = sum(j["kind"] == "repair" and j["finished"] is None for j in self.jobs)
            if repairs >= self.queue_settings.max_jobs-1:
                raise AssertionError("Нарушен резерв служебного места")
        return super()._enqueue(job, t)

    def _repair(self, t, reqs):
        if not self.control.reserve_proof_slot:
            return super()._repair(t, reqs)
        self.proof_scan_seen = set()
        try:
            super()._repair(t, [])
            for obj, indices, nodes in reqs:
                repairs = sum(j["kind"] == "repair" and j["finished"] is None for j in self.jobs)
                owner = next((key for key in (self.active_job, self.parked_job) if key is not None
                    and self.jobs[key]["kind"] == "repair" and self.jobs[key]["obj"] == obj
                    and obj in self.buffers), None)
                permitted = owner is not None or repairs < self.queue_settings.max_jobs-1
                self.repair_admissions.append({"time": float(t), "obj": int(obj), "repair_jobs_before": repairs,
                    "full_buffer_owner": owner, "permitted_for_processing": permitted})
                if not permitted:
                    self.queue_rejections["repair_reserved_for_proof"] += 1
                    continue
                super()._repair(t, [(obj, indices, nodes)])
        finally:
            self.proof_scan_seen = None

    def _schedule_retry(self, job, t, reason):
        if (not self.control.bypass_read_wait or reason != "waiting_read_sources"
                or job["obj"] in self.buffers):
            return super()._schedule_retry(job, t, reason)
        assert self.parked_job is None and self.active_job == job["id"]
        # Только наблюдаемая карта; текущую истинную целостность не читаем.
        known_indices = sorted({int(self.f_idx[f]) for f in self.obj_frags[job["obj"]]})
        known_missing = sorted(map(int, self.missing[job["obj"]]))
        self.parked_job, self.active_job = job["id"], None
        self.wake_ready, self.proof_quota_used = False, False
        wake = t+self.transport.retry_hours/24.
        job["status"] = reason
        self.parked_intervals.append({"job": job["id"], "start": float(t), "end": None,
                                     "wake": float(wake), "pause_type": "read_wait"})
        self.read_wait_pauses.append({"time": float(t), "job": job["id"], "obj": job["obj"],
            "known_indices": known_indices, "known_missing": known_missing,
            "known_holders": [{"copy": int(f), "idx": int(self.f_idx[f]), "node": int(self.f_node[f])}
                              for f in sorted(self.obj_frags[job["obj"]])],
            "read_progress": job.get("read_progress", 0)})
        self.scheduler_events.append({"kind": "park", "job": job["id"], "time": float(t), "wake": float(wake)})
        self._push(wake, "shared_wakeup", (job["id"], len(self.parked_intervals)-1))
        self._pump(t)

    def queue_report(self, end):
        report = super().queue_report(end)
        report["parked_owner_days"] = report["parked_buffer_days"]
        read_days = sum(max(0., min(end, p["end"] if p["end"] is not None else end)-p["start"])
                        for p in self.parked_intervals if p.get("pause_type") == "read_wait")
        report["parked_read_wait_days"] = read_days
        report["parked_buffer_days"] -= read_days
        report["read_wait_pause_count"] = len(self.read_wait_pauses)
        return report

    def run(self, policy, epochs=None, progress=False):
        result = super().run(policy, epochs, progress)
        return QueueControlResult(self, policy, result.daily)
