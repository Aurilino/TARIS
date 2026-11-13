"""I4.6a: read-only текущий контракт, без физического аудита и backdating."""
from copy import deepcopy
from dataclasses import asdict
from math import isfinite
from .targeted_proof import TargetedProofLedger


class ObservableQueueLedger(TargetedProofLedger):
    def _reset_state(self):
        super()._reset_state()
        self.observation_clock = 0.
        self.observation_ready = True

    def _process(self, t_end):
        if not isfinite(t_end) or t_end < self.observation_clock:
            raise ValueError("Время наблюдения должно двигаться вперёд")
        self.observation_ready = False
        super()._process(t_end)
        self.observation_clock = float(t_end)
        self.observation_ready = True

    def queue_observation(self, t):
        if not self.observation_ready or not isfinite(t) or t != self.observation_clock:
            raise ValueError("Доступен только завершённый текущий срез исполнения")
        if self.routing_active_job is not None:
            raise ValueError("Нельзя наблюдать временную маршрутизацию")
        jobs = []
        for job in self.jobs:
            if job["finished"] is not None:
                continue
            # Нет status: некоторые внутренние причины завершения используют истину.
            keys = ("id", "kind", "submitted", "started", "obj", "node", "indices", "nodes")
            jobs.append({key: job[key] for key in keys if key in job})
        acknowledgements = []
        for result in self.block_results:
            segment = self.segments[result["segment"]]
            if segment["end"] > t:
                raise ValueError("Ответ блока находится в будущем")
            row = {key: segment[key] for key in
                   ("job", "phase", "stripe", "idx", "node") if key in segment}
            row.update(time=segment["end"], ok=bool(result["ok"]))
            acknowledgements.append(row)
        planned = None
        if self.active_job is not None and self.segments:
            segment = self.segments[-1]
            if segment["job"] == self.active_job and segment["start"] <= t < segment["end"]:
                planned = {key: segment[key] for key in
                           ("job", "phase", "start", "end", "volume", "stripe", "idx", "node")
                           if key in segment}
        return deepcopy({"schema": "taris-queue-observation/1", "time": float(t),
            "jobs": jobs, "waiting": list(self.waiting), "active_job": self.active_job,
            "parked_job": self.parked_job,
            "full_buffer_objects": sorted(map(int, self.buffers)),
            "planned_channel_phase": planned, "block_acknowledgements": acknowledgements,
            "queue_settings": asdict(self.queue_settings), "transport_settings": asdict(self.transport)})


class ObservingPolicy:
    """Собирает снимок до решения; делегирует поведение и имя исходной политики."""
    def __init__(self, policy, ledger):
        self.policy, self.ledger, self.snapshots = policy, ledger, []

    def __getattr__(self, key):
        return getattr(self.policy, key)

    def on_epoch(self, t, view):
        self.snapshots.append(self.ledger.queue_observation(t))
        return self.policy.on_epoch(t, view)
