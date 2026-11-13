"""I4.4a: отмена наблюдаемо ненужных целей без изменения правил буфера."""
import numpy as np
from .resumable import ResumableLedger, ResumableResult


class CancellationResult(ResumableResult):
    def metrics(self):
        return {"schema": "taris-cancellation-metrics/1", "policy": self.policy_name,
                **self._L.queue_report(self._L.executed_until)}


class CancellationLedger(ResumableLedger):
    def _reset_state(self):
        super()._reset_state()
        self.obsolete_cancellations = []

    def _next(self, job, t):
        obj = job["obj"]
        if obj not in self.buffers and self.obj_lost_t[obj] == np.inf:
            cursor = job.get("cursor", 0)
            kept_idx, kept_nodes = job["indices"][:cursor], job["nodes"][:cursor]
            for idx, node in zip(job["indices"][cursor:], job["nodes"][cursor:]):
                if idx in self.missing[obj]:
                    kept_idx.append(idx)
                    kept_nodes.append(node)
                    continue
                self.obsolete_cancellations.append({"job": job["id"], "time": float(t), "obj": obj,
                    "idx": idx, "node": node, "reason": "not_in_observed_missing",
                    "observed_missing": sorted(map(int, self.missing[obj])),
                    "read_progress": job.get("read_progress", 0)})
                self.used[node] -= 1
                self.reserved[node] -= 1
                self.pending[obj].discard(idx)
                self.operation_outcomes.append((float(t), obj, idx, node, "cancelled_obsolete"))
            job["indices"], job["nodes"] = kept_idx, kept_nodes
            if cursor == len(kept_idx):
                self._finish(job, t, "obsolete_before_read")
                return
        super()._next(job, t)

    def run(self, policy, epochs=None, progress=False):
        result = super().run(policy, epochs, progress)
        return CancellationResult(self, policy, result.daily)

    def queue_report(self, end):
        report = super().queue_report(end)
        report["obsolete_targets_cancelled"] = len(self.obsolete_cancellations)
        report["obsolete_jobs_cancelled"] = sum(j["status"] == "obsolete_before_read" for j in self.jobs)
        return report
