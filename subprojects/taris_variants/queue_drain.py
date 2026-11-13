"""I6.1: узкий платный fallback для освобождения служебного места."""
from .queue_control import QueueControlLedger, QueueControlResult
from .shared_channel import SharedChannelLedger


class QueueDrainResult(QueueControlResult):
    def metrics(self):
        return {"schema": "taris-queue-drain-metrics/1", "policy": self.policy_name,
                **self._L.queue_report(self._L.executed_until)}


class QueueDrainLedger(QueueControlLedger):
    def __init__(self, *args, drain_irrelevant_proof=False, **kwargs):
        if type(drain_irrelevant_proof) is not bool:
            raise ValueError("Флаг fallback должен быть bool")
        self.drain_irrelevant_proof = drain_irrelevant_proof
        super().__init__(*args, **kwargs)

    def _pump(self, t):
        if (not self.drain_irrelevant_proof or self.active_job is not None or self.parked_job is None
                or self.wake_ready or self.proof_quota_used
                or self.parked_intervals[-1].get("pause_type") != "read_wait"
                or self.jobs[self.parked_job]["obj"] in self.buffers
                or t >= self.parked_intervals[-1]["wake"]
                or self._outstanding() != self.queue_settings.max_jobs):
            return super()._pump(t)
        obj, missing, choices = self._candidates(self.parked_job)
        if not choices or any(choice["relevant_indices"] for choice in choices):
            return super()._pump(t)
        selected = choices[0]["job"]
        self.proof_selection_log.append({"time": float(t), "owner": self.parked_job, "obj": obj,
            "observed_missing": missing, "candidates": choices, "selected": selected,
            "selection_rule": "pressure_fallback", "outstanding": self._outstanding()})
        # SharedChannel сам выбирает старейший proof и расходует квоту этой паузы.
        return SharedChannelLedger._pump(self, t)

    def queue_report(self, end):
        result = super().queue_report(end)
        result["pressure_fallback_starts"] = sum(r.get("selection_rule") == "pressure_fallback"
                                                for r in self.proof_selection_log)
        result["proof_opportunities_without_selection"] = result["proof_opportunities_without_relevant"]
        result["proof_opportunities_without_relevant"] += result["pressure_fallback_starts"]
        return result

    def run(self, policy, epochs=None, progress=False):
        result = super().run(policy, epochs, progress)
        return QueueDrainResult(self, policy, result.daily)
