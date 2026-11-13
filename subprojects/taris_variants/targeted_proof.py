"""I4.4c: proof только по наблюдаемой потребности владельца полного буфера."""
from math import isfinite
from .shared_channel import SharedChannelLedger, SharedChannelResult


def relevant_indices(missing, known_holders):
    """Чистое правило: только номера наблюдаемой потребности и прежнего размещения."""
    return sorted(set(missing).intersection(row["idx"] for row in known_holders))


class TargetedProofResult(SharedChannelResult):
    def metrics(self):
        return {"schema": "taris-targeted-proof-metrics/1", "policy": self.policy_name,
                **self._L.queue_report(self._L.executed_until)}


class TargetedProofLedger(SharedChannelLedger):
    def _reset_state(self):
        super()._reset_state()
        self.proof_selection_log = []

    def _candidates(self, owner):
        obj = self.jobs[owner]["obj"]
        missing = sorted(map(int, self.missing[obj]))
        choices = []
        for key in self.waiting:
            job = self.jobs[key]
            if job["kind"] != "proof":
                continue
            node = job["node"]
            known = [{"idx": int(self.f_idx[f]), "declared_at": float(self.f_to[f])}
                     for f in sorted(self.physical_nodes[node])
                     if self.f_obj[f] == obj and isfinite(self.f_to[f])]
            choices.append({"job": key, "node": node, "known_holders": known,
                            "relevant_indices": relevant_indices(missing, known)})
        return obj, missing, choices

    def _pump(self, t):
        if (self.active_job is not None or self.parked_job is None or self.wake_ready
                or t >= self.parked_intervals[-1]["wake"] or self.proof_quota_used):
            return super()._pump(t)
        obj, missing, choices = self._candidates(self.parked_job)
        if not choices:
            return
        selected = next((r["job"] for r in choices if r["relevant_indices"]), None)
        self.proof_selection_log.append({"time": float(t), "owner": self.parked_job, "obj": obj,
            "observed_missing": missing, "candidates": choices, "selected": selected})
        if selected is None:
            return
        # Не выбранные задания сохраняют относительный порядок.
        self.waiting.remove(selected)
        self.waiting.appendleft(selected)
        super()._pump(t)

    def run(self, policy, epochs=None, progress=False):
        result = super().run(policy, epochs, progress)
        return TargetedProofResult(self, policy, result.daily)

    def queue_report(self, end):
        report = super().queue_report(end)
        report["targeted_proof_starts"] = sum(r["selected"] is not None for r in self.proof_selection_log)
        report["proof_opportunities_without_relevant"] = sum(r["selected"] is None for r in self.proof_selection_log)
        return report
