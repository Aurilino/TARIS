"""I4.6b: наблюдаемая собственная память и ответы закрывающих проверок."""
from copy import deepcopy
from .observable_queue import ObservableQueueLedger


class ObservableMemoryLedger(ObservableQueueLedger):
    def _reset_state(self):
        super()._reset_state()
        self.memory_notifications = []

    def _close_buffer(self, obj, t, reason):
        if reason == "buffer_failure":
            # Собственная память контроллера: обнаружение немедленное по модели.
            self.memory_notifications.append({"obj": int(obj), "time": float(t),
                                              "kind": "local_memory_failure"})
        super()._close_buffer(obj, t, reason)

    def queue_observation(self, t):
        snapshot = super().queue_observation(t)
        snapshot["schema"] = "taris-queue-observation/2"
        objects = {r["obj"] for r in snapshot["jobs"] if "obj" in r}
        objects.update(snapshot["full_buffer_objects"])
        snapshot["object_sizes"] = [{"obj": int(obj), "size": float(self.size[obj])}
                                    for obj in sorted(objects)]
        snapshot["k"] = int(self.cfg.k)
        snapshot["local_memory"] = []
        for obj, index in sorted(self.buffers.items()):
            row = self.buffer_intervals[index]
            snapshot["local_memory"].append({"obj": int(obj), "job": row["job"],
                "kind": "full", "size": float(row["size"]), "confirmed_at": row["start"]})
        for row in self.decoded_prefixes:
            if row["end"] is None:
                snapshot["local_memory"].append({key: row[key] for key in
                    ("obj", "job", "stripe", "size")})
                snapshot["local_memory"][-1].update(kind="decoded_block", confirmed_at=row["start"])
        snapshot["closing_acknowledgements"] = [
            {key: row[key] for key in ("job", "time", "ok")} for row in self.close_checks]
        snapshot["memory_notifications"] = deepcopy(self.memory_notifications)
        if any(row["time"] >= t for row in snapshot["closing_acknowledgements"]
               + snapshot["memory_notifications"]):
            raise ValueError("Событие ещё не принадлежит текущему срезу t-")
        return deepcopy(snapshot)
