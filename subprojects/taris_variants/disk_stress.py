"""I5: отдельный адаптер разрушения дисков с явным уведомлением."""
from dataclasses import dataclass
import heapq
from math import isfinite
import numpy as np
from .observable_memory import ObservableMemoryLedger


@dataclass(frozen=True)
class DiskWave:
    time: float
    nodes: tuple


def public_wave_targets(nodes):
    initial = nodes[(nodes.join_t <= 0) & (nodes.dc >= 0)]
    counts = initial.groupby("dc").size().to_dict()
    domains = sorted(counts, key=lambda dc: (-counts[dc], dc))[:3]
    if len(domains) != 3:
        raise ValueError("Нужны три непустых исходных ДЦ")
    groups = [tuple(map(int, initial.index[initial.dc == dc])) for dc in domains]
    return {"domains": list(map(int, domains)), "groups": groups}


class DiskStressLedger(ObservableMemoryLedger):
    def __init__(self, world, chain, data, queue_settings=None, transport_settings=None, waves=()):
        self.stress_waves = tuple(waves)
        super().__init__(world, chain, data, queue_settings, transport_settings)
        seen = set()
        if len({wave.time for wave in self.stress_waves}) != len(self.stress_waves):
            raise ValueError("Одновременные цели должны входить в одну волну")
        for wave in self.stress_waves:
            if not isfinite(wave.time) or not 0 <= wave.time < self.T or not wave.nodes:
                raise ValueError("Некорректное время/пустая волна")
            for node in wave.nodes:
                if type(node) is not int or not 0 <= node < self.N or (wave.time, node) in seen:
                    raise ValueError("Некорректная/повторная цель")
                seen.add((wave.time, node))

    def _reset_state(self):
        super()._reset_state()
        self.disk_notifications = []
        self.stress_processing = False

    def _world_events(self):
        super()._world_events()
        for wave in sorted(self.stress_waves, key=lambda r: r.time):
            for node in sorted(wave.nodes):
                self._push(float(wave.time), "destroy_node", node)
            # Сначала вся волна уничтожена; только затем уведомления этой волны.
            for node in sorted(wave.nodes):
                self._push(float(wave.time), "stress_disk_notice", node)

    def queue_observation(self, t):
        if self.stress_processing:
            raise ValueError("Снимок недоступен в ходе обработки волны")
        # Само расписание и будущие уведомления не входят в контракт.
        return super().queue_observation(t)

    def _process(self, t_end):
        if not self.stress_waves:
            return super()._process(t_end)
        if not isfinite(t_end) or t_end < self.observation_clock:
            raise ValueError("Некорректная граница исполнения")
        self.stress_processing = True
        self.observation_ready = False
        while self.heap and self.heap[0][0] < t_end:
            event = heapq.heappop(self.heap)
            t, _, kind, node = event
            if kind == "stress_disk_notice":
                affected = sorted(self.physical_nodes[node])
                self.handling_world_event = True
                for f in affected:
                    self._declare(f, t, "сбой диска (подтверждён)")
                self.handling_world_event = False
                self.disk_notifications.append({"time": float(t), "node": int(node),
                    "kind": "confirmed_disk_failure", "declared_copies": list(map(int, affected))})
                self._pump(t)
                continue
            remaining, self.heap = self.heap, [event]
            super()._process(np.nextafter(float(t), np.inf))
            generated, self.heap = self.heap, remaining
            for item in generated:
                heapq.heappush(self.heap, item)
        self.observation_clock = float(t_end)
        self.observation_ready = True
        self.stress_processing = False
