"""Р4: автономное событийное исполнение восстановления с общими ресурсами."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction


def q(value):
    return Fraction(str(value))


@dataclass
class StorageNode:
    name: str
    site: str
    capacity: int = 10
    bandwidth: Fraction = Fraction(1)
    up: bool = True


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    k: int
    target_n: int


@dataclass(frozen=True)
class NetworkSnapshot:
    time: Fraction
    objects: tuple[ObjectSpec, ...]
    up: tuple[str, ...]
    catalog: tuple[tuple[str, int, str], ...]
    free_slots: tuple[tuple[str, int], ...]
    pending: tuple[tuple[str, int, str], ...]
    buffered: tuple[str, ...]
    sites: tuple[tuple[str, str], ...] = ()
    alerts: tuple[tuple[Fraction, str, Fraction], ...] = ()


@dataclass
class RepairJob:
    id: int
    obj: str
    index: int
    target: str
    sources: tuple[tuple[int, str], ...]
    created: Fraction
    phase: str
    remaining: Fraction = Fraction(1)
    status: str = "queued"
    finished: Fraction | None = None
    reason: str = ""


class RecoveryEngine:
    """Истина среды закрыта от политики; политика получает только снимок сети."""

    def __init__(self, nodes, objects, *, site_bandwidth=None, parallel_limit=2,
                 overload_flow_limit=None, overload_grace=1, overload_cooldown=2):
        nodes, objects = tuple(nodes), tuple(objects)
        self.nodes = {node.name: node for node in nodes}
        self.objects = {obj.name: obj for obj in objects}
        if (len(self.nodes) != len(nodes) or len(self.objects) != len(objects)
                or not self.nodes or not self.objects or type(parallel_limit) is not int or parallel_limit < 1
                or any(type(n.capacity) is not int or n.capacity < 0 or q(n.bandwidth) <= 0 for n in self.nodes.values())
                or any(type(o.k) is not int or type(o.target_n) is not int or not 1 <= o.k <= o.target_n
                       for o in self.objects.values())):
            raise ValueError("Некорректная конфигурация среды")
        for node in self.nodes.values():
            node.bandwidth = q(node.bandwidth)
        self.site_bandwidth = {site: q(rate) for site, rate in (site_bandwidth or {}).items()}
        if any(rate <= 0 for rate in self.site_bandwidth.values()):
            raise ValueError("Скорость площадки должна быть положительной")
        self.parallel_limit = parallel_limit
        if overload_flow_limit is not None and (type(overload_flow_limit) is not int or overload_flow_limit < 1):
            raise ValueError("Порог перегрузки должен быть положительным целым")
        self.overload_flow_limit = overload_flow_limit
        self.overload_grace, self.overload_cooldown = q(overload_grace), q(overload_cooldown)
        if self.overload_grace <= 0 or self.overload_cooldown <= 0:
            raise ValueError("Длительности перегрузки должны быть положительными")
        self.pressure_since = {}
        self.overloaded_until = {}
        self.alerts = []
        self.time = Fraction()
        self.physical = set()
        self.catalog = set()
        self.observed_up = {n.name for n in self.nodes.values() if n.up}
        self.departed = set()
        self.buffers = set()
        self.jobs = []
        self.log = []
        self.lost = {}
        self.traffic = Fraction()
        self.history = []
        self.maximum_reserved_used = Counter()

    def add_chunk(self, obj, index, node):
        if obj not in self.objects or node not in self.nodes or type(index) is not int or index < 0:
            raise ValueError("Некорректный чанк")
        if any(o == obj and n == node for o, _, n in self.catalog) or self.free_slots(node) < 1:
            raise ValueError("Размещение нарушает ёмкость или правило одного чанка объекта на ноду")
        self.physical.add((obj, index, node))
        self.catalog.add((obj, index, node))

    def live_jobs(self):
        return [job for job in self.jobs if job.status not in ("done", "failed")]

    def free_slots(self, node):
        allocated = sum(n == node for _, _, n in self.catalog)
        reserved = sum(job.target == node for job in self.live_jobs())
        return self.nodes[node].capacity - allocated - reserved

    def snapshot(self):
        return NetworkSnapshot(self.time, tuple(self.objects.values()), tuple(sorted(n for n in self.observed_up if self._online(n))),
            tuple(sorted(self.catalog)), tuple((n, self.free_slots(n)) for n in sorted(self.nodes)),
            tuple((j.obj, j.index, j.target) for j in self.live_jobs()), tuple(sorted(self.buffers)),
            tuple(sorted((n.name, n.site) for n in self.nodes.values())),
            tuple(alert for alert in self.alerts if self.time < alert[2]))

    def _online(self, node):
        return self.nodes[node].up and self.time >= self.overloaded_until.get(self.nodes[node].site, Fraction())

    def request(self, obj, index, target, sources):
        if obj not in self.objects or target not in self.nodes or type(index) is not int or index < 0:
            return False
        if (target not in self.observed_up or not self._online(target) or self.free_slots(target) < 1
                or any(o == obj and n == target for o, _, n in self.catalog)
                or any(j.obj == obj and (j.index == index or j.target == target) for j in self.live_jobs())):
            return False
        sources = tuple(sources)
        if obj not in self.buffers:
            if (len(sources) != self.objects[obj].k or len({idx for idx, _ in sources}) != len(sources)
                    or any((obj, idx, node) not in self.catalog or node not in self.observed_up
                           for idx, node in sources)):
                return False
        job = RepairJob(len(self.jobs), obj, index, target, sources, self.time,
                        "write" if obj in self.buffers else "read")
        self.jobs.append(job)
        self.log.append({"time": self.time, "event": "reserved", "job": job.id, "target": target})
        self._check_capacity()
        return True

    def _check_capacity(self):
        for node in self.nodes:
            used = self.nodes[node].capacity - self.free_slots(node)
            self.maximum_reserved_used[node] = max(self.maximum_reserved_used[node], used)
            if self.free_slots(node) < 0:
                raise AssertionError("Переполнение ёмкости")

    def _fail(self, job, reason):
        job.status, job.finished, job.reason = "failed", self.time, reason
        self.log.append({"time": self.time, "event": "failed", "job": job.id, "reason": reason})

    def apply(self, kind, node=None, obj=None, index=None, site=None, duration=0):
        if kind in ("down", "up", "destroy", "depart", "audit", "silent_loss") and node not in self.nodes:
            raise ValueError("Неизвестная нода события")
        if kind == "silent_loss" and (obj not in self.objects or type(index) is not int or index < 0):
            raise ValueError("Некорректная тихая потеря")
        if kind == "warning":
            if site not in {n.site for n in self.nodes.values()} or q(duration) <= 0:
                raise ValueError("Некорректное предупреждение")
            self.alerts.append((self.time, site, self.time + q(duration)))
        if kind in ("down", "destroy", "depart"):
            self.nodes[node].up = False
            self.observed_up.discard(node)  # в Р4 пинг без задержки
        if kind == "up":
            if node in self.departed:
                raise ValueError("Ушедшая навсегда нода не возвращается")
            self.nodes[node].up = True
            self.observed_up.add(node)
        if kind in ("destroy", "depart"):
            self.physical = {chunk for chunk in self.physical if chunk[2] != node}
        if kind == "depart":
            self.departed.add(node)
            self.catalog = {chunk for chunk in self.catalog if chunk[2] != node}
        if kind == "silent_loss":
            self.physical.discard((obj, index, node))
        elif kind == "audit" and self.nodes[node].up:
            self.catalog = {chunk for chunk in self.catalog if chunk[2] != node or chunk in self.physical}
        elif kind == "buffer_loss":
            self.buffers.clear()
            for job in self.live_jobs():
                if job.phase == "write":
                    self._fail(job, "утрачен буфер восстановленного объекта")
        elif kind not in ("down", "up", "destroy", "depart", "audit", "silent_loss", "warning"):
            raise ValueError("Неизвестное событие")
        self.log.append({"time": self.time, "event": kind, "node": node, "obj": obj, "index": index, "site": site})

    def _rates(self):
        for site in list(self.overloaded_until):
            if self.overloaded_until[site] <= self.time:
                del self.overloaded_until[site]
                self.log.append({"time": self.time, "event": "overload_recovered", "site": site})
        ready = []
        for job in self.live_jobs():
            if job.target in self.departed or job.obj in self.lost:
                self._fail(job, "получатель ушёл или объект потерян")
                continue
            if job.phase == "read":
                invalid = [(idx, node) for idx, node in job.sources
                           if node in self.departed or self._online(node) and (job.obj, idx, node) not in self.physical]
                if invalid:
                    # Исполнитель сообщает провал проверки, не раскрывает скрытую причину заранее.
                    for idx, node in invalid:
                        self.catalog.discard((job.obj, idx, node))
                    self._fail(job, "источник не прошёл проверку")
                    continue
                available = all(self._online(node) for _, node in job.sources)
                participants = tuple(node for _, node in job.sources)
            else:
                available = self._online(job.target) and job.obj in self.buffers
                participants = (job.target,)
            job.status = "queued"
            if available:
                ready.append((job, participants))
        ready = ready[:self.parallel_limit]
        nodes = Counter(n for _, participants in ready for n in participants)
        sites = Counter(self.nodes[n].site for _, participants in ready for n in participants)
        if self.overload_flow_limit is not None:
            for site in list(self.pressure_since):
                if sites[site] <= self.overload_flow_limit:
                    del self.pressure_since[site]
            triggered = []
            for site, count in sites.items():
                if count > self.overload_flow_limit:
                    start = self.pressure_since.setdefault(site, self.time)
                    if self.time - start >= self.overload_grace:
                        triggered.append(site)
            if triggered:
                for site in triggered:
                    self.overloaded_until[site] = self.time + self.overload_cooldown
                    self.pressure_since.pop(site, None)
                    self.log.append({"time": self.time, "event": "overload", "site": site,
                                     "flows": sites[site], "until": self.overloaded_until[site]})
                self._assess()
                return self._rates()
        rates = []
        for job, participants in ready:
            limits = [self.nodes[n].bandwidth / nodes[n] for n in participants]
            limits += [self.site_bandwidth[self.nodes[n].site] / sites[self.nodes[n].site]
                       for n in participants if self.nodes[n].site in self.site_bandwidth]
            rate = min(limits)
            job.status = "reading" if job.phase == "read" else "writing"
            rates.append((job, rate, len(participants)))
        return rates

    def _complete(self, job):
        if job.phase == "read":
            self.buffers.add(job.obj)
            job.phase, job.remaining, job.status = "write", Fraction(1), "queued"
            self.log.append({"time": self.time, "event": "read_confirmed", "job": job.id})
        else:
            self.physical.add((job.obj, job.index, job.target))
            self.catalog.add((job.obj, job.index, job.target))
            job.status, job.finished = "done", self.time
            self.log.append({"time": self.time, "event": "write_confirmed", "job": job.id})

    def _assess(self):
        state = {"time": self.time, "objects": {}}
        for obj, spec in self.objects.items():
            durable = {idx for o, idx, _ in self.physical if o == obj}
            readable = {idx for o, idx, n in self.physical if o == obj and self._online(n)}
            if len(durable) >= spec.target_n and not any(j.obj == obj for j in self.live_jobs()):
                self.buffers.discard(obj)
            if len(durable) < spec.k and obj not in self.buffers and obj not in self.lost:
                self.lost[obj] = self.time
            state["objects"][obj] = {"durable_unique": len(durable), "readable_unique": len(readable),
                                     "buffered": obj in self.buffers, "lost": obj in self.lost}
        self.history.append(state)

    def run(self, end, *, events=(), policy=None, decision_interval=1):
        end, interval = q(end), q(decision_interval)
        if end < self.time or interval <= 0:
            raise ValueError("Некорректный горизонт или шаг решений")
        schedule = sorted([(q(t), dict(event)) for t, event in events], key=lambda item: item[0])
        if any(t < self.time or t > end for t, _ in schedule):
            raise ValueError("Событие вне прогона")
        event_index, decision = 0, self.time
        while True:
            while event_index < len(schedule) and schedule[event_index][0] == self.time:
                self.apply(**schedule[event_index][1])
                event_index += 1
            self._assess()
            if self.time == end:
                self._rates()
                self._check_capacity()
                break
            if policy is not None and self.time == decision:
                for request in policy(self.snapshot()):
                    self.request(*request)
                decision += interval
            rates = self._rates()
            next_times = [end]
            if event_index < len(schedule):
                next_times.append(schedule[event_index][0])
            if policy is not None:
                next_times.append(decision)
            next_times += [until for until in self.overloaded_until.values() if until > self.time]
            next_times += [start + self.overload_grace for start in self.pressure_since.values()
                           if start + self.overload_grace > self.time]
            next_times += [self.time + job.remaining / rate for job, rate, _ in rates]
            next_time = min(t for t in next_times if t > self.time)
            delta = next_time - self.time
            for job, rate, copies in rates:
                transferred = rate * delta
                job.remaining -= transferred
                self.traffic += transferred * copies
            self.time = next_time
            for job, _, _ in rates:
                if job.remaining == 0:
                    self._complete(job)
            self._check_capacity()
        return self


class RecoveryPolicy:
    def __init__(self, risk_model=None):
        self.risk_model = risk_model

    def __call__(self, view: NetworkSnapshot):
        free = dict(view.free_slots)
        requests = []
        for spec in sorted(view.objects, key=lambda obj: (sum(o == obj.name and n in view.up
                                                           for o, _, n in view.catalog), obj.name)):
            mapping = {}
            for obj, idx, node in view.catalog:
                if obj == spec.name and node in view.up:
                    mapping.setdefault(idx, node)
            pending = {idx for obj, idx, _ in view.pending if obj == spec.name}
            if len(set(mapping) | pending) >= spec.target_n:
                continue
            if len(mapping) < spec.k and spec.name not in view.buffered:
                continue
            holders = {node for obj, _, node in view.catalog if obj == spec.name}
            pending_holders = {node for obj, _, node in view.pending if obj == spec.name}
            candidates = [node for node in view.up if free[node] > 0 and node not in holders | pending_holders]
            if not candidates:
                continue
            missing_indices = sorted({idx for obj, idx, _ in view.catalog if obj == spec.name}
                                     - set(mapping) - pending)
            idx = missing_indices[0] if missing_indices else max([*mapping, *pending, -1]) + 1
            if self.risk_model is None:
                target = min(candidates)
            else:
                target = min(candidates, key=lambda node: (
                    self.risk_model.loss.own(node) + sum(self.risk_model.loss.pair(node, held)
                                                       for held in set(mapping.values())), node))
            sources = tuple(sorted(mapping.items())[:spec.k])
            requests.append((spec.name, idx, target, sources))
            free[target] -= 1
        return requests
