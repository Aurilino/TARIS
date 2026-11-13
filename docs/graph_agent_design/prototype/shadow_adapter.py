"""Р6а: только чтение нормализованных наблюдений основной среды."""
from dataclasses import dataclass

from learning import Observation, ObservationStore


@dataclass(frozen=True)
class PublicSlice:
    start_day: float
    end_day: float
    registered: tuple[str, ...]
    down_at_end: tuple[str, ...]


class ChainObservationAdapter:
    """Вход — завершённый час наблюдения, не объект World/Chain/Ledger."""

    def __init__(self, registry):
        self.registry = tuple(registry)
        self.names = {spec.name for spec in self.registry}
        self.store = ObservationStore()
        self.hours = 0

    def ingest(self, payload: PublicSlice):
        start, end = round(payload.start_day * 24), round(payload.end_day * 24)
        if (abs(payload.start_day * 24 - start) > 1e-8
                or abs(payload.end_day * 24 - end) > 1e-8
                or start != self.hours or end != start + 1):
            raise ValueError("Адаптер принимает последовательные завершённые часовые интервалы с нуля")
        registered, down = set(payload.registered), set(payload.down_at_end)
        if not down <= registered or not registered <= self.names:
            raise ValueError("Несогласованный публичный срез")
        values = tuple((spec.name, spec.name in down if spec.name in registered else None)
                       for spec in self.registry)
        context = "day" if 8 <= start % 24 < 20 else "night"
        self.store.add(Observation(start, end, "availability", context, values))
        self.hours = end


def public_slice(chain, node_names, public_join_days, start_hour):
    """Пограничный код читает только Chain.observe и известное время регистрации."""
    a, b = start_hour / 24, (start_hour + 1) / 24
    observed = chain.observe(a, b)
    registered = tuple(name for node, name in node_names.items() if public_join_days[node] <= b)
    down_nodes = set(observed["outages_ongoing"]["node"].tolist())
    down = tuple(name for node, name in node_names.items() if node in down_nodes and name in registered)
    return PublicSlice(a, b, registered, down)
