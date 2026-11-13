"""I4.5b: условная цена уже использованных ресурсов, не актуарный прогноз."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import isfinite


def clipped_days(start, stop, end):
    start = float(start)
    if not isfinite(start) or start < 0 or not isfinite(end) or end <= 0:
        raise ValueError("Некорректный горизонт или начало интервала")
    if stop is not None:
        stop = float(stop)
        if not isfinite(stop) or stop < start:
            raise ValueError("Некорректный конец интервала")
    else:
        stop = end
    return max(0., min(stop, end)-start)


def resource_usage(record, end, k, object_size):
    if type(k) is not int or k < 1 or not isfinite(object_size) or object_size <= 0:
        raise ValueError("Некорректные размер объекта или код")
    copies = sum(clipped_days(f["start"], f["physical_end"], end) for f in record["physical_fragments"])
    buffers = record["buffers"]
    decoded = record["decoded_prefixes"]
    receiver = record["receiver_prefixes"]
    size_days = {}
    central_changes = defaultdict(float)
    for rows, key in ((buffers, "full_buffer_size_days"), (decoded, "decoded_prefix_size_days"),
                       (receiver, "receiver_prefix_size_days")):
        total = 0.
        for r in rows:
            size = float(r["size"])
            if not isfinite(size) or size < 0:
                raise ValueError("Некорректный объём блока")
            dt = clipped_days(r["start"], r["end"], end)
            total += size*dt
            if rows is not receiver and dt > 0:
                central_changes[r["start"]] += size
                central_changes[min(end, r["end"] if r["end"] is not None else end)] -= size
        size_days[key] = total
    # В данном протоколе полный буфер заменяет декодированный префикс этого объекта.
    for prefix in decoded:
        for buffer in buffers:
            if prefix["obj"] != buffer["obj"]:
                continue
            a = max(prefix["start"], buffer["start"])
            b = min(end, prefix["end"] if prefix["end"] is not None else end,
                    buffer["end"] if buffer["end"] is not None else end)
            if a < b:
                raise ValueError("Двойной учёт декодированного префикса и полного буфера")
    used = peak = 0.
    for t in sorted(central_changes):
        used += central_changes[t]
        peak = max(peak, used)
    cap = record.get("queue_settings", {}).get("buffer_capacity_size")
    if cap is not None and peak > cap+1e-8:
        raise ValueError("Центральная память превышает заявленную ёмкость")
    volumes, scheduled_unserved = Counter(), 0.
    previous_end = 0.
    for segment in record["segments"]:
        a, b, volume = float(segment["start"]), float(segment["end"]), float(segment["volume"])
        if b <= a or not isfinite(volume) or volume < 0 or a < previous_end-1e-10:
            raise ValueError("Некорректная или перекрывающаяся фаза канала")
        previous_end = b
        duration = clipped_days(a, b, end)
        served = volume*duration/(b-a)
        volumes[segment["phase"]] += served
        scheduled_unserved += volume-served
    return {"physical_copy_days": copies, "physical_size_days": copies*object_size/k,
        **size_days, "central_size_days": size_days["full_buffer_size_days"]+size_days["decoded_prefix_size_days"],
        "served_volume_by_phase": dict(volumes), "served_volume": float(sum(volumes.values())),
        "scheduled_unserved_volume": scheduled_unserved, "peak_central_size": peak}


@dataclass(frozen=True)
class ResourcePrices:
    fragment_day: float = 1/365
    transfer_unit: float = .1
    central_ratio: float = 1.
    receiver_ratio: float = 1.

    def validate(self):
        if any(not isfinite(x) or x < 0 for x in (self.fragment_day, self.transfer_unit, self.central_ratio, self.receiver_ratio)):
            raise ValueError("Цена должна быть конечной и неотрицательной")


def observed_charges(usage, prices, k, object_size):
    prices.validate()
    if type(k) is not int or k < 1 or not isfinite(object_size) or object_size <= 0:
        raise ValueError("Некорректные единицы кодирования")
    fields = ("physical_copy_days", "full_buffer_size_days", "decoded_prefix_size_days",
              "receiver_prefix_size_days", "served_volume")
    if any(not isfinite(usage[key]) or usage[key] < 0 for key in fields):
        raise ValueError("Некорректный расход ресурсов")
    size_day_price = prices.fragment_day*k/object_size
    disk = prices.fragment_day*usage["physical_copy_days"]
    full = size_day_price*prices.central_ratio*usage["full_buffer_size_days"]
    prefix = size_day_price*prices.central_ratio*usage["decoded_prefix_size_days"]
    receiver = size_day_price*prices.receiver_ratio*usage["receiver_prefix_size_days"]
    transfer = prices.transfer_unit*usage["served_volume"]
    return {"disk": disk, "full_buffer": full, "decoded_prefix": prefix, "receiver_prefix": receiver,
            "channel": transfer, "observed_resource_cost": disk+full+prefix+receiver+transfer,
            "size_day_price": size_day_price, "expected_loss_cost": None, "full_actuarial_reserve": None}
