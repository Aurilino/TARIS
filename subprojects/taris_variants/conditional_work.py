"""I4.7a: условный сценарий передачи, не прогноз обязательств."""
from math import isfinite


def conditional_work(snapshot, mode="trust_ack", transfer_price=.1):
    if snapshot["schema"] != "taris-queue-observation/2" or mode not in ("trust_ack", "restart_partial"):
        raise ValueError("Неизвестный контракт или сценарий")
    t, k, stripes = snapshot["time"], snapshot["k"], snapshot["transport_settings"]["stripes"]
    rate = snapshot["queue_settings"]["rate_size_per_hour"]
    fraction = snapshot["queue_settings"]["proof_fraction"]
    if (type(k) is not int or k < 1 or type(stripes) is not int or stripes < 1
            or any(not isfinite(x) or x < 0 for x in (t, transfer_price))
            or not isfinite(rate) or rate <= 0 or not isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("Некорректные единицы или цена")
    sizes = {r["obj"]: r["size"] for r in snapshot["object_sizes"]}
    if len(sizes) != len(snapshot["object_sizes"]) or any(not isfinite(v) or v <= 0 for v in sizes.values()):
        raise ValueError("Некорректные размеры объектов")
    progress = {}
    for ack in sorted(snapshot["block_acknowledgements"], key=lambda r: r["time"]):
        if not isfinite(ack["time"]) or not 0 <= ack["time"] < t:
            raise ValueError("ACK не принадлежит прошлому среза")
        if ack["phase"] == "write_block" and ack["ok"]:
            stripe = ack["stripe"]
            if type(stripe) is not int or not 0 <= stripe < stripes:
                raise ValueError("Некорректный номер блока")
            progress[ack["job"], ack["idx"], ack["node"]] = stripe+1
    memory = {}
    for row in snapshot["local_memory"]:
        if (row["kind"] not in ("full", "decoded_block") or not isfinite(row["size"])
                or row["size"] <= 0 or not 0 <= row["confirmed_at"] < t):
            raise ValueError("Некорректное подтверждение памяти")
        memory[row["job"]] = memory.get(row["job"], 0.) + row["size"]
    rows, unknown = [], []
    seen = set()
    for job in snapshot["jobs"]:
        if job["id"] in seen:
            raise ValueError("Повтор задания")
        seen.add(job["id"])
        if job["kind"] == "proof":
            unknown.append(job["id"])
            continue
        if job["kind"] != "repair" or len(job["indices"]) != len(job["nodes"]):
            raise ValueError("Некорректная заявка")
        size = sizes[job["obj"]]
        held = memory.get(job["id"], 0.)
        if held > size+1e-10:
            raise ValueError("Память превышает размер объекта")
        read = max(0., size-held)
        write, targets = 0., []
        if len(set(job["indices"])) != len(job["indices"]):
            raise ValueError("Повтор номера цели")
        for idx, node in zip(job["indices"], job["nodes"]):
            confirmed = progress.get((job["id"], idx, node), 0)
            credited = confirmed if mode == "trust_ack" or confirmed == stripes else 0
            volume = size/k*(1-credited/stripes)
            write += volume
            targets.append({"idx": idx, "node": node, "last_ack_blocks": confirmed,
                            "credited_blocks": credited, "write_volume": volume})
        proof = size*fraction
        rows.append({"job": job["id"], "obj": job["obj"], "read_volume": read,
                     "write_volume": write, "closing_proof_volume": proof,
                     "volume": read+write+proof, "targets": targets})
    total = sum(r["volume"] for r in rows)
    return {"schema": "taris-conditional-work/1", "time": t, "mode": mode, "jobs": rows,
        "known_repair_volume": total, "conditional_repair_transfer_cost": total*transfer_price,
        "uninterrupted_transfer_hours": total/rate, "unknown_proof_jobs": unknown,
        "whole_queue_volume": None if unknown else total,
        "expected_future_cost": None, "expected_loss_cost": None, "completion_time": None,
        "full_actuarial_reserve": None}
