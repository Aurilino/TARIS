"""Опциональный анализ защищённого чтения по завершённому журналу.

Не меняет мир/политику, не использует их RNG, не передаёт истину NetworkView.
Сбор шифротекста и раскрытие открытого текста учитываются отдельно.
"""
from __future__ import annotations

import json
import math
from numbers import Integral, Real
from pathlib import Path


SCHEMA = "taris-security-scenario/1"
KINDS = {
    "compromise_node": {"t", "kind", "attacker", "node"},
    "compromise_owner": {"t", "kind", "attacker", "owner"},
    "compromise_access_key": {"t", "kind", "attacker"},
    "compromise_object_key": {"t", "kind", "attacker", "obj"},
    "compromise_manifest_key": {"t", "kind", "attacker", "obj"},
    "client_key_unavailable": {"t", "kind", "obj", "until"},
    "manifest_unavailable": {"t", "kind", "obj", "until"},
    "envelope_unavailable": {"t", "kind", "obj", "until"},
    "deny_node": {"t", "kind", "node", "until"},
    "corrupt_node": {"t", "kind", "node", "until"},
}


class SecurityError(ValueError):
    pass


def _int(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _time(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate(document):
    if not isinstance(document, dict) or set(document) != {"schema", "events"} or document["schema"] != SCHEMA:
        raise SecurityError("Ожидается taris-security-scenario/1 с полями schema и events")
    if not isinstance(document["events"], list) or len(document["events"]) > 10000:
        raise SecurityError("events должен быть списком не более 10000 событий")
    events = []
    for i, source in enumerate(document["events"]):
        if not isinstance(source, dict) or not isinstance(source.get("kind"), str):
            raise SecurityError(f"Событие {i}: неверная структура")
        kind = source["kind"]
        if kind not in KINDS or set(source) != KINDS[kind]:
            raise SecurityError(f"Событие {i}: неизвестный вид или набор полей")
        if not _time(source["t"]):
            raise SecurityError(f"Событие {i}: t задаётся конечным числом суток >= 0")
        for name in ("obj", "node", "owner"):
            if name in source and not _int(source[name]):
                raise SecurityError(f"Событие {i}: {name} должен быть целым >= 0")
        if "attacker" in source and (not isinstance(source["attacker"], str) or
                                     not source["attacker"] or len(source["attacker"]) > 64):
            raise SecurityError(f"Событие {i}: неверный идентификатор коалиции")
        if "until" in source and (not _time(source["until"]) or source["until"] <= source["t"]):
            raise SecurityError(f"Событие {i}: until должен быть больше t")
        events.append(dict(source))
    return sorted(events, key=lambda e: e["t"])


def load(path):
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024:
        raise SecurityError("Файл сценария больше 2 МиБ")
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise SecurityError("Повторяющееся поле JSON")
            result[k] = v
        return result
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                              parse_constant=lambda _: (_ for _ in ()).throw(SecurityError("Неконечное число")))
    except (ValueError, UnicodeError) as error:
        raise SecurityError(str(error)) from error
    validate(document)
    return document


class SecurityReplay:
    """Истинные состояния только для оценивания, не канал признаков политики.

    Один пользователь и неизменяемая криптографическая версия каждого obj.
    Компрометация ключа доступа без ротации распространяется на будущие объекты.
    Зашифрованные манифесты/оболочки публично копируемы с момента публикации.
    """
    def __init__(self, result, document=None):
        self.result = result
        self.events = validate({"schema": SCHEMA, "events": []} if document is None else document)
        self.horizon = float(result.daily["epoch"].max() + 1) if len(result.daily) else 0.
        self.objects = {int(o): float(r.created_epoch) for o, r in result.objects.iterrows()
                        if float(r.created_epoch) < self.horizon}
        self.owner = {int(i): int(v) for i, v in result._L.w.nodes["owner"].items()}
        self.rows = {}
        for r in result.fragments.itertuples(index=False):
            self.rows.setdefault(int(r.obj), []).append(r)
        owners = set(self.owner.values())
        for e in self.events:
            if e["t"] >= self.horizon or e.get("until", 0) > self.horizon:
                raise SecurityError("Событие выходит за фактически выполненный горизонт")
            if "obj" in e and (e["obj"] not in self.objects or e["t"] < self.objects[e["obj"]]):
                raise SecurityError("Объект отсутствует или ещё не создан к событию")
            if "node" in e and e["node"] not in self.owner:
                raise SecurityError("Неизвестный узел")
            if "owner" in e and e["owner"] not in owners:
                raise SecurityError("Неизвестный владелец узла")

    def _query(self, obj, t):
        if not _int(obj) or obj not in self.objects or not _time(t) or not self.objects[obj] <= t < self.horizon:
            raise SecurityError("Запрос должен относиться к созданному объекту внутри выполненного горизонта")

    def initial_n(self, obj):
        created = self.objects[obj]
        indices = {int(r.idx) for r in self.rows.get(obj, ()) if r.t_from == created}
        if indices != set(range(len(indices))):
            raise SecurityError("Начальные индексы фрагментов не образуют непрерывный набор")
        if any(int(r.idx) not in indices for r in self.rows.get(obj, ())):
            raise SecurityError("Новый индекс требует новой версии манифеста; режим ещё не поддержан")
        return len(indices)

    def _blocked(self, kind, obj, t):
        return any(e["kind"] == kind and e["obj"] == obj and e["t"] <= t < e["until"]
                   for e in self.events)

    def node_action(self, kind, node, t):
        return any(e["kind"] == kind and e["node"] == node and e["t"] <= t < e["until"]
                   for e in self.events)

    def live_holders(self, obj, t):
        self._query(obj, t)
        holders = {}
        for r in self.rows.get(obj, ()):
            if r.t_from <= t < min(r.t_dead, r.t_to) and self.result._L.is_up(int(r.node), t):
                holders.setdefault(int(r.idx), []).append(int(r.node))
        return holders

    def good_indices(self, obj, t):
        return {idx for idx, nodes in self.live_holders(obj, t).items()
                if any(not self.node_action("deny_node", node, t) and
                       not self.node_action("corrupt_node", node, t) for node in nodes)}

    def _key_time(self, attacker, obj):
        times = []
        for e in self.events:
            if e.get("attacker") != attacker:
                continue
            if e["kind"] == "compromise_access_key":
                times.append(max(e["t"], self.objects[obj]))
            elif e["kind"] in ("compromise_object_key", "compromise_manifest_key") and e["obj"] == obj:
                times.append(e["t"])
        return min(times, default=math.inf)

    def copies(self, attacker, obj, t):
        """Первый момент копирования каждого индекса. Копии не забываются.

        Компрометация держателя даёт доступ к его целым локальным байтам даже при
        сетевом простое. Интервалы существования берутся из модели Ledger.
        """
        self._query(obj, t)
        controls = {}
        for e in self.events:
            if e.get("attacker") != attacker or e["t"] > t:
                continue
            if e["kind"] == "compromise_node":
                nodes = [e["node"]]
            elif e["kind"] == "compromise_owner":
                nodes = [n for n, owner in self.owner.items() if owner == e["owner"]]
            else:
                continue
            for node in nodes:
                controls[node] = min(controls.get(node, math.inf), e["t"])
        copies = {}
        for r in self.rows.get(obj, ()):
            acquired = max(float(r.t_from), controls.get(int(r.node), math.inf))
            if acquired <= t and acquired < min(r.t_dead, r.t_to):
                idx = int(r.idx)
                copies[idx] = min(copies.get(idx, math.inf), acquired)
        return copies

    def query(self, obj, t):
        self._query(obj, t)
        k = self.result.cfg.k
        holders = self.live_holders(obj, t)
        good = self.good_indices(obj, t)
        reasons = []
        for kind in ("client_key_unavailable", "manifest_unavailable", "envelope_unavailable"):
            if self._blocked(kind, obj, t):
                reasons.append(kind)
        if len(good) < k:
            reasons.append("insufficient_fragments")
        attackers = []
        for name in sorted({e["attacker"] for e in self.events if "attacker" in e}):
            copies = self.copies(name, obj, t)
            threshold = sorted(copies.values())[k-1] if len(copies) >= k else math.inf
            key_time = self._key_time(name, obj)
            reveal = max(threshold, key_time)
            attackers.append({"attacker": name, "copied_indices": sorted(copies),
                              "ciphertext_collected": threshold <= t,
                              "object_key_exposed": key_time <= t,
                              "plaintext_disclosed": reveal <= t,
                              "first_ciphertext_t": threshold if threshold <= t else None,
                              "first_disclosure_t": reveal if reveal <= t else None})
        return {"obj": obj, "t": float(t), "k": k, "initial_n": self.initial_n(obj),
                "live_indices": sorted(holders), "good_indices": sorted(good),
                "client_readable": not reasons, "unavailability_reasons": reasons,
                "attackers": attackers}

    def report(self, queries):
        rows = [self.query(obj, t) for obj, t in queries]
        return {"schema": "taris-security-report/1", "mode": "encrypted-manifest-single-version",
                "scope": "post-run truth analysis; not a policy observation or full adversary module",
                "queries": rows,
                "summary": {"query_count": len(rows),
                            "unreadable_queries": sum(not r["client_readable"] for r in rows),
                            "ciphertext_objects": len({r["obj"] for r in rows if any(a["ciphertext_collected"] for a in r["attackers"])}),
                            "plaintext_disclosed_objects": len({r["obj"] for r in rows if any(a["plaintext_disclosed"] for a in r["attackers"])})}}
