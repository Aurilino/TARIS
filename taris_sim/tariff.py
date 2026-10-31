"""Тариф узла по оценкам агента (пункт 2.7, подшаг 5.2)."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np


@dataclass(frozen=True)
class PriceBasis:
    """Цены для чистой премии на единицу хранимого объёма.

    g — стоимость восстановления единицы объёма при полной гибели;
    u — стоимость недоступности единицы объёма за сутки.
    Значения задаются вызывающей стороной: симулятор не подменяет их c_tr или W.
    """

    g: float
    u: float

    def __post_init__(self):
        if not isfinite(self.g) or self.g < 0:
            raise ValueError("g должен быть конечной неотрицательной ценой")
        if not isfinite(self.u) or self.u < 0:
            raise ValueError("u должен быть конечной неотрицательной ценой за сутки")


@dataclass(frozen=True)
class TariffSettings:
    """Явные денежные параметры тарифного размещения, без значений по умолчанию."""

    basis: PriceBasis
    theta: float
    eta: float
    zeta: float
    reward0: float
    psi: float

    def __post_init__(self):
        vals = np.asarray([self.theta, self.eta, self.zeta, self.reward0, self.psi], dtype=float)
        if not np.isfinite(vals).all() or np.any(vals < 0):
            raise ValueError("коэффициенты и цены тарифа должны быть конечными и неотрицательными")


def own_pure_premium(model, nodes, t: float, basis: PriceBasis) -> dict[str, np.ndarray]:
    """Чистая премия собственного риска узлов на единицу объёма за сутки.

    Для каждого узла возвращает тариф и разложение на полную гибель и частичный
    простой. Сетевые шоки здесь намеренно не включены: их тарифная тяжесть требует
    отдельной оценки на уровне кластеров.
    """
    nodes = np.asarray(nodes, dtype=int)
    lam = model.lam(nodes, t)
    phi = model.phi(nodes, t)
    m = model.partial_excess(nodes)
    full_loss = lam * phi * basis.g
    partial_downtime = lam * (1.0 - phi) * basis.u * m
    return {
        "claim_rate": lam,
        "full_loss_share": phi,
        "partial_excess_days": m,
        "full_loss_cost": full_loss,
        "partial_downtime_cost": partial_downtime,
        "pure_premium": full_loss + partial_downtime,
    }


def cluster_pure_premium(model, nodes, basis: PriceBasis) -> dict[str, np.ndarray]:
    """Чистая премия за наблюдаемые общие шоки, суммированная по доменам узла.

    Кластерные частоты, доли полных потерь и тяжесть частичного простоя берутся из
    `RiskModel.fit`. Для каждого типа домена возвращается отдельный вектор тарифа.
    """
    nodes = np.asarray(nodes, dtype=int)
    by_type = {}
    for ct in model.shock_types:
        membership = getattr(model.reg, ct)
        out = np.zeros(len(nodes), dtype=float)
        for cl in np.unique(membership[nodes]):
            if cl < 0:
                continue
            key = (ct, int(cl))
            phi = model.shock_phi[key]
            partial = 0.0 if phi >= 1.0 else (1.0 - phi) * basis.u * model.shock_m[key]
            out[membership[nodes] == cl] += model.shock_nu[key] * (phi * basis.g + partial)
        by_type[ct] = out
    by_type["total"] = sum(by_type.values(), np.zeros(len(nodes), dtype=float))
    return by_type


def pure_premium(model, nodes, t: float, basis: PriceBasis) -> dict[str, np.ndarray]:
    """Полная чистая премия: собственный риск плюс наблюдаемые кластерные шоки."""
    own = own_pure_premium(model, nodes, t, basis)
    clusters = cluster_pure_premium(model, nodes, basis)
    return {
        **own,
        "cluster_pure_premium": clusters["total"],
        "cluster_by_type": clusters,
        "total_pure_premium": own["pure_premium"] + clusters["total"],
    }


def cluster_volume_share(volumes, clusterings) -> np.ndarray:
    """Для каждого узла — максимальная доля сетевого объёма в его найденном кластере.

    `clusterings` — одна или несколько меток кластеров; -1 означает одиночный узел.
    Объём должен учитывать размер хранимых фрагментов, а не только их количество.
    """
    volumes = np.asarray(volumes, dtype=float)
    if volumes.ndim != 1 or not np.isfinite(volumes).all() or np.any(volumes < 0):
        raise ValueError("volumes должны быть конечным неотрицательным вектором")
    total = float(volumes.sum())
    share = np.zeros(len(volumes), dtype=float)
    if total <= 0:
        return share
    if isinstance(clusterings, np.ndarray) and clusterings.ndim == 1:
        clusterings = (clusterings,)
    for labels in clusterings:
        labels = np.asarray(labels, dtype=int)
        if labels.shape != volumes.shape:
            raise ValueError("метки кластеров должны совпадать по размеру с volumes")
        for cl in np.unique(labels[labels >= 0]):
            member = labels == cl
            share[member] = np.maximum(share[member], volumes[member].sum() / total)
    return share


def gross_premium(pure, concentration, suspicion, *, theta: float, eta: float, zeta: float) -> dict[str, np.ndarray]:
    """Брутто-тариф по формуле 2.7 с разложением надбавок.

    Величины концентрации и подозрительности передаются из текущего снимка графа,
    а коэффициенты надбавок задаются явно, без встроенных значений по умолчанию.
    """
    coeff = np.asarray([theta, eta, zeta], dtype=float)
    if not np.isfinite(coeff).all() or np.any(coeff < 0):
        raise ValueError("theta, eta и zeta должны быть конечными неотрицательными коэффициентами")
    pure, concentration, suspicion = np.broadcast_arrays(
        np.asarray(pure, dtype=float), np.asarray(concentration, dtype=float), np.asarray(suspicion, dtype=float)
    )
    if not (np.isfinite(pure).all() and np.isfinite(concentration).all() and np.isfinite(suspicion).all()):
        raise ValueError("тарифные входы должны быть конечными")
    if np.any(pure < 0) or np.any((concentration < 0) | (concentration > 1)) or np.any((suspicion < 0) | (suspicion > 1)):
        raise ValueError("pure должен быть >= 0, а концентрация и подозрительность — в [0, 1]")
    risk_loading = theta * pure
    concentration_loading = eta * concentration * pure
    suspicion_loading = zeta * suspicion * pure
    return {
        "pure_premium": pure,
        "risk_loading": risk_loading,
        "concentration_loading": concentration_loading,
        "suspicion_loading": suspicion_loading,
        "gross_premium": pure + risk_loading + concentration_loading + suspicion_loading,
    }


def node_financial_terms(gross, volume, trust, basis: PriceBasis, *, reward0: float, psi: float) -> dict[str, np.ndarray]:
    """Вознаграждение и залог узла по тарифу и объёму хранимых данных.

    Новые данные допускаются только при брутто-тарифе ниже базовой цены хранения.
    Вознаграждение может быть отрицательным для уже размещённых данных: это показывает
    экономическую невыгодность, а не автоматически запускает выселение узла.
    """
    if not np.isfinite(reward0) or reward0 < 0:
        raise ValueError("reward0 должен быть конечной неотрицательной ценой")
    if not np.isfinite(psi) or psi < 0:
        raise ValueError("psi должен быть конечным неотрицательным коэффициентом")
    gross, volume, trust = np.broadcast_arrays(
        np.asarray(gross, dtype=float), np.asarray(volume, dtype=float), np.asarray(trust, dtype=float)
    )
    if not (np.isfinite(gross).all() and np.isfinite(volume).all() and np.isfinite(trust).all()):
        raise ValueError("gross, volume и trust должны быть конечными")
    if np.any(gross < 0) or np.any(volume < 0) or np.any((trust < 0) | (trust > 1)):
        raise ValueError("gross и volume должны быть >= 0, trust должен быть в [0, 1]")
    return {
        "reward_per_volume": reward0 - gross,
        "new_data_eligible": gross < reward0,
        "collateral": volume * basis.g * (1.0 + psi * (1.0 - trust)),
    }
