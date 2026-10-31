"""Целочисленный расчёт лёгкой части тарифа (пункт 2.8)."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from math import isfinite

SCALE = 10**9


def encode(value: float | int | str) -> int:
    """Квантовать внешнее число в десятичную фиксированную точку, округляя половины вверх."""
    try:
        x = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("значение должно быть конечным числом") from exc
    if not x.is_finite():
        raise ValueError("значение должно быть конечным числом")
    return int((x * SCALE).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def decode(value: int) -> float:
    return int(value) / SCALE


def _round_div(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError("знаменатель должен быть положительным")
    if num < 0:
        return -_round_div(-num, den)
    return (num + den // 2) // den


def gross_premium_q(pure: int, concentration: int, suspicion: int,
                    theta: int, eta: int, zeta: int) -> int:
    """Брутто-тариф в масштабе SCALE; все входы уже закодированы в том же масштабе."""
    values = (pure, concentration, suspicion, theta, eta, zeta)
    if any(not isinstance(x, int) for x in values):
        raise TypeError("фиксированные входы должны быть целыми числами")
    if pure < 0 or not 0 <= concentration <= SCALE or not 0 <= suspicion <= SCALE:
        raise ValueError("pure должен быть >= 0, concentration и suspicion — в [0, 1]")
    if theta < 0 or eta < 0 or zeta < 0:
        raise ValueError("коэффициенты должны быть неотрицательными")
    # Сложить множитель в масштабе SCALE^2 и округлить один раз после умножения на pure.
    factor_q2 = SCALE * SCALE + theta * SCALE + eta * concentration + zeta * suspicion
    return _round_div(pure * factor_q2, SCALE * SCALE)


def financial_terms_q(gross: int, volume: int, trust: int, g: int,
                      reward0: int, psi: int) -> tuple[int, bool, int]:
    """Вознаграждение на объём, допуск новых данных и залог в фиксированной точке."""
    if any(not isinstance(x, int) for x in (gross, volume, trust, g, reward0, psi)):
        raise TypeError("фиксированные входы должны быть целыми числами")
    if gross < 0 or volume < 0 or g < 0 or reward0 < 0 or psi < 0 or not 0 <= trust <= SCALE:
        raise ValueError("некорректные финансовые входы")
    reward = reward0 - gross
    eligible = gross < reward0
    trust_loading = _round_div(psi * (SCALE - trust), SCALE)
    collateral = _round_div(volume * g * (SCALE + trust_loading), SCALE * SCALE)
    return reward, eligible, collateral
