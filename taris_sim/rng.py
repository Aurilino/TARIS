"""Независимые потоки случайных чисел.

Каждый поток задаётся зерном прогона и набором меток (тип потока, узел, эпоха...).
Поэтому случайность мира не зависит от решений политики: общие случайные числа.
"""
import numpy as np

# Номера потоков мира
NODES, OPERATORS, OUTAGE, DURATION, CHURN, SHOCK, SILENT, MICRO, STORM, PING, AUDIT, NOISE, DATA, CRISIS, DEVICES = range(1, 16)
ADVERSARY = 16


def stream(seed: int, *labels: int) -> np.random.Generator:
    """Генератор для потока с данными метками; один и тот же набор меток всегда даёт одну и ту же последовательность."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), *[int(x) for x in labels]]))
