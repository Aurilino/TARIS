"""Данные и кодирование (шаг 3, пункт 3.2): поток блоков и ёмкость узлов.

Размещение фрагментов выбирает политика; здесь — только то, что от политики не зависит.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, YEAR
from . import rng as R


class DataStream:
    def __init__(self, cfg: Config, n_nodes: int):
        self.cfg = cfg
        c = cfg
        rows = []
        oid = 0
        for t in range(c.horizon_days):
            b = int(round(c.blocks_per_epoch * (1.0 + c.block_growth_per_year) ** (t / YEAR)))
            for _ in range(b):
                rows.append((oid, t, c.object_size))
                oid += 1
        self.objects = pd.DataFrame(rows, columns=["obj", "created_epoch", "size"]).set_index("obj")
        g = R.stream(c.seed, R.DATA, 0)
        s2 = c.capacity_sd ** 2
        self.capacity = np.maximum(10, np.round(c.capacity_mean_fragments * np.exp(g.normal(-s2 / 2, c.capacity_sd, n_nodes)))).astype(int)

    def new_objects(self, epoch: int) -> np.ndarray:
        """Номера объектов, поступивших в эпоху epoch."""
        return self.objects.index[self.objects["created_epoch"] == epoch].to_numpy()

    def fragment_size(self, obj: int) -> float:
        return float(self.objects.at[obj, "size"]) / self.cfg.k
