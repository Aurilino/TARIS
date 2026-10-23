"""Прогон одной политики и журнал для просмотрщика (схема taris-log/1, docs/42_log_schema.md).

python scripts/export_run.py runs main  [политика] [режим мира]   # основной мир, около 200 узлов, сводки по суткам
python scripts/export_run.py runs full  [политика] [режим мира]   # полный мир S6, сводки по 7 суткам
Политика: crisis_oracle13 (по умолчанию) | oracle13 | uniform15 | uniform13.
Файл: <папка>/<мир>_<политика>_<режим>.taris.json.gz
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import UniformPolicy, OraclePolicy, CrisisOraclePolicy
from taris_sim.export import export_run

# основной мир просмотрщика: около 200 узлов в сети одновременно. Вместе с сетью уменьшены частота регистраций
# и поток блоков (ёмкость узлов). Дата-центров не меньше n = 13 (оракул кладёт в дата-центр один фрагмент,
# а второй — только в крайнем случае), провайдеров и регионов — столько, чтобы при лимите r = 3 набиралось 13 узлов.
# Больше дата-центров — не нейтрально: частота бурь и пожаров задана на один дата-центр
MAIN = dict(n_initial_nodes=200, node_join_rate=0.12, blocks_per_epoch=4, n_dc=16, n_providers=6, n_regions=5)
WORLDS = {"main": (MAIN, 1), "full": ({}, 7)}

out, which = sys.argv[1], sys.argv[2]
pol_name = sys.argv[3] if len(sys.argv) > 3 else "crisis_oracle13"
mode = sys.argv[4] if len(sys.argv) > 4 else "matched"
overrides, agg = WORLDS[which]
cfg = scenario("S6", world_mode=mode, **overrides)
t0 = time.time()
w = World(cfg); ch = Chain(w); ds = DataStream(cfg, len(w.nodes)); L = Ledger(w, ch, ds)
pol = {"uniform15": lambda: UniformPolicy(15), "uniform13": lambda: UniformPolicy(13), "oracle13": lambda: OraclePolicy(w, 13),
       "crisis_oracle13": lambda: CrisisOraclePolicy(w, 13)}[pol_name]()
res = L.run(pol)
t1 = time.time()
path = os.path.join(out, f"{which}_{pol_name}_{mode}.taris.json.gz")
log = export_run(w, ch, res, path, scenario="S6", agg_days=agg)
print(f"мир {which}: узлов {len(w.nodes)}, кризисов {len(w.crises)}; прогон {t1 - t0:.0f} с, экспорт {time.time() - t1:.0f} с")
print(f"журнал {path}: {os.path.getsize(path) / 2**20:.2f} МБ")
