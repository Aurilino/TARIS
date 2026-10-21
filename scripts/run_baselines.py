"""Первый сквозной прогон в сценарии S2 (согласованный мир). Запуск по одной политике:
python3 scripts/run_baselines.py runs uniform15 | uniform13 | oracle13
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import UniformPolicy, OraclePolicy

out, which = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
cfg = scenario("S2")
w = World(cfg); ch = Chain(w); ds = DataStream(cfg, len(w.nodes)); L = Ledger(w, ch, ds)
pol = {"uniform15": lambda: UniformPolicy(15), "uniform13": lambda: UniformPolicy(13), "oracle13": lambda: OraclePolicy(w, 13)}[which]()
t = time.time()
res = L.run(pol)
un = res.unavailability()
m = res.metrics(); m["Недоступность, доля времени"] = float(un.unavail_share.mean())
m["Время прогона, с"] = round(time.time() - t, 1)
sample = res.fragments[res.fragments.obj.isin(range(0, len(ds.objects), 97))]
with open(f"{out}/{which}.pkl", "wb") as fh:
    pickle.dump({"metrics": m, "losses": res.losses, "repairs": res.repairs, "daily": res.daily,
                 "objects": res.objects[["created_epoch", "n", "lost_t"]], "fragments_sample": sample,
                 "unavail": un, "shocks": w.shocks, "disclosures": res.disclosures}, fh)
print(pd.Series(m).to_string())
