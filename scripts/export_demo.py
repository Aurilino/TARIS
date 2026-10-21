"""Экспорт небольшого мира для интерактивного демонстратора среды (шаг 3, пункт 3.8, уровень 1+).
Естественные события (простои, шоки, микропрерывания, уход) берутся из Python-мира без изменений;
страница добавляет к ним только ручные вмешательства пользователя."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain

cfg = scenario("S2", seed=3, n_initial_nodes=22, n_dc=8, n_providers=3, n_regions=2,
               node_join_rate=0.02, churn_extra_new=1 / 400, horizon_days=365)
w = World(cfg); ch = Chain(w); nd = w.nodes
td = ch.true_down
HW = {0: "бытовой HDD", 1: "серверный HDD", 2: "SSD"}
nodes = []
for i in nd.index:
    iv = td[td.node == i].sort_values("start")
    nodes.append({
        "id": int(i), "dc": int(nd.dc[i]), "prov": int(nd.provider[i]), "reg": int(nd.region[i]), "owner": int(nd.owner[i]),
        "hw": HW[int(nd.hw_true[i])], "join": round(float(nd.join_t[i]), 4),
        "leave": None if not np.isfinite(nd.leave_t[i]) else round(float(nd.leave_t[i]), 4),
        "lam": round(float(nd.lam_avg[i]) * 365, 2), "phi": round(w.true_phi(int(i)), 4),
        "iv": [[round(float(a), 5), round(float(b), 5), s] for a, b, s in zip(iv.start, iv.end, iv.source)],
    })
shocks = [{"ctype": r.ctype, "cluster": int(r.cluster), "start": round(float(r.start), 4), "end": round(float(r.end), 4),
           "destroyed": bool(r.destroyed), "n": int(r.n_nodes)} for r in w.shocks.itertuples()]
data = {"cfg": {"d": cfg.d, "T_rep": cfg.T_rep, "horizon": cfg.horizon_days, "a": cfg.heterogeneity_a,
                "seed": cfg.seed, "hash": cfg.hash()}, "nodes": nodes, "shocks": shocks}
out = sys.argv[1] if len(sys.argv) > 1 else "demo_world.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
print("узлов", len(nodes), "интервалов", sum(len(n["iv"]) for n in nodes), "шоков", len(shocks),
      "ушло", sum(n["leave"] is not None for n in nodes), "размер", os.path.getsize(out) // 1024, "КБ")
print("ДЦ:", nd.groupby("dc").size().to_dict())
print(w.shocks[["ctype", "cluster", "start", "end", "destroyed", "n_nodes"]].round(2).to_string(index=False))
