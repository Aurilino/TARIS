"""Сквозной прогон одной политики. Запуск:
python scripts/run_baselines.py runs uniform15 | uniform13 | oracle13 | crisis_oracle13 [сценарий] [режим мира]
По умолчанию — S2, согласованный мир (файл runs/<политика>.pkl, как раньше);
для других сценариев файл называется runs/<политика>_<сценарий>_<режим>.pkl.
"""
import sys, os, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from taris_sim.config import scenario
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import (UniformPolicy, CrushLikePolicy, CopysetsPolicy,
                              FilecoinInspiredPolicy, SiaInspiredPolicy, StorjInspiredPolicy,
                              DynamicECPolicy, AECInspiredPolicy, GradientBoostingPolicy,
                              K1PoissonGammaPolicy, K2LaplacePolicy, K2NUTSPolicy,
                              OraclePolicy, CrisisOraclePolicy)

out, which = sys.argv[1], sys.argv[2]
scn = sys.argv[3] if len(sys.argv) > 3 else "S2"
mode = sys.argv[4] if len(sys.argv) > 4 else "matched"
os.makedirs(out, exist_ok=True)
cfg = scenario(scn, world_mode=mode)
w = World(cfg); ch = Chain(w); ds = DataStream(cfg, len(w.nodes)); L = Ledger(w, ch, ds)
pol = {"uniform15": lambda: UniformPolicy(15), "uniform13": lambda: UniformPolicy(13), "crush13": lambda: CrushLikePolicy(w, 13), "copysets13": lambda: CopysetsPolicy(w, 13), "filecoin13": lambda: FilecoinInspiredPolicy(w, 13), "sia13": lambda: SiaInspiredPolicy(w, 13), "storj13": lambda: StorjInspiredPolicy(w, 13), "dynamic_ec13": lambda: DynamicECPolicy(w), "aec13": lambda: AECInspiredPolicy(w, 13), "gbm13": lambda: GradientBoostingPolicy(w, 13), "k1_pg13": lambda: K1PoissonGammaPolicy(w, 13), "k2_laplace13": lambda: K2LaplacePolicy(w, 13), "k2_nuts13": lambda: K2NUTSPolicy(w, 13), "oracle13": lambda: OraclePolicy(w, 13),
       "crisis_oracle13": lambda: CrisisOraclePolicy(w, 13)}[which]()
t = time.time()
res = L.run(pol)
un = res.unavailability()
m = res.metrics(); m["Недоступность, доля времени"] = float(un.unavail_share.mean())
if isinstance(pol, FilecoinInspiredPolicy):
    m["Штрафов за провал аудита"] = int(pol.faults)
    m["Списано залога, норм. ед."] = float(pol.slashed)
    m["Операторов с исчерпанным залогом"] = int(len(pol.exhausted_owners))
    m["Пустой пул допустимых узлов"] = int(pol.empty_eligible_fallbacks)
if isinstance(pol, SiaInspiredPolicy):
    m["Наблюдаемых аудитов Sia"] = int(sum(len(x) for x in pol.audit_history))
if isinstance(pol, StorjInspiredPolicy):
    m["Событий допуска ниже порога Storj"] = int(pol.disqualified)
    m["Размещений с ослаблением /24"] = int(pol.subnet_relaxations)
if isinstance(pol, DynamicECPolicy):
    m["Итоговое число фрагментов Dynamic-EC"] = int(pol.target_n)
    m["Доля провалов аудита Dynamic-EC"] = float(pol.audit_fail_rate)
if isinstance(pol, K2NUTSPolicy):
    m["NUTS: число подгонок"] = int(pol.fit_count)
    m["NUTS: время подгонки, с"] = float(pol.fit_seconds)
    m["NUTS: расхождения"] = int(pol.divergences)
    m["NUTS: максимальный R-hat"] = float(pol.max_rhat)
if isinstance(pol, AECInspiredPolicy):
    m["Аудитов AEC-DS"] = int(pol.audit_count)
    m["Проактивных переносов AEC-DS"] = int(pol.migration_candidates)
summary, crisis_table = res.crisis_report()
m.update(summary)
m["Время прогона, с"] = round(time.time() - t, 1)
sample = res.fragments[res.fragments.obj.isin(range(0, len(ds.objects), 97))]
name = which if (scn, mode) == ("S2", "matched") else f"{which}_{scn}_{mode}"
with open(f"{out}/{name}.pkl", "wb") as fh:
    pickle.dump({"metrics": m, "losses": res.losses, "repairs": res.repairs, "daily": res.daily,
                 "objects": res.objects[["created_epoch", "n", "lost_t"]], "fragments_sample": sample,
                 "unavail": un, "shocks": w.shocks, "disclosures": res.disclosures,
                 "migrations": res.migrations, "alarms": res.alarms, "crisis_table": crisis_table,
                 "crises": getattr(w, "crises", None),
                 "policy_state": ({"balance": pol.balance, "faults": pol.faults, "slashed": pol.slashed,
                                   "exhausted_owners": pol.exhausted_owners,
                                   "empty_eligible_fallbacks": pol.empty_eligible_fallbacks}
                                  if isinstance(pol, FilecoinInspiredPolicy) else
                                  {"audit_history": pol.audit_history} if isinstance(pol, SiaInspiredPolicy) else
                                  {"good": pol.good, "bad": pol.bad, "audit_count": pol.audit_count,
                                   "disqualified_events": pol.disqualified,
                                   "subnet_relaxations": pol.subnet_relaxations}
                                  if isinstance(pol, StorjInspiredPolicy) else None)}, fh)
print(pd.Series(m).to_string())
