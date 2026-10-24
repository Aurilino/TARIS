"""Проверки устройств пользователей (шаг 3, пункты 3.7 и 3.10) на мире S8, оба режима.

1. Частоты у устройств против параметров: простои длиннее франшизы, доля полной гибели, уход, тихие потери,
   микропрерывания (z-тесты, ожидаемое — по формулам 3.10 с множителями класса).
2. Численность: живых устройств примерно в r_dev раз больше, чем узлов в дата-центрах.
3. Кризисы дата-центров (буря, пожар, каскад, износ партии) не задевают устройства.
4. Общие случайные числа: узлы в дата-центрах в S8 совпадают с S6 (признаки, жизнь, beta_0, простои,
   тихие потери, микропрерывания).

    python scripts/check_devices.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from taris_sim.config import scenario
from taris_sim.scenario import MAIN
from taris_sim.world import World
from taris_sim.chain import Chain

ok_all = True


def z(o, e):
    return (o - e) / np.sqrt(max(e, 1e-12))


def frequencies(w, ch) -> pd.DataFrame:
    c, nd = w.cfg, w.nodes
    dev = nd["is_device"].to_numpy()
    idx = np.flatnonzero(dev)
    rows = []
    # простои длиннее франшизы: ожидаемое — интеграл lambda_dev по итоговой жизни
    E = w.expected_own_counts()
    O = w.raw_own_counts.to_numpy()
    rows.append(("Устройства: простои длиннее франшизы", O[dev].sum(), E[dev].sum(), z(O[dev].sum(), E[dev].sum())))
    # доля полной гибели (длиннее T_rep) — по истинной phi с множителем длительности
    own = w.raw_own[np.isin(w.raw_own["node"], idx)]
    ph = np.array([w.true_phi(i) for i in range(len(nd))])[own["node"]]
    obs, exp, var = int((own["dur"] > c.T_rep).sum()), ph.sum(), (ph * (1 - ph)).sum()
    rows.append(("Устройства: простои длиннее T_rep", obs, exp, (obs - exp) / np.sqrt(var)))
    # уход: вероятность уйти за базовую жизнь (до кризисов) — по интенсивности ухода устройства
    exp_l, var_l = 0.0, 0.0
    for i in idx:
        a0, j0 = nd["alive_from"].iat[i], nd["join_t"].iat[i]
        if "join_t_base" in nd:
            j0 = nd["join_t_base"].iat[i]
        t = np.linspace(a0, c.horizon_days, 400)
        if t[-1] <= t[0]:
            continue
        q = 1 - np.exp(-np.trapezoid(w.churn_rate_node(int(i), t - j0), t))
        exp_l += q; var_l += q * (1 - q)
    obs_l = int(np.isfinite(nd["leave_t_base"].to_numpy()[idx]).sum())
    rows.append(("Устройства: уход из сети (базовая жизнь)", obs_l, exp_l, (obs_l - exp_l) / np.sqrt(var_l)))
    # тихие потери и микропрерывания: интенсивность x итоговая жизнь
    expo = (nd["alive_to"] - nd["alive_from"]).clip(lower=0).to_numpy()
    sil = w.silent[np.isin(w.silent["node"], idx)]
    exp_s = (expo * nd["lam_silent"].to_numpy())[dev].sum()
    rows.append(("Устройства: тихие потери", len(sil), exp_s, z(len(sil), exp_s)))
    mic = ch.micro[np.isin(ch.micro["node"], idx)]
    exp_m = (expo * nd["lam_micro"].to_numpy())[dev].sum()
    rows.append(("Устройства: микропрерывания", len(mic), exp_m, z(len(mic), exp_m)))
    df = pd.DataFrame(rows, columns=["Проверка", "Наблюдается", "Ожидается", "z"])
    df["Итог"] = np.where(df["z"].abs() < 3, "OK", "ПРОВЕРИТЬ")
    return df


def report(name, ok, extra=""):
    global ok_all
    ok_all &= bool(ok)
    print(f"{name}: {'OK' if ok else 'ОШИБКА'}{(' — ' + extra) if extra else ''}")


for mode in ("matched", "mismatched"):
    print(f"\n=== S8, основной мир, {mode}")
    w8 = World(scenario("S8", world_mode=mode, **MAIN)); ch8 = Chain(w8)
    w6 = World(scenario("S6", world_mode=mode, **MAIN)); ch6 = Chain(w6)
    nd, dev = w8.nodes, w8.nodes["is_device"].to_numpy()
    f = frequencies(w8, ch8)
    print(f.round(2).to_string(index=False))
    ok_all &= bool((f["Итог"] == "OK").all())
    # численность
    al = lambda m, t: int(((nd["alive_from"] <= t) & (nd["alive_to"] > t)).to_numpy()[m].sum())
    ratios = [al(dev, t) / max(al(~dev, t), 1) for t in range(90, w8.cfg.horizon_days, 30)]
    r = w8.cfg.dev_ratio
    report(f"Численность: устройств к узлам в ДЦ от {min(ratios):.1f} до {max(ratios):.1f} (цель {r:.0f})",
           0.5 * r <= np.mean(ratios) <= 1.5 * r, f"в среднем {np.mean(ratios):.2f}")
    # кризисы дата-центров не задевают устройства
    hit = sum(1 for e in w8.crisis.episodes if e["type"] in ("storm", "fire", "wear", "cascade") and dev[e["targets"]].any())
    report("Кризисы дата-центров не задевают устройства", hit == 0, f"эпизодов с устройствами {hit}")
    # общие случайные числа: узлы в ДЦ совпадают с S6
    ns = len(w6.nodes)
    cols = ["owner", "join_t", "dc", "provider", "region", "hw_true", "theta", "leave_t_base", "lam_avg", "lam_micro", "lam_silent"]
    same = (nd.iloc[:ns][cols].reset_index(drop=True).equals(w6.nodes[cols].reset_index(drop=True)) and w8.beta0 == w6.beta0
            and w8.raw_own[w8.raw_own["node"] < ns].reset_index(drop=True).equals(w6.raw_own)
            and ch8.micro[ch8.micro["node"] < ns].reset_index(drop=True).equals(ch6.micro.reset_index(drop=True)))
    report("Узлы в ДЦ в S8 совпадают с S6 (признаки, жизнь, beta_0, простои, микропрерывания)", same)

print("\nИтог:", "все проверки пройдены" if ok_all else "ЕСТЬ ОШИБКИ")
sys.exit(0 if ok_all else 1)
