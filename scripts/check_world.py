"""Проверки корректности модуля world (шаг 3, пункт 3.7): сгенерированный мир совпадает с заданными параметрами."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from taris_sim.config import scenario, YEAR
from taris_sim.world import World, CLUSTER_TYPES


def z(obs, exp):
    return (obs - exp) / np.sqrt(max(exp, 1e-12))


def check(w: World) -> pd.DataFrame:
    c = w.cfg
    rows = []
    # 1. Частота собственных простоев
    E = w.expected_own_counts()
    O = w.raw_own_counts.to_numpy()
    rows.append(("Собственные простои, всего", O.sum(), E.sum(), z(O.sum(), E.sum())))
    # 2. Доля полной гибели среди собственных простоев
    own = w.raw_own
    phi_true = np.array([w.true_phi(i) for i in range(len(w.nodes))])
    ph = phi_true[own["node"]]
    exp_total, var_total = ph.sum(), (ph * (1 - ph)).sum()
    obs_total = int((own["dur"] > c.T_rep).sum())
    rows.append(("Собственные простои дольше T_rep", obs_total, exp_total, (obs_total - exp_total) / np.sqrt(var_total)))
    # 3. Шоки кластеров по типам
    if c.shocks_enabled:
        n_cl = {"dc": c.n_dc, "provider": c.n_providers, "region": c.n_regions}
        for ti, ct in enumerate(CLUSTER_TYPES):
            exp = c.shock_rate_per_year[ti] / YEAR * c.horizon_days * n_cl[ct]
            obs = int((w.shocks["ctype"] == ct).sum())
            rows.append((f"Шоки: {ct}", obs, exp, z(obs, exp)))
    # 4. Уход узлов
    nd = w.nodes
    exp_leave = 0.0
    for i in range(len(nd)):
        a0, j0 = nd["alive_from"].iat[i], nd["join_t"].iat[i]
        t = np.linspace(a0, c.horizon_days, 400)
        if t[-1] <= t[0]:
            continue
        cum = np.trapezoid(w.churn_rate(t - j0), t)
        exp_leave += 1 - np.exp(-cum)
    obs_leave = int(np.isfinite(nd["leave_t"]).sum())
    rows.append(("Уход из сети", obs_leave, exp_leave, z(obs_leave, exp_leave)))
    # 5. Тихие потери
    expo = (nd["alive_to"] - nd["alive_from"]).clip(lower=0).sum()
    exp_sil = expo * c.silent_rate
    rows.append(("Тихие потери", len(w.silent), exp_sil, z(len(w.silent), exp_sil)))
    df = pd.DataFrame(rows, columns=["Проверка", "Наблюдается", "Ожидается", "z"])
    df["Итог"] = np.where(df["z"].abs() < 3, "OK", "ПРОВЕРИТЬ")
    return df


if __name__ == "__main__":
    for mode in ("matched", "mismatched"):
        w = World(scenario("S2", world_mode=mode))
        print(f"\n=== Мир: {mode}, узлов {len(w.nodes)}, хэш {w.fingerprint()}")
        print(check(w).round(2).to_string(index=False))
    # 6. Воспроизводимость
    f1 = World(scenario("S2")).fingerprint()
    f2 = World(scenario("S2")).fingerprint()
    f3 = World(scenario("S2", seed=2)).fingerprint()
    print(f"\nВоспроизводимость: зерно 1 дважды -> {f1} / {f2} ({'OK' if f1 == f2 else 'ОШИБКА'}); зерно 2 -> {f3} ({'OK' if f3 != f1 else 'ОШИБКА'})")
