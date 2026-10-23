"""Проверки кризисов и предвестников (шаг 3, пункты 3.7 и 3.9).

Мир S6 с зерном 1:
1. Число кризисов, ложных тревог и доля предсказуемых — против параметров config.
2. Согласованность жизни узлов: уходы не позже базовых, нет событий после ухода и у отменённых узлов,
   предвестники только у предсказуемых кризисов и ложных тревог и только внутри окна.
3. Общие случайные числа: мир с кризисами против того же мира без кризисов.
4. Воспроизводимость.
По нескольким зёрнам (кризисов в одном мире мало):
5. Параметры эпизодов: длина окна, доля узлов с предвестником, каскады, задержка сбоя ПО.
6. Сила предвестника по каналам: число событий против ожидаемого по формулам 3.9, провалы аудитов.

    python scripts/check_crisis.py [число зёрен, по умолчанию 10]
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import pandas as pd
from scipy import stats
from taris_sim.config import scenario, YEAR, HOUR
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.crisis import type_params

PRE_TYPES = ("storm", "wear", "exodus", "software")       # типы с окном предвестника, длина которого разыгрывается


def z(obs, exp):
    return (obs - exp) / np.sqrt(max(exp, 1e-12))


def counts(w: World) -> pd.DataFrame:
    c, cr, nd = w.cfg, w.crises, w.nodes
    years = c.horizon_days / YEAR
    n_batch = nd.groupby(["dc", "hw_true"]).ngroups
    n_cl = {"storm": c.n_dc, "wear": n_batch, "exodus": c.n_regions, "fire": c.n_dc}
    rows = []
    for typ, n in n_cl.items():
        p = type_params(c, typ)
        real, fa = cr[(cr.type == typ) & ~cr.false_alarm], cr[(cr.type == typ) & cr.false_alarm]
        exp = p["rate"] * years * n
        rows.append((f"Кризисы: {typ}", len(real), exp, z(len(real), exp)))
        if p.get("q", 0) > 0:
            exp_fa = p["q"] * p["pi"] * exp
            rows.append((f"Ложные тревоги: {typ}", len(fa), exp_fa, z(len(fa), exp_fa)))
    # уязвимость ПО: каждая выпущенная версия неудачна с вероятностью rate * интервал / год
    p = type_params(c, "software")
    p_bad = min(1.0, p["rate"] * c.sw_release_every_days / YEAR)
    n_rel = len(w.versions) - 1
    real = cr[(cr.type == "software") & ~cr.false_alarm]
    rows.append(("Кризисы: software (на версию)", len(real), p_bad * n_rel, z(len(real), p_bad * n_rel)))
    # каскад: у каждой бури и пожара — Bernoulli(p) на каждый другой дата-центр провайдера
    p = type_params(c, "cascade")
    prov = w.dc_provider
    prim = cr[cr.type.isin(["storm", "fire"]) & ~cr.false_alarm]
    sib = sum(int((prov == prov[int(dc)]).sum()) - 1 for dc in prim.cluster)
    casc = int((cr.type == "cascade").sum())
    rows.append(("Кризисы: cascade", casc, p["p"] * sib, z(casc, p["p"] * sib)))
    # доля предсказуемых среди настоящих кризисов (кроме каскадов и пожаров: там pi = 1 и 0)
    pis = cr[~cr.false_alarm & cr.type.isin(["storm", "wear", "exodus", "software"])]
    pi = pis.type.map(lambda t: type_params(c, t)["pi"]).to_numpy()
    obs, exp, var = int(pis.predictable.sum()), pi.sum(), (pi * (1 - pi)).sum()
    rows.append(("Предсказуемые кризисы", obs, exp, (obs - exp) / np.sqrt(var)))
    df = pd.DataFrame(rows, columns=["Проверка", "Наблюдается", "Ожидается", "z"])
    df["Итог"] = np.where(df["z"].abs() < 3, "OK", "ПРОВЕРИТЬ")
    return df


def consistency(w: World) -> list:
    nd, cr = w.nodes, w.crises
    live = ~nd["cancelled"]
    res = []
    res.append(("Уход не позже базового", bool((nd.loc[live, "leave_t"] <= nd.loc[live, "leave_t_base"]).all())))
    out_end = nd["alive_to"].to_numpy()[w.outages["node"]]
    res.append(("Нет простоев после ухода", bool((w.outages["start"].to_numpy() < out_end).all())))
    canc = set(np.flatnonzero(nd["cancelled"].to_numpy()))
    res.append(("У отменённых узлов нет событий",
                not canc & set(w.outages["node"]) and not canc & set(w.silent["node"]) and not canc & set(w.pre_events["node"])))
    pe = w.pre_events.merge(cr[["id", "predictable", "false_alarm", "w0", "tau"]], left_on="episode", right_on="id")
    res.append(("Предвестники только у предсказуемых и ложных", bool((pe["predictable"] | pe["false_alarm"]).all())))
    res.append(("Предвестники внутри окна", bool(((pe["start"] >= pe["w0"] - 1e-9) & (pe["start"] < pe["tau"])).all())))
    return res


def crn(w: World) -> list:
    """Тот же мир без кризисов: события узла до его итогового ухода совпадают."""
    w0 = World(scenario("S6", world_mode=w.cfg.world_mode, seed=w.cfg.seed, crises_enabled=False))
    a1 = w.nodes["alive_to"].to_numpy()
    raw0 = w0.raw_own[w0.raw_own["start"].to_numpy() < a1[w0.raw_own["node"]]].reset_index(drop=True)
    sil0 = w0.silent[w0.silent["t"].to_numpy() < a1[w0.silent["node"]]]
    base_sil = w.silent.iloc[:len(sil0)].reset_index(drop=True)
    cols = ["ctype", "cluster", "start", "end", "destroyed"]
    nd, nd0 = w.nodes, w0.nodes
    return [
        ("beta_0 совпадает", w.beta0 == w0.beta0),
        ("Базовая жизнь совпадает", bool(nd["leave_t_base"].equals(nd0["leave_t"]) and nd["join_t_base"].equals(nd0["join_t"]))),
        ("Собственные простои совпадают", raw0.equals(w.raw_own)),
        ("Тихие потери (без предвестников) совпадают", sil0.reset_index(drop=True).equals(base_sil)),
        ("Шоки провайдеров и регионов совпадают", w.shocks[cols].equals(w0.shocks[cols])),
    ]


def verdict(df: pd.DataFrame) -> pd.DataFrame:
    """OK, если |z| < 3 и (если есть) p-значение теста Колмогорова–Смирнова > 0,001."""
    p = df["p (КС)"].fillna(1.0) if "p (КС)" in df else 1.0
    df["Итог"] = np.where((df["z"].abs() < 3) & (p > 0.001), "OK", "ПРОВЕРИТЬ")
    return df


def window_dist(c, mean_days):
    """Теоретическое распределение длины окна предвестника (пункт 3.9)."""
    if c.world_mode == "matched":
        return stats.gamma(c.pre_k_W, scale=mean_days / c.pre_k_W)
    s = c.pre_sigma_W
    return stats.lognorm(s, scale=np.exp(np.log(mean_days) - s * s / 2))


def episode_params(ws: list) -> pd.DataFrame:
    """Д1. Параметры эпизодов по нескольким мирам."""
    c = ws[0].cfg
    cr = pd.concat([w.crises.assign(seed=w.cfg.seed) for w in ws], ignore_index=True)
    rows = []
    # длина окна W: среднее (z) и форма (Колмогоров–Смирнов) — у всех эпизодов типа, W разыгрывается всегда
    for typ in PRE_TYPES:
        Wm = type_params(c, typ)["W_days"]
        W = cr.loc[cr["type"] == typ, "W"].to_numpy()
        d = window_dist(c, Wm)
        rows.append((f"Окно W, сут: {typ} (n={len(W)})", W.mean(), Wm,
                     (W.mean() - Wm) / (d.std() / np.sqrt(len(W))), stats.kstest(W, d.cdf).pvalue))
    # доля узлов кластера-цели с предвестником: биномиальная проверка
    for typ in PRE_TYPES + ("cascade",):
        s = type_params(c, typ)["s"]
        e = cr[(cr["type"] == typ) & (cr["predictable"] | cr["false_alarm"])]
        n, obs = int(e["n_targets"].sum()), int(e["n_xi"].sum())
        rows.append((f"Доля узлов с предвестником: {typ}", obs / max(n, 1), s,
                     (obs - s * n) / np.sqrt(max(n * s * (1 - s), 1e-12)), np.nan))
    # каскад: каждый соседний дата-центр провайдера — испытание Бернулли с вероятностью p
    p = type_params(c, "cascade")
    trials, obs, delays = 0, 0, []
    for w in ws:
        prov, cw = w.dc_provider, w.crises
        prim = cw[cw["type"].isin(["storm", "fire"]) & ~cw["false_alarm"]]
        trials += sum(int((prov == prov[int(dc)]).sum()) - 1 for dc in prim["cluster"])
        cas = cw[cw["type"] == "cascade"]
        obs += len(cas)
        delays += list(cas["W"].to_numpy() / HOUR)
    rows.append(("Каскад: доля соседних дата-центров", obs / max(trials, 1), p["p"],
                 (obs - p["p"] * trials) / np.sqrt(trials * p["p"] * (1 - p["p"])), np.nan))
    dm = p["delay_hours"]
    delays = np.asarray(delays)
    rows.append((f"Каскад: задержка, ч (n={len(delays)})", delays.mean(), dm,
                 (delays.mean() - dm) / (dm / np.sqrt(len(delays))), stats.kstest(delays, stats.expon(scale=dm).cdf).pvalue))
    # уязвимость ПО: задержка от выпуска версии до сбоя ~ Exp(trigger_days)
    trig = type_params(c, "software")["trigger_days"]
    dl = np.concatenate([(w.crises.loc[w.crises["type"] == "software", "tau"].to_numpy()
                          - w.crisis.release[w.crises.loc[w.crises["type"] == "software", "cluster"].astype(int).to_numpy()])
                         for w in ws])
    rows.append((f"Сбой ПО: задержка после выпуска, сут (n={len(dl)})", dl.mean(), trig,
                 (dl.mean() - trig) / (trig / np.sqrt(len(dl))), stats.kstest(dl, stats.expon(scale=trig).cdf).pvalue))
    df = pd.DataFrame(rows, columns=["Проверка", "Наблюдается", "Ожидается", "z", "p (КС)"])
    return verdict(df)


def precursor_strength(ws: list, chains: list) -> pd.DataFrame:
    """Д2. Сила предвестника по каналам: число предвестниковых событий (по истине, до обрезки по итоговой
    жизни) против ожидаемого по формулам 3.9 — суммарно по эпизодам и зёрнам."""
    obs, exp, var = {}, {}, {}
    for w in ws:
        cr = w.crisis
        fr = cr.frames()
        ep_type = {e["id"]: e["type"] for e in cr.episodes}
        ep_tau = {e["id"]: e["tau"] for e in cr.episodes}
        cnt = []
        pe = fr["pre_events"]
        cnt += [(ep_type[e], ch) for e, ch in zip(pe["episode"], pe["channel"])]
        cnt += [(ep_type[e], "silent") for e in fr["pre_silent"]["episode"]]
        fo = fr["forced"]
        cnt += [(ep_type[e], "churn") for e in fo.loc[fo["reason"] == "pre_churn", "episode"]]
        ca = fr["cancelled"]
        cnt += [(ep_type[e], "join") for e, j in zip(ca["episode"], ca["join_t"]) if j < ep_tau[e]]
        for key in cnt:
            obs[key] = obs.get(key, 0) + 1
        for e in cr.episodes:
            if e["predictable"] or e["false_alarm"]:
                for ch, (m, v) in cr.expected_precursor(e).items():
                    exp[(e["type"], ch)] = exp.get((e["type"], ch), 0.0) + m
                    var[(e["type"], ch)] = var.get((e["type"], ch), 0.0) + v
    rows = [(f"Предвестник: {typ}, {ch}", obs.get((typ, ch), 0), exp[(typ, ch)],
             (obs.get((typ, ch), 0) - exp[(typ, ch)]) / np.sqrt(max(var[(typ, ch)], 1e-12)))
            for typ, ch in sorted(exp)]
    # провалы аудитов в окнах износа: доп. провал среди аудитов без шумового провала
    o, m, v = 0, 0.0, 0.0
    for w, ch in zip(ws, chains):
        a = ch.audits
        c = w.cfg
        node, t = a["node"].to_numpy(), a["t"].to_numpy()
        extra = np.zeros(len(a))
        for wr in w.pre_audit.itertuples(index=False):
            sel = (node == wr.node) & (t >= wr.start) & (t < wr.end)
            extra[sel] += c.audit_noise * (wr.rho - 1.0) * w.crisis.g_shape((t[sel] - wr.w0) / max(wr.tau - wr.w0, 1e-12))
        q = np.minimum(extra / (1.0 - c.audit_noise), 1.0)[~a["noise_fail"].to_numpy()]
        o += int(a["pre_fail"].sum()); m += q.sum(); v += (q * (1 - q)).sum()
    rows.append(("Предвестник: wear, провалы аудитов", o, m, (o - m) / np.sqrt(max(v, 1e-12))))
    df = pd.DataFrame(rows, columns=["Проверка", "Наблюдается", "Ожидается", "z"])
    df["Итог"] = np.where(df["z"].abs() < 3, "OK", "ПРОВЕРИТЬ")
    return df


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for mode in ("matched", "mismatched"):
        w = World(scenario("S6", world_mode=mode))
        print(f"\n=== Мир S6: {mode}, узлов {len(w.nodes)}, эпизодов {len(w.crises)}, хэш {w.fingerprint()}")
        print(counts(w).round(2).to_string(index=False))
        for name, ok in consistency(w) + crn(w):
            print(f"{name}: {'OK' if ok else 'ОШИБКА'}")
    f1, f2 = World(scenario("S6")).fingerprint(), World(scenario("S6")).fingerprint()
    f3 = World(scenario("S6", seed=2)).fingerprint()
    print(f"\nВоспроизводимость: зерно 1 дважды -> {f1} / {f2} ({'OK' if f1 == f2 else 'ОШИБКА'}); зерно 2 -> {f3} ({'OK' if f3 != f1 else 'ОШИБКА'})")

    for mode in ("matched", "mismatched"):
        ws = [World(scenario("S6", world_mode=mode, seed=s)) for s in range(1, n_seeds + 1)]
        print(f"\n=== Параметры эпизодов, {mode}, зёрен {n_seeds}")
        print(episode_params(ws).round(4).to_string(index=False))
        chains = [Chain(w) for w in ws]
        print(f"\n=== Сила предвестника по каналам, {mode}, зёрен {n_seeds}")
        print(precursor_strength(ws, chains).round(2).to_string(index=False))
