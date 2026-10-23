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
7. Отсутствие утечки истины: нет избытка наблюдаемых событий перед внезапными кризисами и вне окон
   предвестника, ложные тревоги по наблюдениям неотличимы от настоящих окон, в журнале цепочки нет полей истины.
8. Исходы кризисов: гибель партии при износе, волна ухода и подавление регистраций, пожар, сбой ПО, бури и шоки.
9. Сигнал глазами сети: во сколько раз наблюдаемая частота событий у узлов-целей перед кризисом выше фона —
   для предсказуемых кризисов, ложных тревог и внезапных кризисов (есть ли агенту что ловить).

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


# ------------------------------------------------------------ Д3. отсутствие утечки истины
SUDDEN_W = {"storm": "storm", "wear": "wear", "exodus": "exodus", "software": "software", "fire": "storm"}


def excluded(w) -> dict:
    """По узлам: окна предвестника (у помеченных узлов) и периоды кризисов у узлов-целей —
    всё, что не является «фоном». Возвращает узел -> (начала, концы) объединённых интервалов."""
    cr, T = w.crisis, w.cfg.T_rep
    ex = {}
    for e in cr.episodes:
        if e["predictable"] or e["false_alarm"]:
            for i in cr.precursor_nodes(e):
                lo, hi = cr.precursor_span(e, i)
                if hi > lo:
                    ex.setdefault(int(i), []).append((lo, hi))
        if not e["false_alarm"]:
            for i in e["targets"]:
                ex.setdefault(int(i), []).append((e["tau"], e["end"] + T))
    out = {}
    for i, iv in ex.items():
        iv.sort()
        s, e = [iv[0][0]], [iv[0][1]]
        for a, b in iv[1:]:
            if a <= e[-1]:
                e[-1] = max(e[-1], b)
            else:
                s.append(a); e.append(b)
        out[i] = (np.array(s), np.array(e))
    return out


def clip(times, a, b, ex):
    """Экспозиция и число событий в [a, b) за вычетом исключённых интервалов ex = (начала, концы)."""
    if b <= a:
        return 0.0, 0
    n = int(np.searchsorted(times, b) - np.searchsorted(times, a))
    x = b - a
    s, e = ex
    for lo, hi in zip(np.maximum(s, a), np.minimum(e, b)):
        if hi > lo:
            x -= hi - lo
            n -= int(np.searchsorted(times, hi) - np.searchsorted(times, lo))
    return x, n


def observed_events(w, ch) -> dict:
    """События, которые видит сеть: короткие наблюдаемые простои (не длиннее T_rep, включая шум)
    и проваленные аудиты — моменты по узлам."""
    o = ch.observed
    o = o[o["dur"] <= w.cfg.T_rep]
    a = ch.audits[~ch.audits["passed_avail"]]
    return {"short": {i: np.sort(g["start"].to_numpy()) for i, g in o.groupby("node")},
            "audit": {i: np.sort(g["t"].to_numpy()) for i, g in a.groupby("node")}}


def background(w, ev, ex) -> dict:
    """Фоновая частота событий каждого узла: вне окон предвестника и периодов кризисов, по итоговой жизни."""
    nd = w.nodes
    r = {}
    for kind, by in ev.items():
        rate = np.zeros(len(nd))
        for i in range(len(nd)):
            x, n = clip(by.get(i, np.empty(0)), nd["alive_from"].iat[i], nd["alive_to"].iat[i], ex.get(i, (np.empty(0), np.empty(0))))
            rate[i] = n / x if x > 0 else 0.0
        r[kind] = rate
    return r


def window_excess(w, ev, ex, r0, e, kind, a, b, exclude_window=True):
    """Наблюдаемое и ожидаемое по фону число событий у узлов-целей эпизода в окне [a, b)."""
    nd = w.nodes
    O, E = 0, 0.0
    empty = (np.empty(0), np.empty(0))
    for i in e["targets"]:
        lo, hi = max(a, nd["alive_from"].iat[i]), min(b, nd["alive_to"].iat[i])
        x, n = clip(ev[kind].get(i, np.empty(0)), lo, hi, ex.get(int(i), empty) if exclude_window else empty)
        O += n; E += x * r0[kind][i]
    return O, E


def leakage(ws, chains, controls) -> pd.DataFrame:
    rows = []
    sud = {}
    fa_real = {}
    for w, ch in zip(ws, chains):
        ev, ex = observed_events(w, ch), excluded(w)
        r0 = background(w, ev, ex)
        for e in w.crisis.episodes:
            typ = e["type"]
            kind = "audit" if typ == "wear" else "short"
            if not e["predictable"] and not e["false_alarm"] and typ in SUDDEN_W:
                # перед внезапным кризисом: последние W̄ суток у узлов-целей (за вычетом чужих окон)
                Wm = type_params(w.cfg, SUDDEN_W[typ])["W_days"]
                O, E = window_excess(w, ev, ex, r0, e, kind, e["tau"] - Wm, e["tau"])
                o, m = sud.get((typ, kind), (0, 0.0))
                sud[(typ, kind)] = (o + O, m + E)
            if (e["predictable"] or e["false_alarm"]) and typ in ("storm", "wear", "software"):
                O, E = window_excess(w, ev, ex, r0, e, kind, e["w0"], e["tau"], exclude_window=False)
                if E > 0:
                    fa_real.setdefault((typ, e["false_alarm"]), []).append(O / E)
    for (typ, kind), (O, E) in sorted(sud.items()):
        rows.append((f"Перед внезапным кризисом: {typ}, {kind}", O, E, (O - E) / np.sqrt(max(E, 1e-12)), np.nan))
    # вне окон: те же события в мире S6 и в том же мире без кризисов (общие случайные числа)
    for kind in ("short", "audit"):
        o6, oc = 0, 0
        for (w, ch), (wc, chc) in zip(zip(ws, chains), controls):
            ev6, evc, ex = observed_events(w, ch), observed_events(wc, chc), excluded(w)
            nd = w.nodes
            for i in range(len(nd)):
                a, b = nd["alive_from"].iat[i], nd["alive_to"].iat[i]
                exi = ex.get(i, (np.empty(0), np.empty(0)))
                o6 += clip(ev6[kind].get(i, np.empty(0)), a, b, exi)[1]
                oc += clip(evc[kind].get(i, np.empty(0)), a, b, exi)[1]
        rows.append((f"Вне окон: S6 против мира без кризисов, {kind}", o6, oc, (o6 - oc) / np.sqrt(max(o6 + oc, 1)), np.nan))
    # ложные тревоги по наблюдениям неотличимы от окон настоящих кризисов (Манн–Уитни)
    for typ in ("storm", "wear", "software"):
        real, fa = fa_real.get((typ, False), []), fa_real.get((typ, True), [])
        if len(real) > 2 and len(fa) > 2:
            p = stats.mannwhitneyu(real, fa).pvalue
            rows.append((f"Избыток в окне: {typ}, настоящие (n={len(real)}) / ложные (n={len(fa)})",
                         float(np.mean(real)), float(np.mean(fa)), 0.0, p))
    df = pd.DataFrame(rows, columns=["Проверка", "Наблюдается", "Ожидается", "z", "p (МУ)"])
    df["Итог"] = np.where((df["z"].abs() < 3) & (df["p (МУ)"].fillna(1.0) > 0.001), "OK", "ПРОВЕРИТЬ")
    return df


# ------------------------------------------------------------ Д4. исходы кризисов
def outcomes(ws) -> pd.DataFrame:
    c = ws[0].cfg
    rows = []
    # износ: каждый живой узел партии гибнет с вероятностью p_death в равномерный момент периода
    p = type_params(c, "wear")
    n_alive, n_dead, u = 0, 0, []
    for w in ws:
        cr = w.crisis
        fo = cr.frames()["forced"]
        for e in cr.episodes:
            if e["type"] != "wear" or e["false_alarm"]:
                continue
            n_alive += len(cr.alive_base(e["targets"], e["tau"]))
            d = fo[(fo["reason"] == "wear_death") & (fo["episode"] == e["id"])]
            n_dead += len(d)
            u += list((d["t"].to_numpy() - e["tau"]) / p["period_days"])
    q = p["p_death"]
    rows.append(("Износ: доля погибших узлов партии", n_dead / max(n_alive, 1), q,
                 (n_dead - q * n_alive) / np.sqrt(n_alive * q * (1 - q)), np.nan))
    rows.append((f"Износ: момент гибели / период (n={len(u)})", float(np.mean(u)), 0.5,
                 (np.mean(u) - 0.5) / np.sqrt(1 / 12 / len(u)), stats.kstest(u, "uniform").pvalue))
    # массовый уход: первое событие потока (Gamma - 1) * lambda_churn на периоде; регистрации подавлены
    p = type_params(c, "exodus")
    o_l, m_l, v_l, o_j, n_j = 0, 0.0, 0.0, 0, 0
    for w in ws:
        cr = w.crisis
        fr = cr.frames()
        for e in cr.episodes:
            if e["type"] != "exodus" or e["false_alarm"]:
                continue
            tau, T = e["tau"], p["period_days"]
            for i in e["targets"]:
                lo, hi = max(tau, cr.a0[i]), min(tau + T, cr.a1[i])
                if hi > lo:
                    t = np.linspace(lo, hi, 200)
                    q = 1 - np.exp(-np.trapezoid(w.churn_rate(t - cr.join[i]) * (p["multiplier"] - 1), t))
                    m_l += q; v_l += q * (1 - q)
            fo = fr["forced"]
            o_l += int(((fo["reason"] == "exodus") & (fo["episode"] == e["id"])).sum())
            j = cr.join[e["targets"]]
            n_j += int(((j >= tau) & (j < tau + T)).sum())
            ca = fr["cancelled"]
            o_j += int(((ca["episode"] == e["id"]) & (ca["join_t"] >= tau)).sum())
    rows.append(("Массовый уход: уходы за период", o_l, m_l, (o_l - m_l) / np.sqrt(max(v_l, 1e-12)), np.nan))
    q = 1 - p["rho_join"]
    rows.append(("Массовый уход: доля отменённых регистраций", o_j / max(n_j, 1), q,
                 (o_j - q * n_j) / np.sqrt(max(n_j * q * (1 - q), 1e-12)), np.nan))
    # пожар: гибнут все живые узлы дата-центра; уязвимость ПО: затронуты ровно живые узлы версии
    fire_ok, sw_ok, sw_n, sw_d = True, True, 0, 0
    for w in ws:
        for e in w.crisis.episodes:
            if e["false_alarm"]:
                continue
            alive = len(w.alive_at(e["targets"], e["tau"]))
            if e["type"] == "fire":
                fire_ok &= e["destroyed"] and e["n_hit"] == alive
            if e["type"] == "software":
                ver = w.crisis.version_at(e["tau"])[e["targets"]]
                sw_ok &= bool((ver == e["cluster"]).all()) and e["n_hit"] == alive
                sw_n += 1; sw_d += int(e["destroyed"])
    rows.append(("Пожар: погибли все живые узлы дата-центра", float(fire_ok), 1.0, 0.0 if fire_ok else 99.0, np.nan))
    rows.append(("Сбой ПО: затронуты ровно живые узлы версии", float(sw_ok), 1.0, 0.0 if sw_ok else 99.0, np.nan))
    q = type_params(c, "software")["total_share"]
    rows.append((f"Сбой ПО: доля с гибелью данных (n={sw_n})", sw_d / max(sw_n, 1), q,
                 (sw_d - q * sw_n) / np.sqrt(max(sw_n * q * (1 - q), 1e-12)), np.nan))
    # буря и каскад: в согласованном мире — общий шок (Вейбулл длиннее франшизы, доля полной гибели),
    # в несогласованном — буря фиксированной длительности
    st = pd.concat([w.crises for w in ws], ignore_index=True)
    st = st[st["type"].isin(["storm", "cascade"]) & ~st["false_alarm"]]
    if c.world_mode == "matched":
        q = c.shock_total_share
        n, d = len(st), int(st["destroyed"].sum())
        rows.append(("Буря и каскад: доля полной гибели", d / n, q, (d - q * n) / np.sqrt(n * q * (1 - q)), np.nan))
        L = ((st["end"] - st["tau"]) / HOUR)[~st["destroyed"]].to_numpy()
        k, th, dh = c.shock_shape, c.shock_scale_hours[0], c.franchise_d_hours
        cdf = lambda l: 1 - np.exp(-(np.maximum(l, dh) / th) ** k + (dh / th) ** k)
        mean = stats.weibull_min(k, scale=th).expect(lambda l: l, lb=dh, conditional=True)
        rows.append((f"Буря и каскад: длительность шока, ч (n={len(L)})", L.mean(), mean,
                     0.0, stats.kstest(L, cdf).pvalue))
    else:
        L = ((st["end"] - st["tau"]) / HOUR).to_numpy()
        ok = np.allclose(L, c.storm_hours)
        rows.append(("Буря и каскад: длительность бури, ч", L.mean(), c.storm_hours, 0.0 if ok else 99.0, np.nan))
    df = pd.DataFrame(rows, columns=["Проверка", "Наблюдается", "Ожидается", "z", "p (КС)"])
    return verdict(df)


# ------------------------------------------------------------ Д5. сигнал глазами сети
SIGNAL_KIND = {"storm": "short", "cascade": "short", "software": "short", "fire": "short", "wear": "audit"}


def signal(ws, chains) -> pd.DataFrame:
    """Во сколько раз частота наблюдаемых событий у узлов-целей перед кризисом выше их фона:
    за последние сутки (у каскада — за всё окно от первичного кризиса). У массового ухода —
    уходы узлов региона за последние 7 суток против ожидаемых по обычной интенсивности ухода.
    Предсказуемые кризисы и ложные тревоги должны давать отношение > 1, внезапные — около 1."""
    agg = {}
    for w, ch in zip(ws, chains):
        ev, ex = observed_events(w, ch), excluded(w)
        r0 = background(w, ev, ex)
        cr = w.crisis
        leave = w.nodes["leave_t"].to_numpy()
        for e in cr.episodes:
            typ = e["type"]
            grp = "ложные" if e["false_alarm"] else ("предсказуемые" if e["predictable"] else "внезапные")
            if typ == "exodus":
                a, b = e["tau"] - 7.0, e["tau"]
                O = int(((leave[e["targets"]] >= a) & (leave[e["targets"]] < b)).sum())
                E = 0.0
                for i in e["targets"]:
                    lo, hi = max(a, cr.a0[i]), min(b, cr.a1[i])
                    if hi > lo:
                        t = np.linspace(lo, hi, 50)
                        E += 1 - np.exp(-np.trapezoid(w.churn_rate(t - cr.join[i]), t))
                kind = "уходы"
            else:
                kind = SIGNAL_KIND[typ]
                a = e["w0"] if typ == "cascade" else e["tau"] - 1.0
                O, E = window_excess(w, ev, ex, r0, e, kind, a, e["tau"], exclude_window=False)
            o, m, n = agg.get((typ, grp, kind), (0, 0.0, 0))
            agg[(typ, grp, kind)] = (o + O, m + E, n + 1)
    rows = []
    for (typ, grp, kind), (O, E, n) in sorted(agg.items()):
        z = (O - E) / np.sqrt(max(E, 1e-12))
        if grp == "внезапные":
            res = "OK" if abs(z) < 3 else "ПРОВЕРИТЬ"
        else:
            res = "сигнал есть" if z >= 3 else "сигнал слабый"
        rows.append((f"{typ}, {grp} (n={n}), {kind}", O, E, O / max(E, 1e-12), z, res))
    return pd.DataFrame(rows, columns=["Эпизоды", "Наблюдается", "Фон", "Отношение", "z", "Итог"])


def view_has_no_truth(w, ch) -> list:
    """В «взгляде сети» (журнал цепочки по эпохам) нет полей истины."""
    allowed = {"outages_started": {"node", "start"}, "outages_ongoing": {"node", "start"},
               "outages_ended": {"node", "start", "end", "dur"}, "audits": {"node", "t", "epoch", "passed_avail"}}
    ch.reset()
    ok = True
    for _ in range(60):
        st = ch.step()
        ok &= all(set(st[k].columns) <= cols for k, cols in allowed.items())
    ch.reset()
    return [("Журнал цепочки без полей истины (источник, шум, предвестник, будущий конец простоя)", ok)]


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
        print(f"\n=== Исходы кризисов, {mode}, зёрен {n_seeds}")
        print(outcomes(ws).round(4).to_string(index=False))
        chains = [Chain(w) for w in ws]
        print(f"\n=== Сила предвестника по каналам, {mode}, зёрен {n_seeds}")
        print(precursor_strength(ws, chains).round(2).to_string(index=False))
        n_ctrl = min(3, n_seeds)
        controls = []
        for w in ws[:n_ctrl]:
            wc = World(scenario("S6", world_mode=mode, seed=w.cfg.seed, crises_enabled=False))
            controls.append((wc, Chain(wc)))
        print(f"\n=== Отсутствие утечки истины, {mode}, зёрен {n_seeds} (мир без кризисов — {n_ctrl})")
        print(leakage(ws, chains, controls).round(3).to_string(index=False))
        for name, ok in view_has_no_truth(ws[0], chains[0]):
            print(f"{name}: {'OK' if ok else 'ОШИБКА'}")
        print(f"\n=== Сигнал глазами сети перед кризисом, {mode}, зёрен {n_seeds}")
        print(signal(ws, chains).round(2).to_string(index=False))
