"""Проверка модели риска агента TARIS (шаг 5, пункт 5.1, подшаг А8).

Основной мир, согласованный режим, сценарий S2 (с шоками кластеров), несколько зёрен. Агент учится по ходу
прогона (размещение — оракул, на оценки агента оно не влияет) и обучается на моментах 180, 365 и 730 суток.
Оценки сравниваются с истиной мира, которую сам агент не видит.

1. Таблица опыта по шагам эпох совпадает с прямым расчётом по chain.observed на момент t (без будущего).
2. Частота: калибровка (сумма фактических случаев = сумме ожидаемых), коэффициенты класса оборудования и
   возраста против истинных (в пределах 3 робастных стандартных ошибок — с кластеризацией по узлам: в мире
   из 300 узлов средний скрытый множитель по классам заметно разный), избыточная дисперсия.
3. Кредибилити: оценка неоднородности a против истинной; у узлов с большим опытом ошибка ln(оценка/истина)
   меньше, чем у оценки класса; ранговая корреляция с истинным риском.
4. Интенсивность потери фрагмента (что идёт в резерв): сумма по живым узлам против истинной.
5. Шоки кластеров: найденные эпизоды против истинных шоков мира (точность, полнота), тип кластера.
6. Тихие потери: оценка против истинной.

    python scripts/check_risk.py [число зёрен, по умолчанию 3]
"""
import sys, os, time
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from taris_sim.config import scenario
from taris_sim.scenario import MAIN
from taris_sim.world import World
from taris_sim.chain import Chain
from taris_sim.data import DataStream
from taris_sim.ledger import Ledger
from taris_sim.policy import OraclePolicy
from taris_sim.experience import Experience
from taris_sim.risk import RiskModel, EstimatedRates
from taris_sim.reserve import TrueRates

TS = (180, 365, 730)
FAIL = []


def check(ok, msg):
    print(("  ок   " if ok else "  ОШИБКА ") + msg)
    if not ok:
        FAIL.append(msg)


class Learner(OraclePolicy):
    """Оракул размещает, агент рядом копит опыт и обучается в моменты TS."""

    def __init__(self, w):
        super().__init__(w, 16)
        self.exp, self.out, self.snap = Experience(w), {}, {}

    def on_epoch(self, t, view):
        self.exp.update(t, view)
        T = int(t)
        if T in (100,) + TS:
            self.snap[T] = self.exp.intervals(t)
        if T in TS:
            m = RiskModel(self.exp).fit(t); m.refresh(t)
            mv = RiskModel(self.exp, claimed=False).fit(t); mv.refresh(t)
            self.out[T] = (m, mv)
        return super().on_epoch(t, view)


def one(seed):
    cfg = scenario("S2", seed=seed, **MAIN)
    w = World(cfg); ch = Chain(w); L = Ledger(w, ch, DataStream(cfg, len(w.nodes)))
    p = Learner(w)
    t0 = time.time()
    L.run(p, epochs=max(TS) + 1)
    print(f"\n=== зерно {seed}: прогон {time.time() - t0:.0f} с, узлов {len(w.nodes)}")
    o, nd, tr = ch.observed, w.nodes, TrueRates(w)

    print("1. Таблица опыта")
    for T, iv in p.snap.items():
        ref = o[o.start < T].copy()
        ref.loc[ref.end > T, "end"] = np.inf
        a = np.nan_to_num(iv[["node", "start", "end"]].to_numpy(), posinf=1e9)
        b = np.nan_to_num(ref.sort_values(["node", "start"])[["node", "start", "end"]].to_numpy(), posinf=1e9)
        check(a.shape == b.shape and np.array_equal(a, b), f"t = {T}: {len(a)} простоев совпадают с chain.observed до t")

    rows = []
    for T, (m, mv) in p.out.items():
        print(f"t = {T}")
        alive = np.flatnonzero((nd.join_t.to_numpy() <= T) & (nd.alive_to.to_numpy() > T))
        lt = w.rate(alive, T)
        tab = m.exp.table(T, *([m.exp.intervals(T)] + [m.exp.shock_episodes(T)[1]]))
        tab = tab[tab.e > 0]
        # 2. частота
        names = (["const"] + [f"reg{j}" for j in range(1, m.reg.n_region)] + ["age", "micro"]
                 + [f"hw{j}" for j in range(1, m.reg.n_hw)] + ["device"])
        names = [n_ for n_, k in zip(names, m.keep) if k]
        b = dict(zip(names, m.beta)); se = dict(zip(names, np.sqrt(np.diag(m.cov_robust))))
        true = {"hw1": cfg.beta_hw[1] - cfg.beta_hw[0], "hw2": cfg.beta_hw[2] - cfg.beta_hw[0], "age": cfg.kappa_age - 1}
        print(f"  коэффициенты: " + ", ".join(f"{k} {b[k]:+.2f}±{se[k]:.2f} (истина {v:+.2f})" for k, v in true.items())
              + f"; Пирсон/df {m.pearson:.2f}")
        if T == max(TS):
            for k, v in true.items():
                check(abs(b[k] - v) < 3 * se[k], f"{k}: оценка в пределах 3 робастных ст. ошибок от истины")
        # 3. кредибилити
        lc, lk = m.lam_class(alive, T), m.lam(alive, T)
        rk = stats.spearmanr(lk, lt)[0]; rc = stats.spearmanr(lc, lt)[0]
        old = m.w_exp[alive] >= 5
        ec, ek = np.log(lc[old] / lt[old]), np.log(lk[old] / lt[old])
        rm_c, rm_k = np.sqrt(((ec - ec.mean()) ** 2).mean()), np.sqrt(((ek - ek.mean()) ** 2).mean())
        print(f"  a: {m.a_hat:.2f} (истина {cfg.heterogeneity_a}); ранговая корреляция с истиной: класс {rc:.2f}, "
              f"с кредибилити {rk:.2f}; разброс ln-ошибки у опытных узлов: {rm_c:.2f} -> {rm_k:.2f}")
        # 4. интенсивность потери фрагмента
        er = EstimatedRates(w, m)
        fe, ft = er.fragment_loss(alive, T).sum(), tr.fragment_loss(alive, T).sum()
        fev = EstimatedRates(w, mv).fragment_loss(alive, T).sum()
        print(f"  потеря фрагмента, сумма по живым узлам: агент {fe:.3f}, только проверяемые {fev:.3f}, истина {ft:.3f}")
        # 5. шоки
        ep, sh = m.episodes, w.shocks[w.shocks.start < T]
        hit = [((sh.ctype == r.ctype) & (sh.cluster == r.cluster) & (abs(sh.start - r.start) < 0.02)).any()
               for r in ep.itertuples()]
        found = [(abs(ep.start - r.start) < 0.02).any() if len(ep) else False for r in sh.itertuples()]
        prec = float(np.mean(hit)) if len(ep) else np.nan
        rec = float(np.mean(found)) if len(sh) else np.nan
        print(f"  шоки: найдено {len(ep)}, истинных {len(sh)}; точность (с типом) {prec:.2f}, полнота {rec:.2f}; "
              f"eta в год: агент " + ", ".join(f"{k} {v * 365:.3f}" for k, v in m.eta.items())
              + "; истина " + ", ".join(f"{k} {v * 365:.3f}" for k, v in zip(("dc", "provider", "region"), tr.eta_type)))
        # 6. тихие потери
        print(f"  тихие потери в сутки на узел: агент {m.lam_sil[alive].mean():.5f}, истина {tr.lam_sil[alive].mean():.5f}")
        rows.append(dict(seed=seed, T=T, rk=rk, rc=rc, rm_c=rm_c, rm_k=rm_k, fe=fe, fev=fev, ft=ft, prec=prec, rec=rec,
                         a=m.a_hat, cal=tab.N.sum() / (tab.e * np.exp(m._cols(tab.region.to_numpy(), tab.age.to_numpy(),
                         tab.micro_rate.to_numpy(), tab.hw.to_numpy(), tab.device.to_numpy())[:, m.keep] @ m.beta)).sum(),
                         sil=m.lam_sil[alive].mean() / tr.lam_sil[alive].mean()))
    return rows


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    import pandas as pd
    R = pd.DataFrame([r for s in range(1, n + 1) for r in one(s)])
    print("\n=== Сводка по зёрнам")
    for T, g in R.groupby("T"):
        print(f"t = {T}: калибровка {g.cal.min():.4f}..{g.cal.max():.4f}; ранговая корр. {g.rc.mean():.2f} -> {g.rk.mean():.2f}; "
              f"ln-ошибка опытных {g.rm_c.mean():.2f} -> {g.rm_k.mean():.2f}; потеря фрагмента агент/истина "
              f"{(g.fe / g.ft).min():.2f}..{(g.fe / g.ft).max():.2f}; шоки: точность {g.prec.mean():.2f}, полнота {g.rec.mean():.2f}; "
              f"тихие агент/истина {g.sil.mean():.2f}")
    check(bool((abs(R.cal - 1) < 1e-6).all()), "калибровка GLM: сумма фактических случаев = сумме ожидаемых")
    late = R[R["T"] >= 365]
    check(bool((late.rk > late.rc).all()), "кредибилити повышает ранговую корреляцию с истинным риском (t >= 365)")
    check(bool((late.rm_k < late.rm_c).all()), "кредибилити уменьшает разброс ошибки у опытных узлов (t >= 365)")
    check(bool(((late.fe / late.ft).between(0.8, 1.25)).all()), "потеря фрагмента: агент / истина в пределах 0,8–1,25 (t >= 365)")
    lt = late[late["T"] == max(TS)]
    check(bool(lt.prec.mean() >= 0.8 and lt.rec.mean() >= 0.7), "шоки к 730 суткам: средние точность >= 0,8 и полнота >= 0,7")
    check(bool(late.sil.between(0.5, 2.0).all()), "тихие потери: агент / истина в пределах 0,5–2")
    print("\nИТОГ:", "все проверки пройдены" if not FAIL else f"не пройдено {len(FAIL)}: " + "; ".join(FAIL))
