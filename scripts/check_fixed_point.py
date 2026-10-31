"""Сверка тарифа с целочисленной фиксированной точкой (подшаг 5.3)."""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from taris_sim.fixedpoint import SCALE, decode, encode, financial_terms_q, gross_premium_q
from taris_sim.tariff import gross_premium, node_financial_terms, PriceBasis


def main():
    rng = np.random.default_rng(5203)
    n = 10_000
    pure = rng.lognormal(mean=-3.0, sigma=2.0, size=n)
    concentration = rng.random(n)
    suspicion = rng.random(n)
    theta, eta, zeta = 0.2, 2.0, 1.0
    floating = gross_premium(pure, concentration, suspicion, theta=theta, eta=eta, zeta=zeta)
    fixed = np.array([decode(gross_premium_q(encode(p), encode(c), encode(s),
        encode(theta), encode(eta), encode(zeta))) for p, c, s in zip(pure, concentration, suspicion)])
    error = np.abs(fixed - floating["gross_premium"])
    max_error = float(error.max())
    max_rel_error = float(np.max(error / np.maximum(np.abs(floating["gross_premium"]), 1e-12)))
    ties = 0
    reward0 = 0.30
    elig_float = floating["gross_premium"] < reward0
    elig_fixed = np.array([gross_premium_q(encode(p), encode(c), encode(s),
        encode(theta), encode(eta), encode(zeta)) < encode(reward0)
        for p, c, s in zip(pure, concentration, suspicion)])
    ties = int(np.count_nonzero(elig_float != elig_fixed))
    vol, trust, g, psi = 125.0, 0.8, 20.0, 1.0
    f_terms = financial_terms_q(encode(0.25), encode(vol), encode(trust), encode(g),
                                encode(reward0), encode(psi))
    fl_terms = node_financial_terms(np.array([0.25]), np.array([vol]), np.array([trust]),
                                    PriceBasis(g=g, u=1.0), reward0=reward0, psi=psi)
    coll_err = abs(decode(f_terms[2]) - float(fl_terms["collateral"][0]))
    tie_ineligible = not financial_terms_q(encode(0.3), 0, SCALE, 0, encode(0.3), 0)[1]
    below_cutoff_eligible = financial_terms_q(encode(0.3), 0, SCALE, 0, encode(0.300000001), 0)[1]
    checks = [
        (max_error <= 1e-7, f"max абсолютная ошибка тарифа {max_error:.3g} <= 1e-7 (относительная до {max_rel_error:.3g} у малых тарифов)"),
        (ties == 0, f"совпадение допуска на случайной выборке: расхождений {ties}"),
        (f_terms[0] == encode(float(fl_terms["reward_per_volume"][0])), "совпадение вознаграждения"),
        (coll_err <= 5e-8, f"ошибка залога {coll_err:.3g} <= 5e-8"),
        (tie_ineligible and below_cutoff_eligible, "порог допуска строго: тариф равный reward0 не допускается"),
        (encode("0.0000000005") == 1, "правило округления половины вверх"),
    ]
    for passed, label in checks:
        print(("ок  " if passed else "НЕТ ") + label)
    print(f"масштаб фиксированной точки: 10^{len(str(SCALE)) - 1}")
    if not all(x[0] for x in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
