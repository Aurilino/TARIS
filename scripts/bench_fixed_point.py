"""Микробенчмарк векторной float- и scalar fixed-point денежной части тарифа."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from taris_sim.fixedpoint import SCALE, encode, financial_terms_q, gross_premium_q
from taris_sim.tariff import PriceBasis, gross_premium, node_financial_terms


def fixed(pure, concentration, suspicion, volume, trust, *, theta, eta, zeta,
          g, reward0, psi):
    out_gross = np.empty(len(pure), dtype=np.int64)
    eligible = np.empty(len(pure), dtype=bool)
    collateral = np.empty(len(pure), dtype=np.int64)
    tq, eq, zq = encode(theta), encode(eta), encode(zeta)
    gq, rq, psq = encode(g), encode(reward0), encode(psi)
    for i, (p, c, s, v, tr) in enumerate(zip(pure, concentration, suspicion, volume, trust)):
        gp = gross_premium_q(encode(p), encode(c), encode(s), tq, eq, zq)
        _, ok, kq = financial_terms_q(gp, encode(v), encode(tr), gq, rq, psq)
        out_gross[i], eligible[i], collateral[i] = gp, ok, kq
    return out_gross, eligible, collateral


def main():
    rng = np.random.default_rng(5302)
    print("Узлов | float, мс | fixed, мс | замедление", flush=True)
    for n in (1_000, 10_000, 50_000):
        pure = rng.lognormal(-3.0, 2.0, n)
        concentration, suspicion, trust = rng.random((3, n))
        volume = rng.uniform(0.0, 250.0, n)
        settings = dict(theta=0.2, eta=2.0, zeta=1.0, g=20.0, reward0=0.3, psi=1.0)
        basis = PriceBasis(g=settings["g"], u=1.0)

        def floating():
            gp = gross_premium(pure, concentration, suspicion, theta=settings["theta"],
                               eta=settings["eta"], zeta=settings["zeta"])["gross_premium"]
            return node_financial_terms(gp, volume, trust, basis,
                                        reward0=settings["reward0"], psi=settings["psi"])

        def fixed_call():
            return fixed(pure, concentration, suspicion, volume, trust, **settings)

        floating(); fixed_call()  # разогрев импортов и аллокатора
        times = []
        for fn in (floating, fixed_call):
            samples = []
            for _ in range(3):
                t0 = time.perf_counter(); fn(); samples.append(time.perf_counter() - t0)
            times.append(float(np.median(samples)))
        print(f"{n:>6} | {times[0]*1000:>9.2f} | {times[1]*1000:>9.2f} | {times[1]/times[0]:>8.1f}x",
              flush=True)


if __name__ == "__main__":
    main()
