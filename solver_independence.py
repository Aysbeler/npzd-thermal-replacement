"""Solver-independence check for the reference coexistence state.

The attractor at the reference parameters is recomputed with five integrators of different type,
two explicit (RK45, DOP853), two implicit (Radau, BDF) and one automatically switching (LSODA), all
at rtol 1e-10 and atol 1e-12. Agreement across schemes of different stiffness handling is evidence
that the reported equilibrium is a property of the vector field rather than of any one integrator.
"""
import warnings
import numpy as np
from scipy.integrate import solve_ivp

import model

warnings.filterwarnings('ignore')

METHODS = ['RK45', 'DOP853', 'Radau', 'BDF', 'LSODA']
Y0 = [1.0, 0.8, 0.8, 0.4, 0.4]
TEND = 2.0e4


def run(T=19.0, I0=0.5, rtol=1e-10, atol=1e-12):
    f = model.make_rhs(model.make_p(T=T, I0=I0))
    out = {}
    for m in METHODS:
        sol = solve_ivp(f, [0.0, TEND], Y0, method=m, rtol=rtol, atol=atol)
        out[m] = sol.y[:, -1]
    return out


if __name__ == '__main__':
    res = run()
    print("solver independence at the reference state (T=19 C, iota_0=0.5)")
    for m, y in res.items():
        print(f"  {m:7s} {np.round(y, 10)}")
    A = np.array(list(res.values()))
    print(f"\n  maximum deviation across the five schemes: {np.max(np.abs(A - A.mean(axis=0))):.2e}")
