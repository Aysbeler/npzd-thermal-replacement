"""
q10_plane.py

Scans the coexistence window over independent temperature dependences for grazing and for
zooplankton mortality.

Applying a common Q10 factor to both rates leaves the window unchanged, but that invariance is
partly structural: the grazer balance e*gamma*g(P) = nu_Z that fixes the single-type equilibrium
contains gamma and nu_Z as a ratio, so a common factor cancels.  Scaling the two rates by
independent factors,

    gamma(T)  = gamma  * Q10g^((T - Tref)/10),
    nu_Z(T)   = nu_Z   * Q10m^((T - Tref)/10),

removes the cancellation and tests whether the coexistence structure depends on the two
dependences being tied together.

Window edges are the roots of the invasion fitness at the single-type equilibria, the same
convention used elsewhere.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import brentq
import model

p = model.make_p()
TREF = p['Tref']

def rates(T, Qg, Qm):
    """Temperature-scaled grazing rate and zooplankton mortality."""
    return (p['mZ']*Qg**((T-TREF)/10.0),
            p['dZ']*Qm**((T-TREF)/10.0))

def G(N, T, i):
    th = model.theta(T, p[f'Topt{i}'], p[f'w{i}'], p[f'wskew{i}'])
    return p[f'mu{i}']*th*model.f(N, p[f'KN{i}'])

def single_type_N(T, i, Qg, Qm):
    """Nutrient coordinate of the grazer-bearing single-type equilibrium E_i, or None."""
    gam, nuZ = rates(T, Qg, Qm)
    if p['e']*gam <= nuZ:
        return None                       # grazer cannot persist
    P = p[f'KP{i}']*nuZ/(p['e']*gam - nuZ)
    rho = model.rho(T, p['rho0'], p['Q10'], p['Tref'])
    # nutrient balance with D eliminated:  iota0 - lN N - (lD/(rho+lD)) P G_i(N) = 0
    c = p['lD']/(rho + p['lD'])
    f = lambda N: p['I0'] - p['lN']*N - c*P*G(N, T, i)
    if f(1e-12) <= 0:
        return None
    hi = p['I0']/p['lN']
    if f(hi) > 0:
        return None
    N = brentq(f, 1e-12, hi, xtol=1e-13)
    return N if G(N, T, i) > p[f'd{i}'] else None   # E_i admissible only if Z* > 0

def invasion(T, resident, Qg, Qm):
    """Fitness of the missing type at E_resident."""
    N = single_type_N(T, resident, Qg, Qm)
    if N is None:
        return np.nan
    j = 2 if resident == 1 else 1
    return G(N, T, j) - p[f'd{j}']

def window(Qg, Qm, Ts=np.linspace(12, 26, 400)):
    v2 = np.array([invasion(T, 1, Qg, Qm) for T in Ts])   # P2 invades E1
    v1 = np.array([invasion(T, 2, Qg, Qm) for T in Ts])   # P1 invades E2
    def roots(fun, v):
        out = []
        for k in range(len(Ts)-1):
            if np.isfinite(v[k]) and np.isfinite(v[k+1]) and v[k]*v[k+1] < 0:
                out.append(brentq(lambda T: fun(T), Ts[k], Ts[k+1], xtol=1e-5))
        return out
    r2 = roots(lambda T: invasion(T, 1, Qg, Qm), v2)
    r1 = roots(lambda T: invasion(T, 2, Qg, Qm), v1)
    if not r2 or not r1:
        return None
    T1, T2 = min(r2), max(r1)
    return (T1, T2) if T2 > T1 else None

QS = [1.5, 2.0, 2.5, 3.0]
print("window width (deg C); rows = Q10 grazing, columns = Q10 mortality")
print(f"{'Qg\\Qm':>7}" + "".join(f"{q:>9.1f}" for q in QS))
widths = []
for Qg in QS:
    row = f"{Qg:>7.1f}"
    for Qm in QS:
        w = window(Qg, Qm)
        if w is None:
            row += f"{'closed':>9}"
        else:
            row += f"{w[1]-w[0]:9.2f}"
            widths.append((w[1]-w[0], Qg, Qm))
    print(row)

widths.sort()
print(f"\nwindow present in {len(widths)}/{len(QS)**2} cells")
print(f"narrowest {widths[0][0]:.2f} at (Qg,Qm)=({widths[0][1]},{widths[0][2]})")
print(f"widest    {widths[-1][0]:.2f} at (Qg,Qm)=({widths[-1][1]},{widths[-1][2]})")
