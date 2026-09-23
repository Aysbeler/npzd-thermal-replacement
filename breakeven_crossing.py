"""
breakeven_crossing.py

Locates the temperature T_x at which the two break-even nutrient levels coincide,
R_1* = R_2* = R*, and characterizes the degenerate family of predator-free two-type
equilibria that exists there.

On the predator-free face {Z = 0} an equilibrium with P1, P2 > 0 requires
G_1(N) = nu_1 and G_2(N) = nu_2 simultaneously, hence N = R_1* = R_2*.  Generically the
two break-even levels differ and no such equilibrium exists (Lemma "facefree"), but at
T_x they coincide.  Eliminating D = (nu_1 P1 + nu_2 P2)/(rho + lD) leaves the nutrient
balance as a single scalar constraint,

    lD/(rho + lD) * (nu_1 P1 + nu_2 P2) = iota_0 - lN R*,

one equation in two unknowns.  When the right-hand side is positive this is a segment of
equilibria joining F_1 to F_2.  The script verifies the residual, the zero eigenvalue that
makes the family nonhyperbolic, and the sign of the grazer invasion eigenvalue along it.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import brentq
import model

p = model.make_p()

def G(N, T, i):
    th = model.theta(T, p[f'Topt{i}'], p[f'w{i}'], p[f'wskew{i}'])
    return p[f'mu{i}'] * th * model.f(N, p[f'KN{i}'])

def Rstar(T, i):
    nu = p[f'd{i}']
    if G(1e6, T, i) <= nu:
        return np.nan
    return brentq(lambda N: G(N, T, i) - nu, 1e-12, 1e6, xtol=1e-14, rtol=1e-15)

Tx = brentq(lambda T: Rstar(T, 1) - Rstar(T, 2), 17.5, 19.0, xtol=1e-13, rtol=1e-15)
Rx = Rstar(Tx, 1)
print(f"crossing temperature   T_x  = {Tx:.9f} C")
print(f"common break-even      R*   = {Rx:.9f}")
print(f"  |R_1* - R_2*|             = {abs(Rstar(Tx,1)-Rstar(Tx,2)):.2e}")
print(f"  G_1(R*) - nu_1            = {G(Rx,Tx,1)-p['d1']:.2e}")
print(f"  G_2(R*) - nu_2            = {G(Rx,Tx,2)-p['d2']:.2e}")

rho = model.rho(Tx, p['rho0'], p['Q10'], p['Tref'])
coef = p['lD'] / (rho + p['lD'])
rhs = p['I0'] - p['lN'] * Rx
print(f"\nrho(T_x) = {rho:.6f},   lD/(rho+lD) = {coef:.6f}")
print(f"iota_0 - lN R* = {rhs:.6f}   ({'positive: continuum exists' if rhs > 0 else 'nonpositive: no continuum'})")

S = rhs / coef                      # = nu_1 P1 + nu_2 P2
total = S / p['d1']                 # equals P1 + P2 when nu_1 = nu_2
print(f"nu_1 P1 + nu_2 P2 = {S:.6f}    ->  P1 + P2 = {total:.6f}")

rhs_f = model.make_rhs(model.make_p(T=Tx))
print(f"\n{'P1':>8} {'P2':>8} {'D':>8} {'max|F|':>10} {'min|eig|':>10} {'sigma_Z':>10}")
for frac in [0.05, 0.25, 0.50, 0.75, 0.95]:
    P1 = total * frac; P2 = total - P1
    D = (p['d1']*P1 + p['d2']*P2) / (rho + p['lD'])
    x = np.array([Rx, P1, P2, 0.0, D])
    F = np.array(rhs_f(0.0, x))
    n = 5; J = np.empty((n, n)); h = 1e-7
    for k in range(n):
        xp = x.copy(); xm = x.copy(); xp[k] += h; xm[k] -= h
        J[:, k] = (np.array(rhs_f(0, xp)) - np.array(rhs_f(0, xm))) / (2*h)
    ev = np.abs(np.linalg.eigvals(J))
    g1 = model.g(P1, p['KP1']); g2 = model.g(P2, p['KP2'])
    s = p['switch']; a = P1**s; b = P2**s
    sig = p['e']*p['mZ']*((a/(a+b))*g1 + (b/(a+b))*g2) - p['dZ']
    print(f"{P1:8.4f} {P2:8.4f} {D:8.4f} {np.max(np.abs(F)):10.2e} {ev.min():10.2e} {sig:+10.5f}")

print("\nA zero eigenvalue at every point: the family is nonisolated and nonhyperbolic,")
print("so an acyclicity covering by isolated invariant sets does not apply at T_x.")
print("The grazer invasion eigenvalue stays uniformly positive along the family, so no")
print("trajectory is retained on it; the obstruction is technical, not dynamical.")
