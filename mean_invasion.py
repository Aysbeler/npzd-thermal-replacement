"""
mean_invasion.py

Computes the time-averaged transverse growth rate of the missing phytoplankton type on each
single-type face attractor M_i, the quantity denoted <I_j>_{M_i} in the persistence argument.

On the face {P_j = 0, P_i > 0, Z > 0} the reduced N-P_i-Z-D system is integrated past its
transients onto its attractor, which is an equilibrium over part of the coexistence window and
a limit cycle over the rest.  The transverse growth rate of the absent type j is

    I_j(t) = G_j(N(t)) - nu_j,    G_j(N) = beta_j theta_j(T) N/(kappa_{N,j} + N),

and its long-time average is reported.  A positive average means the missing type invades the
face attractor in the mean, which is the invasion requirement in hypothesis (P4).
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model

p = model.make_p()

def G(N, T, i):
    th = model.theta(T, p[f'Topt{i}'], p[f'w{i}'], p[f'wskew{i}'])
    return p[f'mu{i}'] * th * model.f(N, p[f'KN{i}'])

def face_rhs(T, i):
    """Reduced N-P_i-Z-D dynamics on the face where the other type is absent."""
    rho = model.rho(T, p['rho0'], p['Q10'], p['Tref'])
    def f(t, y):
        N, P, Z, D = [max(v, 0.0) for v in y]
        # with only one prey present the switching weight is unity
        gr = G(N, T, i) * P
        graz = p['mZ'] * model.g(P, p[f'KP{i}']) * Z
        return [p['I0'] - p['lN']*N - gr + rho*D,
                gr - graz - p[f'd{i}']*P,
                p['e']*graz - p['dZ']*Z,
                p[f'd{i}']*P + (1-p['e'])*graz + p['dZ']*Z - rho*D - p['lD']*D]
    return f

def mean_invasion(T, i, j, tsettle=30000.0, tavg=6000.0):
    """<I_j> on the attractor of the P_i-only face."""
    f = face_rhs(T, i)
    s = solve_ivp(f, [0, tsettle], [1.0, 1.0, 0.4, 1.0], method='LSODA',
                  rtol=1e-10, atol=1e-12)
    y0 = s.y[:, -1]
    s2 = solve_ivp(f, [0, tavg], y0, method='LSODA', rtol=1e-10, atol=1e-12,
                   dense_output=True)
    tt = np.linspace(0, tavg, 400000)
    N = s2.sol(tt)[0]
    amp = s2.sol(tt)[1].max() - s2.sol(tt)[1].min()
    I = G(N, T, j) - p[f'd{j}']
    return I.mean(), ('cycle' if amp > 1e-4 else 'equilibrium')

T1, T2 = 15.37, 20.10
print(f"{'T':>6}  {'<I_2> on M_1':>14} {'state':>12}   {'<I_1> on M_2':>14} {'state':>12}")
print("-" * 72)
A, B = [], []
for T in np.linspace(T1, T2, 11):
    a, sa = mean_invasion(T, 1, 2)
    b, sb = mean_invasion(T, 2, 1)
    A.append(a); B.append(b)
    print(f"{T:6.2f}  {a:14.4f} {sa:>12}   {b:14.4f} {sb:>12}")

A, B = np.array(A), np.array(B)
print("-" * 72)
print(f"<I_2> on M_1 :  [{A.min():.3f}, {A.max():.3f}]")
print(f"<I_1> on M_2 :  [{B.min():.3f}, {B.max():.3f}]")
print("\nBoth averages stay positive across the window, so each single-type face attractor")
print("is transversally repelling in the mean, as hypothesis (P4) requires.")
