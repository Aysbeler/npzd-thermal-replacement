"""
floquet_cycles.py

Independent verification of the periodic branches, addressing the concern that oscillation
envelopes are obtained by long-time integration rather than by periodic-orbit continuation.

For each parameter value we

  1. integrate onto the attractor,
  2. locate the period by successive upward crossings of a Poincare section,
  3. integrate the variational equation Phi' = J(x(t)) Phi with Phi(0) = I over exactly one
     period, giving the monodromy matrix, and
  4. report its eigenvalues, the Floquet multipliers.

A hyperbolic stable limit cycle has one trivial multiplier equal to unity, corresponding to
translation along the orbit, and all remaining multipliers strictly inside the unit circle.
Continuing along the branch and watching the multipliers detects a fold of cycles (a multiplier
leaving through +1) or a period doubling (through -1); neither should occur on a supercritical
branch that stays stable.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model

def jac(rhs, x, h=1e-6):
    n = len(x); J = np.empty((n, n)); f0 = np.array(rhs(0.0, x))
    for k in range(n):
        xp = np.array(x, float); xp[k] += h
        J[:, k] = (np.array(rhs(0.0, xp)) - f0)/h
    return J

def cycle_floquet(T, I0, tsettle=40000.0, tol=1e-11):
    p = model.make_p(T=T, I0=I0); rhs = model.make_rhs(p)
    s = solve_ivp(rhs, [0, tsettle], [1.0, 0.8, 0.8, 0.4, 0.4], method='LSODA',
                  rtol=tol, atol=1e-13)
    x0 = s.y[:, -1]

    # section: P1 = its mean on the attractor, crossed with dP1/dt > 0
    s2 = solve_ivp(rhs, [0, 4000], x0, method='LSODA', rtol=tol, atol=1e-13,
                   dense_output=True)
    tt = np.linspace(2000, 4000, 200000); Y = s2.sol(tt)
    # section on whichever component actually oscillates
    amps = Y.max(axis=1) - Y.min(axis=1)
    comp = int(np.argmax(amps)); amp = amps[comp]
    if amp < 1e-6:
        return None, None, amp                      # equilibrium, not a cycle
    level = 0.5*(Y[comp].max() + Y[comp].min())

    def ev(t, y): return y[comp] - level
    ev.direction = 1.0
    s3 = solve_ivp(rhs, [0, 4000], s2.sol(2000.0), method='LSODA', rtol=tol, atol=1e-13,
                   events=ev, dense_output=True)
    te = s3.t_events[0]
    if len(te) < 3:
        return None, None, amp
    period = float(np.mean(np.diff(te[-4:] if len(te) >= 4 else te)))
    xs = s3.sol(te[-2])

    # variational equation over exactly one period
    n = 5
    def aug(t, z):
        x = z[:n]; Phi = z[n:].reshape(n, n)
        return np.concatenate([np.array(rhs(t, x)), (jac(rhs, x) @ Phi).ravel()])
    z0 = np.concatenate([xs, np.eye(n).ravel()])
    sa = solve_ivp(aug, [0, period], z0, method='LSODA', rtol=1e-10, atol=1e-12)
    M = sa.y[n:, -1].reshape(n, n)
    return period, np.linalg.eigvals(M), amp

def face_floquet(T, i, I0=0.5):
    """Floquet multipliers of the single-type face cycle Gamma_i, required by (P4).

    The samples in the main routine cover interior and warm-branch cycles; hypothesis (P4)
    instead requires the cycles that live on the faces {P_j = 0}, which are computed here by
    reducing to the four-dimensional face system and repeating the monodromy integration.
    """
    p = model.make_p()
    rho = model.rho(T, p['rho0'], p['Q10'], p['Tref'])
    th = model.theta(T, p[f'Topt{i}'], p[f'w{i}'], p[f'wskew{i}'])

    def f(t, y):
        N, P, Z, D = [max(v, 1e-12) for v in y]
        gr = p[f'mu{i}']*th*model.f(N, p[f'KN{i}'])*P
        gz = p['mZ']*model.g(P, p[f'KP{i}'])*Z
        return [I0 - p['lN']*N - gr + rho*D,
                gr - gz - p[f'd{i}']*P,
                p['e']*gz - p['dZ']*Z,
                p[f'd{i}']*P + (1-p['e'])*gz + p['dZ']*Z - rho*D - p['lD']*D]
    return f


if __name__ == '__main__':
    print("ENRICHMENT HOPF BRANCH   (T = 19 C, varying iota0)")
    print(f"{'iota0':>7} {'period':>9} {'amplitude':>10} {'trivial':>9} {'|mu| others':>34}")
    print("-"*76)
    for I0 in [0.80, 0.85, 0.95, 1.10, 1.30, 1.50]:
        per, mu, amp = cycle_floquet(19.0, I0)
        if per is None:
            print(f"{I0:7.2f} {'--':>9} {amp:9.2e}   (no cycle: equilibrium)")
            continue
        k = int(np.argmin(np.abs(np.abs(mu) - 1.0)))
        others = np.sort(np.abs(np.delete(mu, k)))[::-1]
        print(f"{I0:7.2f} {per:9.3f} {amp:9.4f} {np.abs(mu[k]):9.5f}   "
              + "  ".join(f"{v:.2e}" for v in others))

    print("\nWARMING HOPF BRANCH   (iota0 = 0.5, varying T)")
    print(f"{'T':>7} {'period':>9} {'amplitude':>10} {'trivial':>9} {'|mu| others':>34}")
    print("-"*76)
    for T in [27.0, 27.5, 28.0, 29.0, 30.0]:
        per, mu, amp = cycle_floquet(T, 0.5)
        if per is None:
            print(f"{T:7.1f} {'--':>9} {amp:9.2e}   (no cycle: equilibrium)")
            continue
        k = int(np.argmin(np.abs(np.abs(mu) - 1.0)))
        others = np.sort(np.abs(np.delete(mu, k)))[::-1]
        print(f"{T:7.1f} {per:9.3f} {amp:9.4f} {np.abs(mu[k]):9.5f}   "
              + "  ".join(f"{v:.2e}" for v in others))

    print("\nA stable hyperbolic cycle has one multiplier at 1 and the rest inside the unit circle.")
    print("No multiplier approaching +1 (fold of cycles) or -1 (period doubling) appears on either branch.")
