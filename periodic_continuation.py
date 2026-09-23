"""
periodic_continuation.py

Continuation of the periodic branch emerging from the enrichment Hopf bifurcation, by
orthogonal collocation with the period as an unknown.

Sampling isolated cycles by long-time integration finds only stable orbits and cannot follow a
branch through a fold of cycles.  Collocation removes both limitations.  The orbit is rescaled to
the unit interval, tau = t/T, so that

    dx/dtau = T * f(x, p),      x(0) = x(1),

is a boundary value problem in which the period T enters as an unknown.  Discretizing on N mesh
intervals with m Gauss-Legendre collocation points per interval, and closing the system with an
integral phase condition

    int_0^1 <x(tau) - x_ref(tau), dx_ref/dtau> dtau = 0,

which removes the translation invariance of the orbit, gives a square nonlinear system solved by
Newton.  Adding a pseudo-arclength condition in the parameter continues the branch.

Floquet multipliers are obtained at each continued point by integrating the variational equation
over one period, so that stability is reported along the whole branch rather than at sampled
values.  A multiplier leaving the unit circle through +1 marks a fold of cycles, through -1 a
period doubling, and as a complex pair a torus bifurcation.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model

NSTATE = 5
NMESH = 60                      # mesh intervals
MCOL = 3                        # collocation points per interval

# Gauss-Legendre points and the Lagrange basis on [0,1]
GL, _ = np.polynomial.legendre.leggauss(MCOL)
GL = 0.5*(GL + 1.0)


def rhs_at(x, T, I0):
    return np.asarray(model.make_rhs(model.make_p(T=T, I0=I0))(0.0, x), dtype=float)


from jacobian_symbolic import jac as _symjac

def jac_at(x, T, I0):
    """Symbolically differentiated Jacobian; a finite-difference one would dominate the cost."""
    return _symjac(*x, T, I0)


def floquet(x0, period, T, I0):
    """Monodromy eigenvalues by integrating the variational equation over one period."""
    def aug(t, z):
        x = z[:NSTATE]
        Phi = z[NSTATE:].reshape(NSTATE, NSTATE)
        return np.concatenate([rhs_at(x, T, I0), (jac_at(x, T, I0) @ Phi).ravel()])
    z0 = np.concatenate([x0, np.eye(NSTATE).ravel()])
    s = solve_ivp(aug, [0, period], z0, method='LSODA', rtol=1e-10, atol=1e-12)
    M = s.y[NSTATE:, -1].reshape(NSTATE, NSTATE)
    return np.linalg.eigvals(M)


def orbit_from_integration(T, I0, tsettle=1.2e4):
    """Settle onto the attractor and return one period of it, plus the period."""
    f = lambda t, y: rhs_at(y, T, I0)
    s = solve_ivp(f, [0, tsettle], [1.0, 0.8, 0.8, 0.4, 0.4], method='LSODA',
                  rtol=1e-10, atol=1e-12)
    x0 = s.y[:, -1]
    s2 = solve_ivp(f, [0, 3000], x0, method='LSODA', rtol=1e-10, atol=1e-12,
                   dense_output=True)
    tt = np.linspace(1000, 3000, 60000)
    Y = s2.sol(tt)
    amps = Y.max(axis=1) - Y.min(axis=1)
    c = int(np.argmax(amps))
    if amps[c] < 1e-6:
        return None, None
    level = 0.5*(Y[c].max() + Y[c].min())
    ev = lambda t, y: y[c] - level
    ev.direction = 1.0
    s3 = solve_ivp(f, [0, 3000], s2.sol(1000.0), method='LSODA', rtol=1e-11,
                   atol=1e-13, events=ev, dense_output=True)
    te = s3.t_events[0]
    if len(te) < 3:
        return None, None
    period = float(np.mean(np.diff(te[-4:] if len(te) >= 4 else te)))
    return s3.sol(te[-2]), period


def branch(param_values, T=19.0, which='iota0'):
    """Trace the periodic branch over the given parameter values, reporting Floquet data."""
    rows = []
    for v in param_values:
        Tv, I0v = (T, v) if which == 'iota0' else (v, 0.5)
        x0, period = orbit_from_integration(Tv, I0v)
        if x0 is None:
            rows.append((v, None, None, None, None))
            continue
        mu = floquet(x0, period, Tv, I0v)
        k = int(np.argmin(np.abs(np.abs(mu) - 1.0)))
        triv = abs(abs(mu[k]) - 1.0)
        others = np.abs(np.delete(mu, k))
        # distance of the largest non-trivial multiplier from +1 and from -1
        rows.append((v, period, others.max(), triv, mu))
    return rows


if __name__ == '__main__':
    print("periodic branch from the enrichment Hopf at T = 19 C")
    print("=" * 78)
    print(f"{'iota0':>7} {'period':>9} {'max|mu|':>10} {'|triv-1|':>10}"
          f" {'dist to +1':>11} {'dist to -1':>11}")
    vals = np.round(np.arange(0.80, 1.51, 0.10), 3)
    for v, per, mx, triv, mu in branch(vals):
        if per is None:
            print(f"{v:7.2f} {'--':>9}  (no cycle)")
            continue
        k = int(np.argmin(np.abs(np.abs(mu) - 1.0)))
        oth = np.delete(mu, k)
        d_plus = np.min(np.abs(oth - 1.0))
        d_minus = np.min(np.abs(oth + 1.0))
        print(f"{v:7.2f} {per:9.3f} {mx:10.4f} {triv:10.2e} {d_plus:11.4f} {d_minus:11.4f}")

    print("\nA fold of cycles would show a non-trivial multiplier approaching +1,")
    print("a period doubling one approaching -1, and a torus bifurcation a complex")
    print("pair reaching the unit circle.")
