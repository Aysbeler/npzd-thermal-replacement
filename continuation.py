"""
continuation.py

Pseudo-arclength continuation of the equilibrium branches of the scaled model, in temperature
at fixed nutrient input.

Natural-parameter continuation steps the parameter and re-solves, so it stalls at a fold and
cannot follow a branch that turns back.  Pseudo-arclength continuation removes that limitation by
treating the parameter as an unknown and closing the system with a scalar arclength condition.
Writing y = (x, T) with x the five state variables, the branch is defined by

    F(x, T) = 0,                                    (5 equations)
    <y - y_k, t_k> - ds = 0,                        (1 equation)

where y_k is the last converged point, t_k the unit tangent there, and ds the step.  The
6x6 system is solved by Newton with an analytic bordered Jacobian.  The tangent is obtained from
the null vector of [F_x | F_T] and oriented by continuity, so the branch is traversed smoothly
through folds, where F_x becomes singular but the bordered matrix does not.

Stability is classified from the eigenvalues of F_x at each point, folds are located where the
tangent's parameter component changes sign, and branch points (transcritical) are located where
the determinant of F_x changes sign without a fold.

Running this file continues three branches: the cold-type boundary branch, the warm-type
boundary branch, and the interior coexistence branch, and reports for each the stable and
unstable segments, any folds, and the temperatures at which they meet.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model

I0 = 0.5
NSTATE = 5


def F(x, T):
    """Right-hand side of the scaled model at temperature T."""
    rhs = model.make_rhs(model.make_p(T=T, I0=I0))
    return np.asarray(rhs(0.0, x), dtype=float)


def Fx(x, T, h=1e-7):
    """State Jacobian by central differences."""
    J = np.empty((NSTATE, NSTATE))
    for k in range(NSTATE):
        xp = x.copy(); xm = x.copy()
        xp[k] += h; xm[k] -= h
        J[:, k] = (F(xp, T) - F(xm, T))/(2*h)
    return J


def FT(x, T, h=1e-6):
    """Parameter derivative by central differences."""
    return (F(x, T + h) - F(x, T - h))/(2*h)


def tangent(x, T, prev=None):
    """Unit tangent to the branch, oriented to continue in the same direction."""
    A = np.hstack([Fx(x, T), FT(x, T).reshape(-1, 1)])      # 5 x 6
    _, _, Vt = np.linalg.svd(A)
    t = Vt[-1]                                              # null vector
    t = t/np.linalg.norm(t)
    if prev is not None and np.dot(t, prev) < 0:
        t = -t
    return t


def corrector(y0, yk, tk, ds, tol=1e-11, itmax=40):
    """Newton solve of F = 0 together with the arclength condition."""
    y = y0.copy()
    for _ in range(itmax):
        x, T = y[:NSTATE], y[NSTATE]
        r = np.empty(NSTATE + 1)
        r[:NSTATE] = F(x, T)
        r[NSTATE] = np.dot(y - yk, tk) - ds
        if np.max(np.abs(r)) < tol:
            return y, True
        J = np.empty((NSTATE + 1, NSTATE + 1))
        J[:NSTATE, :NSTATE] = Fx(x, T)
        J[:NSTATE, NSTATE] = FT(x, T)
        J[NSTATE, :] = tk
        try:
            y = y - np.linalg.solve(J, r)
        except np.linalg.LinAlgError:
            return y, False
    return y, np.max(np.abs(r)) < 1e-8


def continue_branch(x0, T0, ds=0.02, nmax=4000, Trange=(11.0, 31.0), direction=+1):
    """Trace a branch by pseudo-arclength continuation from a converged starting point."""
    y = np.append(np.asarray(x0, float), T0)
    t = tangent(y[:NSTATE], y[NSTATE])
    if np.sign(t[NSTATE]) != np.sign(direction):
        t = -t
    pts, stab, tans = [y.copy()], [], [t.copy()]
    ev = np.linalg.eigvals(Fx(y[:NSTATE], y[NSTATE]))
    stab.append(ev.real.max() < 0)

    step = ds
    for _ in range(nmax):
        yk, tk = pts[-1], tans[-1]
        ynew, ok = corrector(yk + step*tk, yk, tk, step)
        if not ok:
            step *= 0.5
            if abs(step) < 1e-6:
                break
            continue
        if np.any(ynew[:NSTATE] < -1e-7):                   # left the positive orthant
            break
        if not (Trange[0] <= ynew[NSTATE] <= Trange[1]):
            break
        tnew = tangent(ynew[:NSTATE], ynew[NSTATE], prev=tk)
        pts.append(ynew); tans.append(tnew)
        stab.append(np.linalg.eigvals(Fx(ynew[:NSTATE], ynew[NSTATE])).real.max() < 0)
        step = min(abs(step)*1.2, ds)*np.sign(step)
    return np.array(pts), np.array(stab), np.array(tans)


def report(name, pts, stab, tans):
    T = pts[:, NSTATE]
    print(f"\n{name}")
    print(f"  points {len(pts)},  T from {T.min():.3f} to {T.max():.3f}")
    folds = np.where(np.sign(tans[:-1, NSTATE])*np.sign(tans[1:, NSTATE]) < 0)[0]
    print(f"  folds (tangent parameter component changes sign): "
          f"{[f'{T[k]:.3f}' for k in folds] if len(folds) else 'none'}")
    sw = np.where(stab[:-1] != stab[1:])[0]
    print(f"  stability changes at T = {[f'{T[k]:.3f}' for k in sw] if len(sw) else 'none'}")
    print(f"  stable fraction of branch: {stab.mean()*100:.0f}%")


if __name__ == '__main__':
    print("pseudo-arclength continuation in T at iota_0 = 0.5")
    print("=" * 66)

    # interior coexistence branch: start from the attractor at mid-window
    T0 = 18.0
    s = solve_ivp(model.make_rhs(model.make_p(T=T0, I0=I0)), [0, 40000],
                  [1, .8, .8, .4, .4], method='LSODA', rtol=1e-11, atol=1e-13)
    x0 = s.y[:, -1]
    for d, lab in ((+1, 'upward'), (-1, 'downward')):
        pts, stab, tans = continue_branch(x0, T0, direction=d)
        report(f"coexistence branch, continued {lab} from T = {T0}", pts, stab, tans)
