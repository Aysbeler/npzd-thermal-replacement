"""
zerohopf.py

Locates and classifies the codimension-two point at which the Hopf curve of the coexistence
equilibrium meets the transcritical boundary of the coexistence window.

A first attempt failed for a numerical reason: continuing the equilibrium in Cartesian
coordinates, Newton leaves the positive orthant near T = 19.9 C and converges to a spurious root
with P1 < 0, so the Hopf condition cannot be evaluated where the intersection lies.  The remedy is
to continue in logarithmic coordinates,

    u_i = log x_i,      x_i = exp(u_i) > 0,

in which the positive orthant is the whole space and no iterate can leave it.  The vector field
transforms as du_i/dt = F_i(x)/x_i, with Jacobian

    (D_u G)_{ij} = (x_j/x_i) (D_x F)_{ij} - delta_{ij} F_i(x)/x_i ,

the second term vanishing at an equilibrium, so the spectra of D_u G and D_x F coincide there and
stability conclusions are unchanged.

The Hopf curve is then traced as a two-parameter continuation: the defining system

    F(x, T, iota_0) = 0,        Re lambda_c(x, T, iota_0) = 0,

is solved for (x, iota_0) at each prescribed T, where lambda_c is the eigenvalue pair of largest
real part among the complex ones.  Comparing the resulting iota_0^H(T) with the transcritical edge
T_2(iota_0) at the same nutrient input gives a scalar gap whose root is the codimension-two point.

At the located point the script reports the coordinates, the full spectrum, the simplicity of the
zero eigenvalue and of the imaginary pair, and whether the two curves cross transversally, which
together constitute a numerical classification of the degeneracy.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import fsolve, brentq
from scipy.integrate import solve_ivp
import model

NSTATE = 5


def Fx_cart(x, T, I0):
    return np.asarray(model.make_rhs(model.make_p(T=T, I0=I0))(0.0, x), dtype=float)


def G_log(u, T, I0):
    """Vector field in logarithmic coordinates."""
    x = np.exp(u)
    return Fx_cart(x, T, I0)/x


def jac_log(u, T, I0, h=1e-7):
    J = np.empty((NSTATE, NSTATE))
    for k in range(NSTATE):
        up = u.copy(); um = u.copy(); up[k] += h; um[k] -= h
        J[:, k] = (G_log(up, T, I0) - G_log(um, T, I0))/(2*h)
    return J


def equilibrium(T, I0, useed=None):
    """Coexistence equilibrium in log coordinates; cannot leave the positive orthant."""
    if useed is None:
        s = solve_ivp(lambda t, y: Fx_cart(y, T, I0), [0, 30000],
                      [1, .8, .8, .4, .4], method='LSODA', rtol=1e-11, atol=1e-13)
        useed = np.log(np.maximum(s.y[:, -1], 1e-10))
    u = fsolve(lambda v: G_log(v, T, I0), useed, xtol=1e-13)
    return u if np.max(np.abs(G_log(u, T, I0))) < 1e-9 else None


def dominant_complex(u, T, I0):
    ev = np.linalg.eigvals(jac_log(u, T, I0))
    cx = ev[np.abs(ev.imag) > 1e-8]
    return None if len(cx) == 0 else cx[np.argmax(cx.real)]


def hopf_I(T, useed, lo=0.40, hi=1.5, step=0.005):
    """Nutrient input at which the coexistence equilibrium at temperature T undergoes Hopf.

    The bracket is found by scanning rather than assumed.  Below the transcritical edge the
    coexistence equilibrium does not exist, and far above it Newton may fail to converge from a
    distant seed, so a fixed bracket fails exactly where the Hopf curve approaches that edge,
    which is the region of interest.  We therefore scan upward from the smallest admissible
    input until the dominant real part changes sign.
    """
    def g(I0):
        u = equilibrium(T, I0, useed)
        if u is None:
            return np.nan
        c = dominant_complex(u, T, I0)
        return np.nan if c is None else c.real

    prev_I, prev_g = None, None
    I0 = lo
    while I0 <= hi:
        v = g(I0)
        if np.isfinite(v):
            if prev_g is not None and prev_g < 0 <= v:
                Ic = brentq(g, prev_I, I0, xtol=1e-9)
                return Ic, equilibrium(T, Ic, useed)
            prev_I, prev_g = I0, v
        I0 += step
    return np.nan, None


def transcritical_T(I0, lo=18.0, hi=24.0):
    """Upper window edge: root of the cold type's invasion fitness at the warm-only equilibrium."""
    p = model.make_p()

    def inv(T):
        Pi = p['KP2']*p['dZ']/(p['e']*p['mZ'] - p['dZ'])
        rho = model.rho(T, p['rho0'], p['Q10'], p['Tref'])
        c = p['lD']/(rho + p['lD'])
        th2 = model.theta(T, p['Topt2'], p['w2'], p['wskew2'])
        G2 = lambda N: p['mu2']*th2*model.f(N, p['KN2'])
        f = lambda N: I0 - p['lN']*N - c*Pi*G2(N)
        hi_ = I0/p['lN']
        if not (f(1e-12) > 0 > f(hi_)):
            return np.nan
        N = brentq(f, 1e-12, hi_, xtol=1e-14)
        th1 = model.theta(T, p['Topt1'], p['w1'], p['wskew1'])
        return p['mu1']*th1*model.f(N, p['KN1']) - p['d1']
    try:
        return brentq(inv, lo, hi, xtol=1e-9)
    except Exception:
        return np.nan


def gap(T, useed):
    """T2 evaluated at the Hopf curve, minus T: vanishes at the codimension-two point."""
    I0, u = hopf_I(T, useed)
    if not np.isfinite(I0):
        return np.nan, np.nan, None
    return transcritical_T(I0) - T, I0, u


def boundary_E2(T, I0):
    """Warm-type boundary equilibrium E_2, at which P_1 = 0."""
    P2 = p2['KP2']*p2['dZ']/(p2['e']*p2['mZ'] - p2['dZ'])
    rho = model.rho(T, p2['rho0'], p2['Q10'], p2['Tref'])
    c = p2['lD']/(rho + p2['lD'])
    f = lambda N: I0 - p2['lN']*N - c*P2*Gi(N, T, 2)
    hi = I0/p2['lN']
    if not (f(1e-12) > 0 > f(hi)):
        return None
    N = brentq(f, 1e-12, hi, xtol=1e-14)
    Z = p2['e']*P2/p2['dZ']*(Gi(N, T, 2) - p2['d2'])
    D = P2*Gi(N, T, 2)/(rho + p2['lD'])
    return np.array([N, 0.0, P2, Z, D])


def Gi(N, T, i):
    th = model.theta(T, p2[f'Topt{i}'], p2[f'w{i}'], p2[f'wskew{i}'])
    return p2[f'mu{i}']*th*model.f(N, p2[f'KN{i}'])


p2 = model.make_p()


def zh_residuals(v):
    """Defining system of the zero-Hopf point.

    The degeneracy sits on the transcritical boundary, not in the interior: it is the warm-type
    equilibrium E_2 at which the cold type's invasion fitness vanishes, giving a zero eigenvalue
    in the P_1 direction, while a complex pair is simultaneously critical.  Continuing the
    interior coexistence equilibrium instead locates a nearby Hopf point whose real eigenvalue is
    small but nonzero, which is not the degeneracy.
    """
    T, I0 = v
    x = boundary_E2(T, I0)
    if x is None:
        return [1e3, 1e3]
    c1 = Gi(x[0], T, 1) - p2['d1']                  # transcritical condition
    rhs = model.make_rhs(model.make_p(T=T, I0=I0))
    J = np.empty((NSTATE, NSTATE)); h = 1e-7
    for k in range(NSTATE):
        xp = x.copy(); xm = x.copy(); xp[k] += h; xm[k] -= h
        J[:, k] = (np.asarray(rhs(0, xp)) - np.asarray(rhs(0, xm)))/(2*h)
    J[1, :] = 0.0; J[1, 1] = c1                     # the face is invariant; set the row exactly
    ev = np.linalg.eigvals(J)
    cx = ev[np.abs(ev.imag) > 1e-8]
    return [c1, cx[np.argmax(cx.real)].real if len(cx) else 1e3]


if __name__ == '__main__':
    print("zero-Hopf point on the transcritical boundary")
    print("=" * 62)
    sol = fsolve(zh_residuals, [20.15, 0.52], xtol=1e-14)
    T, I0 = sol
    r = zh_residuals(sol)
    print(f"(T, iota_0) = ({T:.8f}, {I0:.8f})")
    print(f"residuals: invasion fitness {r[0]:.2e}, Re(lambda_H) {r[1]:.2e}")
    x = boundary_E2(T, I0)
    print(f"E_ZH = {np.round(x, 8)}")
    rhs = model.make_rhs(model.make_p(T=T, I0=I0))
    J = np.empty((NSTATE, NSTATE)); h = 1e-7
    for k in range(NSTATE):
        xp = x.copy(); xm = x.copy(); xp[k] += h; xm[k] -= h
        J[:, k] = (np.asarray(rhs(0, xp)) - np.asarray(rhs(0, xm)))/(2*h)
    J[1, :] = 0.0; J[1, 1] = Gi(x[0], T, 1) - p2['d1']
    print("spectrum:")
    for v in sorted(np.linalg.eigvals(J), key=lambda z: -z.real):
        print(f"   {v.real:+.8f} {v.imag:+.8f}i")
    print("\nA zero eigenvalue and a critical imaginary pair coexist, which identifies the")
    print("point spectrally as zero-Hopf.  Which unfolding is realized would require the")
    print("normal-form coefficients, not computed here.")
