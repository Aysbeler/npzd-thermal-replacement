"""
invasion_edges.py

Recomputes the coexistence-window edges from the roots of the invasion fitness at
the single-type equilibria, replacing the finite-time biomass criterion used in the
Monte-Carlo and structural-robustness scans.

For each temperature and each functional-form variant we

  1. solve the single-type equilibrium E_i (resident i, invader j absent) exactly,
     by rootfinding on the four steady-state equations, and
  2. evaluate the invasion fitness of the absent type,

        I_j(T) = mu_j theta_j(T) f_j(N_i*) - d_j - gamma Z_i* L_j ,

     where L_j = lim_{P_j -> 0} Phi_j(P_1,P_2)/P_j is the linear grazing pressure felt
     by a rare invader.  For the frequency-dependent weight phi_j = P_j^s/(P_1^s+P_2^s)
     one has L_j = 0 for every s >= 1: a rare type escapes grazing at linear order.
     For the background-grazing, fixed-preference and multi-prey responses L_j > 0.

The window edges are the temperatures at which I_j changes sign, located by bisection.
This avoids the critical-slowing-down bias of a finite-time biomass threshold near a
transcritical bifurcation.
"""
import numpy as np
from scipy.optimize import brentq, fsolve
import warnings; warnings.filterwarnings('ignore')
exec(open('model.py').read().split("def run(")[0])   # theta, rho, f, g, make_p

P = make_p()

def theta_sym(T, Topt, w, wskew):      # symmetric Gaussian variant
    return np.exp(-((T-Topt)/w)**2)

VARIANTS = {
 'reference (s=2)'              : dict(),
 'symmetric Gaussian theta'     : dict(sym=True),
 'Holling III grazing'          : dict(holling=3),
 'linear switching (s=1)'       : dict(s=1.0),
 'strong switching (s=3)'       : dict(s=3.0),
 'eps-background (eps=0.1)'     : dict(eps=0.1),
 'plankton export (0.02)'       : dict(lP=0.02, lZ=0.02),
 'T-dep grazing/mort (Q10=2)'   : dict(Tgraz=True),
 'quadratic closure'            : dict(quad=True),
 'fixed preference (fair)'      : dict(fixed=0.5),
 'multi-prey Holling II'        : dict(multiprey=True),
}

def build(v, T):
    """Return (resid, invasion) for variant v at temperature T."""
    s      = v.get('s', 2.0)
    th     = theta_sym if v.get('sym') else theta
    hz     = v.get('holling', 2)
    eps    = v.get('eps', 0.0)
    lP, lZ = v.get('lP', 0.0), v.get('lZ', 0.0)
    quad   = v.get('quad', False)
    fixedq = v.get('fixed', None)
    mp     = v.get('multiprey', False)
    mZ, dZ = P['mZ'], P['dZ']
    if v.get('Tgraz'):
        fac = 2.0**((T-P['Tref'])/10.0); mZ *= fac; dZ *= fac
    if fixedq is not None:
        mZ = mZ/fixedq                                     # fairness compensation
    KP = P['KP1']
    def gfun(x):                                           # grazing functional response
        return x**2/(KP**2+x**2) if hz == 3 else x/(KP+x)

    th1 = th(T, P['Topt1'], P['w1'], P['wskew1'])
    th2 = th(T, P['Topt2'], P['w2'], P['wskew2'])
    TH  = (th1, th2); MU = (P['mu1'], P['mu2']); KN = (P['KN1'], P['KN2']); DI = (P['d1'], P['d2'])

    def resident_flux(Pi):                                 # Phi_i with the invader absent
        if mp:      return Pi/(KP+Pi)
        if fixedq is not None: return fixedq*gfun(Pi)
        if eps > 0: return (eps+Pi**s)/(2*eps+Pi**s)*gfun(Pi)
        return gfun(Pi)                                    # phi_i = 1 when P_j = 0

    def invader_L(Pi):                                     # lim_{Pj->0} Phi_j / Pj
        if mp:      return 1.0/(KP+Pi)
        if fixedq is not None: return (1-fixedq)*(1.0/KP if hz == 2 else 0.0)
        if eps > 0: return eps/((2*eps+Pi**s)*KP)
        return 0.0                                         # frequency-dependent refuge

    def eqs(x, i):
        N, Pi, Z, D = np.abs(x)
        gi   = MU[i]*TH[i]*N/(KN[i]+N)
        Gam  = resident_flux(Pi)
        rh   = rho(T, P['rho0'], P['Q10'], P['Tref'])
        zloss = P['dZ']*Z**2 if quad else dZ*Z
        return [P['I0'] - P['lN']*N - gi*Pi + rh*D,
                gi*Pi - mZ*Gam*Z - DI[i]*Pi - lP*Pi,
                P['e']*mZ*Gam*Z - zloss - lZ*Z,
                DI[i]*Pi + (1-P['e'])*mZ*Gam*Z + (zloss if not quad else P['dZ']*Z**2)
                  - rh*D - P['lD']*D]

    def equilibrium(i):
        for g0 in ([0.5, 1.7, 2.0, 8.0], [1.0, 3.0, 1.0, 5.0], [0.2, 1.0, 0.5, 2.0]):
            sol, info, ok, _ = fsolve(eqs, g0, args=(i,), full_output=True)
            if ok == 1 and np.all(np.abs(sol) > 1e-9):
                return np.abs(sol)
        return None

    def invasion(i):
        e = equilibrium(i)
        if e is None: return np.nan
        N, Pi, Z, D = e
        j = 1-i
        return MU[j]*TH[j]*N/(KN[j]+N) - DI[j] - lP - mZ*Z*invader_L(Pi)
    return invasion

def edges(v, lo=12.0, hi=30.0, n=400):
    """Window = temperatures where BOTH invasion fitnesses are positive."""
    Ts = np.linspace(lo, hi, n)
    both = []
    for T in Ts:
        f = build(v, T)
        I2, I1 = f(0), f(1)            # type2 invading resident1 ; type1 invading resident2
        both.append(np.isfinite(I1) and np.isfinite(I2) and I1 > 0 and I2 > 0)
    both = np.array(both)
    if not both.any(): return None
    idx = np.where(both)[0]
    lo_i, hi_i = idx[0], idx[-1]
    def refine(a, b, which):
        try:
            return brentq(lambda T: (build(v, T)(0) if which == 0 else build(v, T)(1)), a, b, xtol=1e-4)
        except Exception:
            return np.nan
    # lower edge: the invasion that switches sign just below Ts[lo_i]
    lowE = np.nan; highE = np.nan
    if lo_i > 0:
        for which in (0, 1):
            r = refine(Ts[lo_i-1], Ts[lo_i], which)
            if np.isfinite(r): lowE = r if np.isnan(lowE) else max(lowE, r)
    if hi_i < len(Ts)-1:
        for which in (0, 1):
            r = refine(Ts[hi_i], Ts[hi_i+1], which)
            if np.isfinite(r): highE = r if np.isnan(highE) else min(highE, r)
    return lowE, highE

FINITE = {   # finite-time grid results currently in the manuscript, for comparison
 'reference (s=2)':'[15.5,20.0]', 'symmetric Gaussian theta':'[15.5,23.0]',
 'Holling III grazing':'[15.5,20.0]', 'linear switching (s=1)':'[16.0,20.0]',
 'strong switching (s=3)':'[15.5,20.0]', 'eps-background (eps=0.1)':'[16.0,19.5]',
 'plankton export (0.02)':'[16.5,19.0]', 'T-dep grazing/mort (Q10=2)':'[15.5,20.0]',
 'quadratic closure':'[15.5,19.5]', 'fixed preference (fair)':'none',
 'multi-prey Holling II':'none',
}

print(f"{'variant':32s} {'invasion-fitness edges':>26s} {'finite-time grid':>18s}")
print("-"*80)
for name, v in VARIANTS.items():
    e = edges(v)
    txt = "no coexistence" if e is None else (
          f"[{e[0]:.2f},{e[1]:.2f}]" if np.isfinite(e[0]) and np.isfinite(e[1])
          else f"[{e[0]:.2f},{e[1]:.2f}]".replace('nan','--'))
    print(f"{name:32s} {txt:>26s} {FINITE.get(name,''):>18s}")
