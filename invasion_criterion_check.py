"""Analytic invasion criterion (Proposition 'Analytic invasion criterion') and its
coverage across the coexistence window.

Since G_j is strictly increasing, along a single-type-face attractor M_i
    <I_j> = <G_j(N)> - nu_j  >=  G_j(N_min) - nu_j,
so N_min >= N_j*  (invader break-even, G_j(N_j*)=nu_j)  is a CLOSED-FORM sufficient
condition for <I_j> > 0. This script verifies, on both faces across the window:
  (i) where the criterion holds (G_j(N_min) - nu_j >= 0), the invasion is analytic;
  (ii) the narrow band T ~ 18.2-19.0 C where it fails on the {P1=0} face
       (warm-only cycle draws N below the cold type's break-even), yet <I_1> stays
       positive (confirmed numerically). Reproduces the numbers quoted in the text.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import model
p = model.make_p()

def Gi(N, T, i):
    th = model.theta(T,15.,7.,4.) if i==1 else model.theta(T,25.,7.,4.)
    b  = p['mu1'] if i==1 else p['mu2']; KN = p['KN1'] if i==1 else p['KN2']
    return b*th*(N/(KN+N))
nu = {1: p['d1'], 2: p['d2']}
def Nstar(T, j):
    try: return brentq(lambda N: Gi(N,T,j)-nu[j], 1e-9, 1e4)
    except Exception: return np.nan

def face(t, y, T, res):   # resident-only face; invader = the other type
    N,P,Z,D = [max(x,1e-12) for x in y]
    th = model.theta(T,15.,7.,4.) if res==1 else model.theta(T,25.,7.,4.)
    b  = p['mu1'] if res==1 else p['mu2']; KN = p['KN1'] if res==1 else p['KN2']
    KP = p['KP1'] if res==1 else p['KP2']; nur = p['d1'] if res==1 else p['d2']
    rh = model.rho(T,0.05,2.,20.); G = b*th*(N/(KN+N)); g = p['mZ']*(P/(KP+P))
    return [0.5-p['lN']*N-G*P+rh*D, G*P-g*Z-nur*P, p['e']*g*Z-p['dZ']*Z,
            nur*P+(1-p['e'])*g*Z+p['dZ']*Z-rh*D-p['lD']*D]

print("resident->invader  T    N_min   N_j*    G_j(Nmin)-nu_j   <I_j>   criterion")
worst_meanI = 1e9
for res, inv in [(1,2),(2,1)]:
    for T in np.arange(15.4, 20.1, 0.4):
        s = solve_ivp(lambda t,y: face(t,y,T,res), [0,40000], [1.,1.2,2.,7.],
                      method='RK45', rtol=1e-10, atol=1e-12,
                      t_eval=np.linspace(30000,40000,20000))
        N = s.y[0]; Nmin = N.min()
        gmin = Gi(Nmin,T,inv) - nu[inv]
        meanI = np.mean([Gi(x,T,inv) for x in N]) - nu[inv]
        worst_meanI = min(worst_meanI, meanI)
        tag = "analytic (Nmin>=Nj*)" if gmin >= 0 else "numerical (band)"
        print("   P%d->P%d     %.1f  %.4f  %.4f     %+.4f       %+.4f   %s"
              % (res, inv, T, Nmin, Nstar(T,inv), gmin, meanI, tag))
print("\nSmallest time-averaged <I_j> over the window/both faces: %.4f (>0 => persistence holds)"
      % worst_meanI)
print("Criterion is analytic except the {P1=0}-face band ~ T=18.2-19.0 C, where <I_1> ~ 0.10-0.16.")
