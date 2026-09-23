"""
quasistatic_check.py

Tests the claim that under seasonal forcing the trajectory quasi-statically tracks the
branch of autonomous attractors, so that the transcritical edges predict when in the year
the dominant phytoplankton type changes.

The calibration already integrates the full non-autonomous system with T = T(t); the
question is whether the autonomous edges predict the timing of the composition handover,
or whether critical slowing down near the transcritical points introduces a lag.

We compare, at the calibrated optima:

  (i)  the non-autonomous annually periodic solution,
  (ii) the quasi-static prediction, i.e. the autonomous attractor evaluated at the
       instantaneous temperature T(t),
  (iii) the temperatures at which T(t) crosses the autonomous transcritical edges.

Output: the handover dates predicted by each, and the lag between them.
"""
import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, fsolve
import model

rows = list(csv.DictReader(open('station_climatology_point.csv')))
sst  = np.array([float(r['sst']) for r in rows])
DC   = (np.arange(12)+0.5)*365/12.0
dcx  = np.concatenate([[DC[-1]-365], DC, [DC[0]+365]])
sx   = np.concatenate([[sst[-1]], sst, [sst[0]]])
Tfun = lambda t: float(np.interp(t % 365.0, dcx, sx))

P0 = model.make_p()
O1, O2 = 15.70, 19.20              # calibrated optima

def rates(T):
    th1 = model.theta(T, O1, P0['w1'], P0['wskew1'])
    th2 = model.theta(T, O2, P0['w2'], P0['wskew2'])
    return th1, th2, model.rho(T, P0['rho0'], P0['Q10'], P0['Tref'])

# ---------------- autonomous machinery: edges and attractors ----------------
def single_type_eq(T, i):
    """Equilibrium with resident i, invader absent."""
    th = rates(T); TH = (th[0], th[1]); rh = th[2]
    MU = (P0['mu1'], P0['mu2']); KN = (P0['KN1'], P0['KN2']); DI = (P0['d1'], P0['d2'])
    def eqs(x):
        N, Pi, Z, D = np.abs(x)
        gi = MU[i]*TH[i]*N/(KN[i]+N); G = Pi/(P0['KP1']+Pi)
        return [P0['I0']-P0['lN']*N-gi*Pi+rh*D,
                gi*Pi-P0['mZ']*G*Z-DI[i]*Pi,
                P0['e']*P0['mZ']*G*Z-P0['dZ']*Z,
                DI[i]*Pi+(1-P0['e'])*P0['mZ']*G*Z+P0['dZ']*Z-rh*D-P0['lD']*D]
    for g0 in ([0.5,1.7,2.0,8.0],[1.0,3.0,1.0,5.0],[0.2,1.0,0.5,2.0]):
        sol, _, ok, _ = fsolve(eqs, g0, full_output=True)
        if ok == 1 and np.all(np.abs(sol) > 1e-9): return np.abs(sol)
    return None

def invasion(T, i):
    e = single_type_eq(T, i)
    if e is None: return np.nan
    N = e[0]; j = 1-i; th = rates(T)
    MU = (P0['mu1'], P0['mu2']); KN = (P0['KN1'], P0['KN2']); DI = (P0['d1'], P0['d2'])
    return MU[j]*th[j]*N/(KN[j]+N) - DI[j]      # frequency-dependent refuge: no grazing term

Ts = np.linspace(11, 25, 400)
I2v = np.array([invasion(T,0) for T in Ts])
I1v = np.array([invasion(T,1) for T in Ts])
both = np.isfinite(I1v) & np.isfinite(I2v) & (I1v > 0) & (I2v > 0)
idx = np.where(both)[0]

def safe_root(fv, i):
    """root of the invasion function whose sign changes between Ts[i-1] and Ts[i]"""
    for k in (0, 1):
        f = (lambda T: invasion(T, k))
        a, b = Ts[i-1], Ts[i]
        try:
            fa, fb = f(a), f(b)
            if np.isfinite(fa) and np.isfinite(fb) and fa*fb < 0:
                return brentq(f, a, b, xtol=1e-5)
        except Exception:
            pass
    return None

T1 = safe_root(None, idx[0]) if idx[0] > 0 else None
T2 = safe_root(None, idx[-1]+1) if idx[-1] < len(Ts)-1 else None
print(f"autonomous edges at the calibrated optima (15.70, 19.20):")
print(f"  lower edge T1 = {'below the scanned range' if T1 is None else f'{T1:.3f} degC'}")
print(f"  upper edge T2 = {'above the scanned range' if T2 is None else f'{T2:.3f} degC'}")
print(f"site SST range: {sst.min():.2f} - {sst.max():.2f} degC\n")

# ---------------- non-autonomous annual cycle ----------------
def rhs(t, y):
    N,P1,P2,Z,D = [max(x,1e-12) for x in y]
    th1, th2, rh = rates(Tfun(t))
    s = P0['switch']; a=P1**s; b=P2**s; tot=a+b+1e-12
    gP1 = (a/tot)*model.g(P1,P0['KP1']); gP2 = (b/tot)*model.g(P2,P0['KP2'])
    gr1 = P0['mu1']*th1*model.f(N,P0['KN1'])*P1
    gr2 = P0['mu2']*th2*model.f(N,P0['KN2'])*P2
    return [P0['I0']-P0['lN']*N-gr1-gr2+rh*D,
            gr1-P0['mZ']*gP1*Z-P0['d1']*P1,
            gr2-P0['mZ']*gP2*Z-P0['d2']*P2,
            P0['e']*P0['mZ']*(gP1+gP2)*Z-P0['dZ']*Z,
            P0['d1']*P1+P0['d2']*P2+(1-P0['e'])*P0['mZ']*(gP1+gP2)*Z+P0['dZ']*Z-rh*D-P0['lD']*D]

YEARS = 12
sol = solve_ivp(rhs, [0, YEARS*365], [1,0.5,0.5,0.4,2.0], method='RK45',
                rtol=1e-9, atol=1e-11, max_step=2.0, dense_output=True)
t0 = (YEARS-1)*365.0
tt = np.linspace(t0, t0+365, 7300)
Y  = sol.sol(tt); P1t, P2t = Y[1], Y[2]
day = tt - t0
Tt  = np.array([Tfun(t) for t in tt])

def crossings(f, x):
    """days where f changes sign, with linear interpolation."""
    out = []
    for k in range(len(f)-1):
        if np.isfinite(f[k]) and np.isfinite(f[k+1]) and f[k]*f[k+1] < 0:
            out.append(x[k] + (x[k+1]-x[k])*abs(f[k])/(abs(f[k])+abs(f[k+1])))
    return out

# quasi-static prediction: the dominant type changes when T(t) crosses the edges
qs_cross = (crossings(Tt - T2, day) if T2 is not None else []) \
         + (crossings(Tt - T1, day) if T1 is not None else [])
# actual handover in the forced solution: when P1 - P2 changes sign
act_cross = crossings(P1t - P2t, day)

MONTH = lambda d: 1 + int((d % 365)/365*12)
print("quasi-static prediction, days when T(t) crosses an autonomous edge:")
for d in sorted(qs_cross):
    print(f"   day {d:6.1f}  (month {MONTH(d)})   T={np.interp(d, day, Tt):.2f}")
print("\nnon-autonomous solution, days when the dominant type actually changes:")
if not act_cross:
    dom = 'P1' if P1t.mean() > P2t.mean() else 'P2'
    print(f"   none: {dom} dominates all year "
          f"(min P1/P2 ratio {np.min(P1t/P2t):.3f}, max {np.max(P1t/P2t):.3f})")
else:
    for d in sorted(act_cross):
        print(f"   day {d:6.1f}  (month {MONTH(d)})   T={np.interp(d, day, Tt):.2f}")

if qs_cross and act_cross:
    lags = [min(abs(a-q) for q in qs_cross) for a in sorted(act_cross)]
    print(f"\nlag between predicted and actual handover: "
          + ", ".join(f"{l:.1f} d" for l in lags))

lo = -1e9 if T1 is None else T1; hi = 1e9 if T2 is None else T2
print(f"\nfraction of the year inside the autonomous window: {np.mean((Tt>lo)&(Tt<hi)):.2f}")
print(f"annual mean P1 = {P1t.mean():.4f}, P2 = {P2t.mean():.4f}, "
      f"P1 share = {P1t.mean()/(P1t.mean()+P2t.mean()):.3f}")
