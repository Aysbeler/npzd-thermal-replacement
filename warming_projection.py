"""
warming_projection.py

Warming projection at the calibrated parameters.  The seasonal temperature signal is shifted
uniformly, T(t) -> T(t) + dT, and the non-autonomous system is integrated to its annually
periodic state for each dT.  This turns the observation that the reference bifurcation sequence
is not visible in the present-day climatology into a forward statement: how much local warming
would be needed before the model predicts the cold type to be suppressed.

Reported for each dT:
  * annual mean cold-type share P1/(P1+P2)
  * its late-winter maximum
  * the number of days per year on which T(t) exceeds the autonomous upper edge T2
  * the threshold dT* at which the cold type falls below a 1 per cent annual share
"""
import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq
import model

rows = list(csv.DictReader(open('station_climatology_point.csv')))
sst  = np.array([float(r['sst']) for r in rows])
DC   = (np.arange(12)+0.5)*365/12.0
dcx  = np.concatenate([[DC[-1]-365], DC, [DC[0]+365]])
sx   = np.concatenate([[sst[-1]], sst, [sst[0]]])
P0   = model.make_p()
O1, O2 = 15.70, 19.20                      # calibrated optima

def Tfun(t, dT):
    return float(np.interp(t % 365.0, dcx, sx)) + dT

def rates(T):
    return (model.theta(T, O1, P0['w1'], P0['wskew1']),
            model.theta(T, O2, P0['w2'], P0['wskew2']),
            model.rho(T, P0['rho0'], P0['Q10'], P0['Tref']))

# ---- autonomous upper edge at the calibrated optima ----
def single_eq(T, i):
    th = rates(T); TH = (th[0], th[1]); rh = th[2]
    MU = (P0['mu1'], P0['mu2']); KN = (P0['KN1'], P0['KN2']); DI = (P0['d1'], P0['d2'])
    def eqs(x):
        N, Pi, Z, D = np.abs(x)
        gi = MU[i]*TH[i]*N/(KN[i]+N); G = Pi/(P0['KP1']+Pi)
        return [P0['I0']-P0['lN']*N-gi*Pi+rh*D, gi*Pi-P0['mZ']*G*Z-DI[i]*Pi,
                P0['e']*P0['mZ']*G*Z-P0['dZ']*Z,
                DI[i]*Pi+(1-P0['e'])*P0['mZ']*G*Z+P0['dZ']*Z-rh*D-P0['lD']*D]
    for g0 in ([0.5,1.7,2.0,8.0],[1.0,3.0,1.0,5.0],[0.2,1.0,0.5,2.0]):
        sol,_,ok,_ = fsolve(eqs, g0, full_output=True)
        if ok == 1 and np.all(np.abs(sol) > 1e-9): return np.abs(sol)
    return None

def invasion(T, i):
    e = single_eq(T, i)
    if e is None: return np.nan
    N = e[0]; j = 1-i; th = rates(T)
    MU = (P0['mu1'], P0['mu2']); KN = (P0['KN1'], P0['KN2']); DI = (P0['d1'], P0['d2'])
    return MU[j]*th[j]*N/(KN[j]+N) - DI[j]

T2 = brentq(lambda T: invasion(T, 1), 19.0, 22.0, xtol=1e-5)
print(f"autonomous upper edge at the calibrated optima: T2 = {T2:.3f} degC\n")

def rhs(t, y, dT):
    N,P1,P2,Z,D = [max(x,1e-14) for x in y]
    th1, th2, rh = rates(Tfun(t, dT))
    s = P0['switch']; a = P1**s; b = P2**s; tot = a+b+1e-12
    gP1 = (a/tot)*model.g(P1,P0['KP1']); gP2 = (b/tot)*model.g(P2,P0['KP2'])
    gr1 = P0['mu1']*th1*model.f(N,P0['KN1'])*P1
    gr2 = P0['mu2']*th2*model.f(N,P0['KN2'])*P2
    return [P0['I0']-P0['lN']*N-gr1-gr2+rh*D, gr1-P0['mZ']*gP1*Z-P0['d1']*P1,
            gr2-P0['mZ']*gP2*Z-P0['d2']*P2, P0['e']*P0['mZ']*(gP1+gP2)*Z-P0['dZ']*Z,
            P0['d1']*P1+P0['d2']*P2+(1-P0['e'])*P0['mZ']*(gP1+gP2)*Z+P0['dZ']*Z-rh*D-P0['lD']*D]

def annual_cycle(dT, years=12):
    sol = solve_ivp(lambda t,y: rhs(t,y,dT), [0, years*365], [1,0.5,0.5,0.4,2.0],
                    method='RK45', rtol=1e-9, atol=1e-11, max_step=2.0, dense_output=True)
    t0 = (years-1)*365.0; tt = np.linspace(t0, t0+365, 4000)
    Y = sol.sol(tt); frac = Y[1]/(Y[1]+Y[2])
    Tt = np.array([Tfun(t, dT) for t in tt])
    return frac, Tt, tt-t0

print(f"{'dT':>5} {'mean P1 share':>15} {'winter max':>12} {'days T>T2':>11}")
print("-"*48)
res = []
for dT in [0.0, 1.0, 2.0, 3.0, 4.0]:
    frac, Tt, day = annual_cycle(dT)
    days = np.mean(Tt > T2)*365
    res.append((dT, frac.mean(), frac.max(), days))
    print(f"{dT:5.1f} {frac.mean():15.4f} {frac.max():12.4f} {days:11.0f}")

# threshold at which the annual cold-type share drops below one per cent
def share(dT): return annual_cycle(dT)[0].mean() - 0.01
lo, hi = 0.0, 6.0
if share(hi) < 0 < share(lo):
    for _ in range(14):
        mid = 0.5*(lo+hi)
        if share(mid) > 0: lo = mid
        else: hi = mid
    print(f"\ncold type falls below a 1% annual share at dT* = {0.5*(lo+hi):.2f} degC")
else:
    print(f"\nno 1% crossing within dT in [0,6]; share at dT=6 is {share(6.0)+0.01:.4f}")


# ---- threshold distribution over the warm-optimum bootstrap interval ----
def threshold_for(o2_val):
    global O2
    O2_save = O2; O2 = o2_val
    def share(dT): return annual_cycle(dT)[0].mean() - 0.01
    lo, hi = 0.0, 6.0
    out = None
    if share(hi) < 0 < share(lo):
        for _ in range(12):
            mid = 0.5*(lo+hi)
            if share(mid) > 0: lo = mid
            else: hi = mid
        out = 0.5*(lo+hi)
    O2 = O2_save
    return out

print("\nthreshold dT* across the warm-optimum 95% bootstrap interval [18.8, 22.8]:")
import numpy as _np
_dist = []
for o2 in _np.linspace(18.8, 22.8, 9):
    th = threshold_for(o2)
    print(f"  Topt2={o2:5.2f} -> dT* = {th:.2f}" if th else f"  Topt2={o2:5.2f} -> no crossing")
    if th: _dist.append(th)
if _dist:
    print(f"\n  dT* range over the interval: {min(_dist):.1f} to {max(_dist):.1f} C, median {_np.median(_dist):.1f}")
