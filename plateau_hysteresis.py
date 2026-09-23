"""
plateau_hysteresis.py

Two checks requested in review.

(1) Plateau under a quadratic closure.  Under linear zooplankton mortality the grazer
    equation e*gamma*Gamma_i = nu_Z fixes the resident phytoplankton biomass at
    P_i* = kappa_P nu_Z/(e gamma - nu_Z), independent of temperature.  Under a quadratic
    closure the same equation reads e*gamma*Gamma_i*Z = nu_Z Z^2, which determines Z given
    P_i* but leaves P_i* free, so the plateau should disappear.  We verify this by solving
    the single-type equilibrium across temperature under both closures.

(2) Hysteresis.  The transitions are described as continuous exchanges of stability rather
    than catastrophic regime shifts.  We test this by continuing the coexistence state
    forward in temperature and then backward, each step started from the previous endpoint.
    Coinciding branches mean no bistability, no hysteresis and no alternative stable states.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
exec(open('model.py').read().split("def run(")[0])
P = make_p()

# ---------------- (1) plateau ----------------
def single_eq(T, i, quad=False):
    th = theta(T, P['Topt1'] if i == 0 else P['Topt2'], P['w1'], P['wskew1'])
    mu = P['mu1'] if i == 0 else P['mu2']; KN = P['KN1'] if i == 0 else P['KN2']
    d  = P['d1'] if i == 0 else P['d2'];   rh = rho(T, P['rho0'], P['Q10'], P['Tref'])
    def eqs(x):
        N, Pi, Z, D = np.abs(x)
        gi = mu*th*N/(KN+N); G = Pi/(P['KP1']+Pi)
        zl = P['dZ']*Z**2 if quad else P['dZ']*Z
        return [P['I0']-P['lN']*N-gi*Pi+rh*D,
                gi*Pi-P['mZ']*G*Z-d*Pi,
                P['e']*P['mZ']*G*Z-zl,
                d*Pi+(1-P['e'])*P['mZ']*G*Z+zl-rh*D-P['lD']*D]
    for g0 in ([0.5,1.7,2.0,8.0],[1.0,3.0,1.0,5.0],[0.3,1.0,0.6,3.0]):
        sol, _, ok, _ = fsolve(eqs, g0, full_output=True)
        if ok == 1 and np.all(np.abs(sol) > 1e-9): return np.abs(sol)
    return None

print("(1) SINGLE-TYPE PLATEAU vs TEMPERATURE   (resident = cold type)")
print(f"{'T':>6} {'P* linear':>12} {'P* quadratic':>14}")
lin, qd, Ts = [], [], np.arange(13.0, 21.01, 1.0)
for T in Ts:
    a = single_eq(T, 0, quad=False); b = single_eq(T, 0, quad=True)
    la = a[1] if a is not None else np.nan; qa = b[1] if b is not None else np.nan
    lin.append(la); qd.append(qa)
    print(f"{T:6.1f} {la:12.4f} {qa:14.4f}")
lin, qd = np.array(lin), np.array(qd)
pred = P['KP1']*P['dZ']/(P['e']*P['mZ']-P['dZ'])
print(f"\nanalytic linear-closure plateau  kappa_P nu_Z/(e gamma - nu_Z) = {pred:.4f}")
print(f"linear    : spread {np.nanmax(lin)-np.nanmin(lin):.2e}  -> "
      f"{'temperature independent' if np.nanmax(lin)-np.nanmin(lin) < 1e-6 else 'varies'}")
print(f"quadratic : spread {np.nanmax(qd)-np.nanmin(qd):.4f}  "
      f"({np.nanmin(qd):.3f} to {np.nanmax(qd):.3f}) -> "
      f"{'temperature dependent' if np.nanmax(qd)-np.nanmin(qd) > 1e-3 else 'flat'}")

# ---------------- (2) hysteresis ----------------
def rhs(t, y, T):
    N,P1,P2,Z,D = [max(x,1e-14) for x in y]
    th1 = theta(T,P['Topt1'],P['w1'],P['wskew1']); th2 = theta(T,P['Topt2'],P['w2'],P['wskew2'])
    rh = rho(T,P['rho0'],P['Q10'],P['Tref']); s = P['switch']
    a=P1**s; b=P2**s; tot=a+b+1e-12
    gP1=(a/tot)*g(P1,P['KP1']); gP2=(b/tot)*g(P2,P['KP2'])
    gr1=P['mu1']*th1*f(N,P['KN1'])*P1; gr2=P['mu2']*th2*f(N,P['KN2'])*P2
    return [P['I0']-P['lN']*N-gr1-gr2+rh*D, gr1-P['mZ']*gP1*Z-P['d1']*P1,
            gr2-P['mZ']*gP2*Z-P['d2']*P2, P['e']*P['mZ']*(gP1+gP2)*Z-P['dZ']*Z,
            P['d1']*P1+P['d2']*P2+(1-P['e'])*P['mZ']*(gP1+gP2)*Z+P['dZ']*Z-rh*D-P['lD']*D]

def sweep(Tlist, y0):
    out, y = [], np.array(y0, float)
    for T in Tlist:
        sol = solve_ivp(lambda t,x: rhs(t,x,T), [0,20000], y, method='LSODA',
                        rtol=1e-11, atol=1e-13)
        y = np.maximum(sol.y[:,-1], 1e-12)
        out.append(y.copy())
    return np.array(out)

print("\n(2) FORWARD / BACKWARD TEMPERATURE SWEEP  (continuation, iota0 = 0.5)")
Tf = np.arange(13.0, 24.001, 0.25)
F = sweep(Tf, [1.0,0.8,0.8,0.4,0.4])
B = sweep(Tf[::-1], F[-1])[::-1]
d1 = np.max(np.abs(F[:,1]-B[:,1])); d2 = np.max(np.abs(F[:,2]-B[:,2]))
print(f"  max |P1_forward - P1_backward| = {d1:.3e}")
print(f"  max |P2_forward - P2_backward| = {d2:.3e}")
tol = 1e-8
print(f"  -> {'branches coincide: NO hysteresis, no alternative stable states' if max(d1,d2) < tol else 'BRANCHES DIFFER: hysteresis present'}")
worst = Tf[np.argmax(np.abs(F[:,1]-B[:,1]))]
print(f"  largest discrepancy at T = {worst:.2f} degC")
