"""
mechanism_factorial.py

Addresses two referee requests:

(A2) TRUE no-switching controls. The reference weight phi_i = P_i^s/(P1^s+P2^s) is
     frequency dependent even at s=1, and gives a rare type a grazing loss of order
     O(P_i^2), i.e. a rare-prey refuge. Two genuinely non-switching responses are added:

       'fixed'    fixed preferences: gflux_i = q_i * P_i/(K_P + P_i),  q_1=q, q_2=1-q
       'multiprey' shared multi-prey Holling II (no preference):
                  gflux_i = P_i / (K_P + P_1 + P_2)

     Both give grazing loss O(P_i) on a rare type: no refuge.
     Note on fairness: 'multiprey' reduces exactly to the single-prey Holling II response
     when one type is absent, so it is directly comparable to the reference. 'fixed' does
     not (effort allocated to an absent type is wasted), so its single-type plateau shifts;
     this is the genuine meaning of non-adaptive preference and is reported as such.

(B1) FACTORIAL trait decomposition, 2x2:
       thermal optimum contrast  on/off   x   nutrient affinity contrast  on/off
     'off' = both types given the midpoint value, so the two traits are switched off
     independently without changing the average.

Protocol matches Section 7 (T in [13,24] step 0.5, iota0=0.5, integrate to t=6000,
type present if terminal value > 1e-2). A drift diagnostic is reported so that slow
transients are not mistaken for coexistence.
"""
import numpy as np
from scipy.integrate import solve_ivp
exec(open('model.py').read().split("def run(")[0])   # theta, rho, f, g, make_p

# ---------------- trait scenarios (B1) ----------------
# reference: Topt=(15,25), KN=(1.2,0.5).  midpoints: Topt=20, KN=0.85
SCEN = {
 'A  thermal+ affinity+': dict(Topt=(15.0, 25.0), KN=(1.2, 0.5)),
 'B  thermal+ affinity-': dict(Topt=(15.0, 25.0), KN=(0.85, 0.85)),
 'C  thermal- affinity+': dict(Topt=(20.0, 20.0), KN=(1.2, 0.5)),
 'D  thermal- affinity-': dict(Topt=(20.0, 20.0), KN=(0.85, 0.85)),
}

def rhs_factory(graz='switch', Topt=(15.0,25.0), KN=(1.2,0.5), s=2.0, q=0.5):
    def rhs(t, y, T, I0):
        N,P1,P2,Z,D = [max(x,1e-12) for x in y]
        p = make_p(T=T, I0=I0)
        th1 = theta(T, Topt[0], p['w1'], p['wskew1'])
        th2 = theta(T, Topt[1], p['w2'], p['wskew2'])
        rh  = rho(T, p['rho0'], p['Q10'], p['Tref'])
        fN1 = f(N, KN[0]); fN2 = f(N, KN[1])
        KP  = p['KP1']

        if graz == 'switch':
            w1s, w2s = P1**s, P2**s
            tot = w1s + w2s + 1e-12
            gf1 = (w1s/tot)*g(P1,KP); gf2 = (w2s/tot)*g(P2,KP)
        elif graz == 'fixed':
            # fixed preferences split the grazer's effort between the two types, so the
            # grazing rate is doubled to keep the total pressure comparable to the
            # switching case, where the weights sum to one: q = 1/2 with gamma -> 2 gamma
            gf1 = 2.0*q*g(P1,KP); gf2 = 2.0*(1.0-q)*g(P2,KP)
        elif graz == 'multiprey':
            den = KP + P1 + P2
            gf1 = P1/den; gf2 = P2/den
        else:
            raise ValueError(graz)

        gr1 = p['mu1']*th1*fN1*P1; gr2 = p['mu2']*th2*fN2*P2; remin = rh*D
        mZ, e, dZ = p['mZ'], p['e'], p['dZ']
        return [p['I0'] - p['lN']*N - gr1 - gr2 + remin,
                gr1 - mZ*gf1*Z - p['d1']*P1,
                gr2 - mZ*gf2*Z - p['d2']*P2,
                e*mZ*(gf1+gf2)*Z - dZ*Z,
                p['d1']*P1 + p['d2']*P2 + (1-e)*mZ*(gf1+gf2)*Z + dZ*Z - remin - p['lD']*D]
    return rhs

def analyse(graz, scen, thr=1e-2):
    rhs = rhs_factory(graz=graz, **SCEN[scen])
    Ts = np.arange(13.0, 24.01, 0.5)
    coex, drift_max = [], 0.0
    for T in Ts:
        sol = solve_ivp(lambda t,y: rhs(t,y,T,0.5), [0,6000], [1.0,0.8,0.8,0.4,0.4],
                        method='RK45', rtol=1e-8, atol=1e-10)
        P1e, P2e = sol.y[1,-1], sol.y[2,-1]
        if P1e > thr and P2e > thr:
            coex.append(T)
            # drift diagnostic: relative growth rate of the minority type at the end
            dy = rhs(0, sol.y[:,-1], T, 0.5)
            Pm, dPm = (P1e, dy[1]) if P1e < P2e else (P2e, dy[2])
            drift_max = max(drift_max, abs(dPm/Pm))
    if not coex:
        return None, 0.0
    return (coex[0], coex[-1]), drift_max

print("VALIDATION (scenario A, switching) - target window [15.5,20.0]")
w,d = analyse('switch','A  thermal+ affinity+')
print(f"   window={w}  max|dlnP_min/dt|={d:.2e}\n")

print("="*74)
print("A2  TRUE NO-SWITCHING CONTROLS  (reference traits, scenario A)")
print("="*74)
for graz,label in [('switch','active switching  phi=P^s/(sum P^s), s=2'),
                   ('fixed','fixed preferences  q=0.5 (non-adaptive)'),
                   ('multiprey','shared multi-prey Holling II (no preference)')]:
    w,d = analyse(graz,'A  thermal+ affinity+')
    ws = f"[{w[0]:.1f},{w[1]:.1f}]" if w else "no coexistence"
    print(f"  {label:46s} {ws:16s} drift={d:.1e}")

print()
print("="*74)
print("B1  FACTORIAL TRAIT DECOMPOSITION  (window per grazing form)")
print("="*74)
print(f"  {'scenario':24s} {'switching':>16s} {'fixed pref.':>16s} {'multi-prey':>16s}")
for scen in SCEN:
    row=[]
    for graz in ['switch','fixed','multiprey']:
        w,d = analyse(graz,scen)
        row.append(f"[{w[0]:.1f},{w[1]:.1f}]" if w else "none")
    print(f"  {scen:24s} {row[0]:>16s} {row[1]:>16s} {row[2]:>16s}")


# ---------------------------------------------------------------------------
# Exclusion exchange and grazer floor under the non-switching responses
# ---------------------------------------------------------------------------
# The manuscript reports the temperature at which the two invasion fitnesses change
# sign together, and the smallest grazer stock encountered, so that the exclusion can
# be attributed to competition rather than to a collapse of the top-down loop.

def _settle(rhs, T, I0=0.5, tend=60000.0):
    sol = solve_ivp(lambda t, y: rhs(t, y, T, I0), [0, tend], [1.0, 0.8, 0.8, 0.4, 0.4],
                    method='LSODA', rtol=1e-9, atol=1e-11)
    return sol.y[:, -1]

def exclusion_exchange(graz, scen='A  thermal+ affinity+', lo=16.5, hi=20.0):
    """Temperature where the surviving type changes, with the grazer stock there."""
    rhs = rhs_factory(graz=graz, **SCEN[scen])
    def gap(T):
        y = _settle(rhs, T)
        return y[1] - y[2]
    Tx = brentq(gap, lo, hi, xtol=1e-4)
    Zs = [_settle(rhs, T)[3] for T in np.linspace(lo, hi, 25)]
    return Tx, min(Zs)

print()
print("=" * 74)
print("A3  EXCLUSION EXCHANGE AND GRAZER FLOOR (reference traits, scenario A)")
print("=" * 74)
for graz in ('fixed', 'multiprey'):
    try:
        Tx, Zmin = exclusion_exchange(graz)
        print(f"  {graz:10s} exchange at T = {Tx:.3f} C,  min grazer stock Z = {Zmin:.4f}")
    except Exception as exc:
        print(f"  {graz:10s} could not be bracketed: {exc}")
