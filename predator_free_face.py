"""
predator_free_face.py

Tests the analytic condition that rules out two-type invariant sets on the
predator-free face {Z=0} of the model.

On that face the dynamics are

    N' = I0 - lN N - g1(N) P1 - g2(N) P2 + rho(T) D
    Pi'= Pi ( gi(N) - di ),      gi(N) = mu_i theta_i(T) N/(KN_i + N)
    D' = d1 P1 + d2 P2 - (rho(T)+lD) D

Because each per-capita growth rate Gi(N) = gi(N) - di depends only on N, any
compact invariant set K with P1,P2 > 0 carries an invariant measure mu with

    int Gi dmu = 0,   i = 1,2.

If there exists lambda > 0 with G_j(N) - lambda G_i(N) < 0 for every attainable N
(i = better competitor, i.e. smaller break-even N_i*), then

    0 = int G_j = lambda * 0 + int (G_j - lambda G_i) < 0,

a contradiction, so no such K exists.  Writing r(N) = G_j(N)/G_i(N), the condition is

    sup_{N > N_i*} r(N)  <  inf_{0 <= N < N_i*} r(N),

and any lambda strictly between the two works.  This script evaluates both sides
on a fine grid over the attainable range of N, for each temperature.
"""
import numpy as np
exec(open('model.py').read().split("def run(")[0])   # theta, rho, f, g, make_p

p = make_p()
MU = (p['mu1'], p['mu2']); KN = (p['KN1'], p['KN2']); DD = (p['d1'], p['d2'])
TOPT = (p['Topt1'], p['Topt2']); W = (p['w1'], p['w2']); WS = (p['wskew1'], p['wskew2'])

def Gi(N, i, T):
    th = theta(T, TOPT[i], W[i], WS[i])
    return MU[i]*th*N/(KN[i]+N) - DD[i]

def breakeven(i, T):
    """N_i* solving G_i = 0, or None if the type cannot grow at any N."""
    th = theta(T, TOPT[i], W[i], WS[i])
    num = MU[i]*th - DD[i]
    if num <= 0:
        return None
    return KN[i]*DD[i]/num

# The general criterion is a separating-hyperplane test: writing
#     C(T) = { (G_1(N), G_2(N)) : N attainable },
# no invariant measure can satisfy int G_1 = int G_2 = 0 unless the origin lies in
# the closed convex hull of C(T).  Equivalently, if all points of C(T) lie strictly
# inside one open half-plane through the origin, no two-type invariant set exists.
# This is implemented below as an angular-gap test and is strictly stronger than the
# one-sided lambda test.

NMAX = 10.0    # asymptotic mass bound on the face: M <= I0/min(lN,lD) = 10

def halfplane_proof(T, Nmin=0.0, Nmax=NMAX, n=200000):
    """True if all (G1,G2) lie in an open half-plane through the origin."""
    N = np.linspace(max(Nmin,1e-12), Nmax, n)
    ang = np.sort(np.arctan2(Gi(N,1,T), Gi(N,0,T)))
    gaps = np.diff(np.concatenate([ang, [ang[0]+2*np.pi]]))
    return gaps.max() > np.pi
grid = np.linspace(1e-9, NMAX, 400000)

print(f"{'T':>5} {'N1*':>9} {'N2*':>9} {'better':>7} {'sup r':>11} {'inf r':>11} {'result':>26}")
print("-"*84)
rows = []
for T in np.arange(13.0, 24.01, 0.5):
    b = [breakeven(0, T), breakeven(1, T)]
    if b[0] is None or b[1] is None:
        k = 0 if b[0] is None else 1
        print(f"{T:5.1f} {'--' if b[0] is None else f'{b[0]:.4f}':>9} "
              f"{'--' if b[1] is None else f'{b[1]:.4f}':>9} {'':>7} {'':>11} {'':>11} "
              f"{'P'+str(k+1)+' cannot grow: trivial':>26}")
        rows.append((T, True, 'trivial'))
        continue
    i = 0 if b[0] < b[1] else 1      # better competitor
    j = 1 - i
    Gi_v = Gi(grid, i, T); Gj_v = Gi(grid, j, T)
    hi = grid > b[i]                 # G_i > 0 here
    lo = grid < b[i]                 # G_i < 0 here
    sup_r = np.max(Gj_v[hi]/Gi_v[hi])
    inf_r = np.min(Gj_v[lo]/Gi_v[lo])
    ok = sup_r < inf_r
    rows.append((T, ok, (sup_r, inf_r)))
    print(f"{T:5.1f} {b[0]:9.4f} {b[1]:9.4f} {'P'+str(i+1):>7} {sup_r:11.4f} {inf_r:11.4f} "
          f"{('HOLDS  lambda in gap' if ok else 'FAILS'):>26}")

nfail = sum(1 for _,ok,_ in rows if not ok)
print("-"*84)
print(f"condition holds at {len(rows)-nfail}/{len(rows)} temperatures; fails at {nfail}")
if nfail:
    print("failing temperatures:", [f"{T:.1f}" for T,ok,_ in rows if not ok])
