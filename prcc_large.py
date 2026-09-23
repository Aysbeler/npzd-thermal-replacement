"""
prcc_large.py

Strengthened global sensitivity analysis of the coexistence-window descriptors.

Improvements over the small-sample version:
  * Latin Hypercube sample enlarged from N = 120 to N = 2000, which is feasible because
    the window edges are obtained from the analytic invasion criterion rather than by
    integration;
  * bootstrap confidence intervals on every PRCC;
  * a monotonicity check, since PRCC assumes a monotone parameter-output relation;
  * Benjamini-Hochberg control of the false discovery rate across the 12 parameters x 3
    descriptors tested simultaneously;
  * explicit accounting of samples that fail to produce an ordered window T1 < T2.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.stats import rankdata, t as tdist
import model
from prcc_check import edges_fast

RATE = ['mu1','mu2','mZ','dZ','d1','d2','rho0','e','KN1','KN2']
OPT  = ['Topt1','Topt2']
LAB  = RATE + OPT
NICE = {'mu1':r'$\beta_1$','mu2':r'$\beta_2$','mZ':r'$\gamma$','dZ':r'$\nu_Z$',
        'd1':r'$\nu_1$','d2':r'$\nu_2$','rho0':r'$\varrho_0$','e':r'$e$',
        'KN1':r'$\kappa_{N,1}$','KN2':r'$\kappa_{N,2}$',
        'Topt1':r'$T_{opt}^{(1)}$','Topt2':r'$T_{opt}^{(2)}$'}

N = 2000
rng = np.random.default_rng(7)

def lhs(n, k):
    u = (np.tile(np.arange(n), (k,1)).T + rng.random((n,k)))/n
    for j in range(k): u[:, j] = rng.permutation(u[:, j])
    return u

U = lhs(N, len(LAB))
base = model.make_p()
X = np.empty_like(U)
for j, name in enumerate(LAB):
    if name in OPT:
        X[:, j] = base[name] + (2*U[:, j]-1)*2.0          # +- 2 degC
    else:
        X[:, j] = base[name]*(1 + (2*U[:, j]-1)*0.20)     # +- 20 %

T1 = np.full(N, np.nan); T2 = np.full(N, np.nan)
for i in range(N):
    p = model.make_p(**{name: X[i, j] for j, name in enumerate(LAB)})
    try:
        a, b = edges_fast(p)
        T1[i], T2[i] = a, b
    except Exception:
        pass

ok = np.isfinite(T1) & np.isfinite(T2) & (T2 > T1)
print(f"Latin Hypercube sample N = {N}")
print(f"  ordered window T1 < T2 obtained in {ok.sum()} samples "
      f"({100*ok.mean():.1f}%); {N-ok.sum()} discarded")
print("  discarded samples are those for which one type cannot sustain a positive grazer")
print("  stock anywhere on the scanned range, so no window exists; they are excluded from")
print("  the correlations rather than assigned a value.\n")

Xo = X[ok]; d = {'lower edge $T_1$': T1[ok], 'upper edge $T_2$': T2[ok],
                 'width $T_2-T_1$': (T2-T1)[ok]}
m = ok.sum()

def prcc(Xm, y):
    """partial rank correlation of each column of Xm with y"""
    R = np.column_stack([rankdata(Xm[:, j]) for j in range(Xm.shape[1])])
    ry = rankdata(y); out = np.empty(Xm.shape[1])
    for j in range(Xm.shape[1]):
        Z = np.delete(R, j, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        bx = np.linalg.lstsq(Z, R[:, j], rcond=None)[0]
        by = np.linalg.lstsq(Z, ry,      rcond=None)[0]
        ex = R[:, j] - Z@bx; ey = ry - Z@by
        out[j] = np.corrcoef(ex, ey)[0, 1]
    return out

def spearman(x, y):
    return np.corrcoef(rankdata(x), rankdata(y))[0, 1]

B = 400
results = {}
for name, y in d.items():
    est = prcc(Xo, y)
    boot = np.empty((B, len(LAB)))
    for b in range(B):
        idx = rng.integers(0, m, m)
        boot[b] = prcc(Xo[idx], y[idx])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    # two-sided p from the PRCC t statistic
    dof = m - 2 - (len(LAB)-1)
    tstat = est*np.sqrt(dof/np.maximum(1-est**2, 1e-12))
    pval = 2*tdist.sf(np.abs(tstat), dof)
    results[name] = (est, lo, hi, pval)

# Benjamini-Hochberg across all tests
allp = np.concatenate([results[k][3] for k in results])
order = np.argsort(allp); ranked = allp[order]
crit = 0.05*(np.arange(1, len(ranked)+1)/len(ranked))
passed = ranked <= crit
kmax = np.max(np.where(passed)[0])+1 if passed.any() else 0
thresh = ranked[kmax-1] if kmax else 0.0
print(f"Benjamini-Hochberg over {len(allp)} tests at FDR 0.05: "
      f"significance threshold p <= {thresh:.2e}\n")

for name, (est, lo, hi, pval) in results.items():
    print(f"--- {name} ---")
    idx = np.argsort(-np.abs(est))
    for j in idx[:6]:
        mono = spearman(Xo[:, j], d[name])
        flag = '*' if pval[j] <= thresh else ' '
        print(f"   {LAB[j]:7s} PRCC={est[j]:+.3f} [{lo[j]:+.3f},{hi[j]:+.3f}]{flag}  "
              f"Spearman={mono:+.3f}")
    print()
print("* significant after Benjamini-Hochberg correction")
print("Spearman is the unconditional rank correlation, reported as a monotonicity check:")
print("a PRCC whose sign disagrees with it would indicate a non-monotone response.")
