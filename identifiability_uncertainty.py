"""
identifiability_uncertainty.py

Uncertainty quantification for the two fitted thermal optima, replacing the
collinearity index as the sole identifiability evidence.

The seasonal model output depends only on (Topt1, Topt2) and the prescribed SST
forcing, not on the observations: the data enter only through the closed-form
scale c and the residual sum of squares.  We therefore precompute the monthly
model output on a grid of (Topt1, Topt2) once, after which profiling and
resampling are inexpensive.

Reported:
  * point estimate by grid search on RSS
  * profile-likelihood confidence intervals (F threshold, n=12, p=3)
  * iid residual bootstrap
  * circular block bootstrap (block length 3), which respects the seasonal
    autocorrelation of the twelve consecutive climatological months
"""
import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
from scipy.stats import f as fdist
import model

rows = list(csv.DictReader(open('station_climatology_point.csv')))
sst  = np.array([float(r['sst'])  for r in rows])
phyc = np.array([float(r['phyc']) for r in rows])
n = len(phyc); p_fit = 3                      # Topt1, Topt2, scale c

DC = (np.arange(12)+0.5)*365/12.0
dcx = np.concatenate([[DC[-1]-365], DC, [DC[0]+365]])
sx  = np.concatenate([[sst[-1]], sst, [sst[0]]])
Tfun = lambda t: float(np.interp(t % 365.0, dcx, sx))

def seasonal_rhs(p):
    def rhs(t, y):
        N,P1,P2,Z,D = [max(x,1e-12) for x in y]; T = Tfun(t)
        th1=model.theta(T,p['Topt1'],p['w1'],p['wskew1']); th2=model.theta(T,p['Topt2'],p['w2'],p['wskew2'])
        rh=model.rho(T,p['rho0'],p['Q10'],p['Tref']); fN1=model.f(N,p['KN1']); fN2=model.f(N,p['KN2'])
        s=p['switch']; a=P1**s; b=P2**s; tot=a+b+1e-12; phi1=a/tot; phi2=b/tot
        gP1=phi1*model.g(P1,p['KP1']); gP2=phi2*model.g(P2,p['KP2'])
        gr1=p['mu1']*th1*fN1*P1; gr2=p['mu2']*th2*fN2*P2; rm=rh*D
        return [p['I0']-p['lN']*N-gr1-gr2+rm,
                gr1-p['mZ']*gP1*Z-p['d1']*P1,
                gr2-p['mZ']*gP2*Z-p['d2']*P2,
                p['e']*p['mZ']*(gP1+gP2)*Z-p['dZ']*Z,
                p['d1']*P1+p['d2']*P2+(1-p['e'])*p['mZ']*(gP1+gP2)*Z+p['dZ']*Z-rm-p['lD']*D]
    return rhs

def model_monthly(o1, o2, years=6):
    pp = model.make_p(Topt1=o1, Topt2=o2)
    sol = solve_ivp(seasonal_rhs(pp), [0, years*365], [1,0.5,0.5,0.4,2.0], method='RK45',
                    rtol=1e-6, atol=1e-8, max_step=5.0, dense_output=True)
    t0 = (years-1)*365.0; tt = np.linspace(t0, t0+365, 3660)
    P = sol.sol(tt)[1] + sol.sol(tt)[2]
    mon = np.floor(((tt-t0)/365.0)*12).astype(int) % 12
    return np.array([P[mon==m].mean() for m in range(12)])

# ---------------- precompute the model on a grid ----------------
G1 = np.arange(13.0, 18.001, 0.25)
G2 = np.arange(18.0, 23.001, 0.25)
PH = np.empty((len(G1), len(G2), 12))
for i, o1 in enumerate(G1):
    for j, o2 in enumerate(G2):
        PH[i, j] = model_monthly(o1, o2)
print(f"grid precomputed: {len(G1)} x {len(G2)} = {len(G1)*len(G2)} model runs")

def rss_surface(obs):
    """RSS over the whole grid for a given observation vector (scale profiled out)."""
    c = (PH @ obs) / np.sum(obs**2)                 # closed-form scale, shape (n1,n2)
    resid = PH - c[..., None] * obs[None, None, :]
    return np.sum(resid**2, axis=2)

def fit(obs):
    R = rss_surface(obs)
    i, j = np.unravel_index(np.argmin(R), R.shape)
    return G1[i], G2[j], R

o1h, o2h, R0 = fit(phyc)
rssmin = R0.min()
Ph_hat = PH[list(G1).index(o1h), list(G2).index(o2h)]
c_hat  = np.dot(Ph_hat, phyc)/np.sum(phyc**2)
r_hat  = np.corrcoef(Ph_hat, phyc)[0,1]
print(f"\npoint estimate: Topt1={o1h:.2f}  Topt2={o2h:.2f}   r={r_hat:.4f}  RSS={rssmin:.5f}")

# ---------------- profile-likelihood intervals ----------------
thr = rssmin * (1.0 + fdist.ppf(0.95, 1, n-p_fit)/(n-p_fit))
prof1 = R0.min(axis=1); prof2 = R0.min(axis=0)
in1 = G1[prof1 <= thr]; in2 = G2[prof2 <= thr]
print(f"\nprofile-likelihood 95% intervals (F threshold, RSS <= {thr:.5f}):")
print(f"  Topt1 in [{in1.min():.2f}, {in1.max():.2f}]"
      + ("   (touches grid edge)" if in1.min()<=G1[0] or in1.max()>=G1[-1] else ""))
print(f"  Topt2 in [{in2.min():.2f}, {in2.max():.2f}]"
      + ("   (touches grid edge)" if in2.min()<=G2[0] or in2.max()>=G2[-1] else ""))

# ---------------- bootstraps ----------------
resid = Ph_hat - c_hat*phyc
rng = np.random.default_rng(20260718)
B = 4000

def boot(kind, block=3):
    out = np.empty((B, 2))
    for b in range(B):
        if kind == 'iid':
            e = rng.choice(resid, size=n, replace=True)
        else:                                        # circular block bootstrap
            e = np.concatenate([np.roll(resid, -rng.integers(n))[:block]
                                for _ in range(int(np.ceil(n/block)))])[:n]
        synth = (Ph_hat - e)/c_hat                   # synthetic observations
        if np.sum(synth**2) <= 0 or np.any(~np.isfinite(synth)):
            out[b] = np.nan; continue
        a1, a2, _ = fit(synth)
        out[b] = (a1, a2)
    return out[~np.isnan(out).any(axis=1)]

for kind, label in [('iid','iid residual bootstrap'), ('block','circular block bootstrap (L=3)')]:
    Bo = boot(kind)
    q1 = np.percentile(Bo[:,0], [2.5, 97.5]); q2 = np.percentile(Bo[:,1], [2.5, 97.5])
    print(f"\n{label}, B={len(Bo)}:")
    print(f"  Topt1 95% CI [{q1[0]:.2f}, {q1[1]:.2f}]   median {np.median(Bo[:,0]):.2f}")
    print(f"  Topt2 95% CI [{q2[0]:.2f}, {q2[1]:.2f}]   median {np.median(Bo[:,1]):.2f}")
    print(f"  fraction of replicates at a grid edge: "
          f"Topt1 {np.mean((Bo[:,0]<=G1[0])|(Bo[:,0]>=G1[-1])):.3f}, "
          f"Topt2 {np.mean((Bo[:,1]<=G2[0])|(Bo[:,1]>=G2[-1])):.3f}")
