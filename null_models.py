"""
null_models.py

Compares the two-type temperature-structured model against a set of null models
on the CMEMS monthly climatology, as requested for the data section.

Models
  H   single-harmonic regression        a0 + a1 cos(2 pi t/T) + b1 sin(2 pi t/T)   (3 fitted)
  N1  one-type NPZD, thermal            single optimum + scale                     (2 fitted)
  NC  two-type NPZD, common optimum     one shared optimum + scale                 (2 fitted)
  N2  two-type NPZD, distinct optima    two optima + scale                         (3 fitted)
  NT  no thermal dependence, seasonal nutrient forcing  scale only                 (1 fitted)

All ODE models share the N-P-Z-D structure, reference rates, and the same prescribed
SST forcing; NT removes the thermal response (theta = 1) and instead drives the
nutrient input with the observed seasonal nitrate, testing whether the seasonal cycle
can be reproduced without any temperature dependence at all.

Residuals are expressed in observation units: the model is scaled to the data,
a = <Pmod, obs>/<Pmod, Pmod>, so that residual = a*Pmod - obs.

Reported: RSS, r, median absolute error, AICc, BIC, and leave-one-month-out
cross-validated RMSE (an out-of-sample measure).
"""
import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model

rows = list(csv.DictReader(open('station_climatology_point.csv')))
sst  = np.array([float(r['sst'])  for r in rows])
no3  = np.array([float(r['no3'])  for r in rows])
obs  = np.array([float(r['phyc']) for r in rows])
n = len(obs)

DC  = (np.arange(12)+0.5)*365/12.0
dcx = np.concatenate([[DC[-1]-365], DC, [DC[0]+365]])
def periodic(v):
    vx = np.concatenate([[v[-1]], v, [v[0]]])
    return lambda t: float(np.interp(t % 365.0, dcx, vx))
Tfun, Nfun = periodic(sst), periodic(no3)

P0 = model.make_p()

def rhs_factory(o1, o2, two=True, thermal=True, nitrate=False):
    def rhs(t, y):
        N,P1,P2,Z,D = [max(x,1e-12) for x in y]; T = Tfun(t)
        th1 = model.theta(T,o1,P0['w1'],P0['wskew1']) if thermal else 1.0
        th2 = model.theta(T,o2,P0['w2'],P0['wskew2']) if thermal else 1.0
        rh  = model.rho(T,P0['rho0'],P0['Q10'],P0['Tref'])
        I0  = P0['I0']*Nfun(t)/no3.mean() if nitrate else P0['I0']
        s=P0['switch']; a=P1**s; b=P2**s; tot=a+b+1e-12
        phi1, phi2 = (a/tot, b/tot) if two else (1.0, 0.0)
        gP1 = phi1*model.g(P1,P0['KP1']); gP2 = phi2*model.g(P2,P0['KP2'])
        gr1 = P0['mu1']*th1*model.f(N,P0['KN1'])*P1
        gr2 = (P0['mu2']*th2*model.f(N,P0['KN2'])*P2) if two else 0.0
        rm  = rh*D
        return [I0-P0['lN']*N-gr1-gr2+rm,
                gr1-P0['mZ']*gP1*Z-P0['d1']*P1,
                (gr2-P0['mZ']*gP2*Z-P0['d2']*P2) if two else -P2,
                P0['e']*P0['mZ']*(gP1+gP2)*Z-P0['dZ']*Z,
                P0['d1']*P1+P0['d2']*P2+(1-P0['e'])*P0['mZ']*(gP1+gP2)*Z+P0['dZ']*Z-rm-P0['lD']*D]
    return rhs

def monthly(o1, o2, two=True, thermal=True, nitrate=False, years=6):
    y0 = [1,0.5,0.5,0.4,2.0] if two else [1,1.0,0.0,0.4,2.0]
    sol = solve_ivp(rhs_factory(o1,o2,two,thermal,nitrate), [0,years*365], y0,
                    method='RK45', rtol=1e-6, atol=1e-8, max_step=5.0, dense_output=True)
    t0 = (years-1)*365.0; tt = np.linspace(t0, t0+365, 3660)
    Y = sol.sol(tt); Pm = Y[1]+Y[2] if two else Y[1]
    mon = np.floor(((tt-t0)/365.0)*12).astype(int) % 12
    return np.array([Pm[mon==m].mean() for m in range(12)])

# ---------- design matrices / model banks ----------
tgrid = (np.arange(12)+0.5)/12.0
HARM  = np.column_stack([np.ones(n), np.cos(2*np.pi*tgrid), np.sin(2*np.pi*tgrid)])

G1 = np.arange(13.0, 18.01, 0.25); G2 = np.arange(18.0, 23.01, 0.25)
GC = np.arange(13.0, 23.01, 0.25)

print("precomputing model banks ...")
BANK2 = np.array([[monthly(a,b) for b in G2] for a in G1])          # two-type, distinct
BANK1 = np.array([monthly(a, a, two=False) for a in GC])            # one-type
BANKC = np.array([monthly(a, a) for a in GC])                       # two-type, common optimum
NTHERM = monthly(0,0, thermal=False, nitrate=True)                  # no thermal, nitrate forced
print("done.\n")

def scale_fit(pred, y):
    a = np.dot(pred,y)/np.dot(pred,pred)
    return a, a*pred

def metrics(res, k, y):
    rss = np.sum(res**2); K = k+1
    aic = n*np.log(rss/n) + 2*K
    return dict(rss=rss, aicc=aic + 2*K*(K+1)/(n-K-1), bic=n*np.log(rss/n)+K*np.log(n),
                mae=np.median(np.abs(res)))

def fit_bank(bank, y, axes):
    """Best entry of a precomputed bank under least-squares scaling."""
    flat = bank.reshape(-1, n)
    a = (flat @ y)/np.sum(flat**2, axis=1)
    r = a[:,None]*flat - y[None,:]
    i = np.argmin(np.sum(r**2, axis=1))
    return flat[i], a[i]

def evaluate(name, kind, k):
    if kind == 'H':
        beta, *_ = np.linalg.lstsq(HARM, obs, rcond=None); pred = HARM @ beta
    elif kind == 'NT':
        _, pred = scale_fit(NTHERM, obs)
    else:
        bank = {'N2':BANK2, 'N1':BANK1, 'NC':BANKC}[kind]
        base, a = fit_bank(bank, obs, None); pred = a*base
    res = pred - obs
    m = metrics(res, k, obs); m['r'] = np.corrcoef(pred, obs)[0,1]
    # leave-one-month-out
    errs = []
    for j in range(n):
        keep = np.arange(n) != j
        if kind == 'H':
            b, *_ = np.linalg.lstsq(HARM[keep], obs[keep], rcond=None); p = HARM[j] @ b
        elif kind == 'NT':
            a = np.dot(NTHERM[keep],obs[keep])/np.dot(NTHERM[keep],NTHERM[keep]); p = a*NTHERM[j]
        else:
            bank = {'N2':BANK2, 'N1':BANK1, 'NC':BANKC}[kind]
            flat = bank.reshape(-1, n)[:, keep]
            aa = (flat @ obs[keep])/np.sum(flat**2, axis=1)
            rr = aa[:,None]*flat - obs[keep][None,:]
            i = np.argmin(np.sum(rr**2, axis=1))
            p = aa[i]*bank.reshape(-1, n)[i, j]
        errs.append(p - obs[j])
    m['loocv'] = np.sqrt(np.mean(np.array(errs)**2))
    m['name'] = name; m['k'] = k
    return m

MODELS = [("two-type, distinct optima", 'N2', 3),
          ("two-type, common optimum",  'NC', 2),
          ("one-type, thermal",         'N1', 2),
          ("single harmonic",           'H',  3),
          ("no thermal, nitrate-forced",'NT', 1)]

res = [evaluate(*m) for m in MODELS]
best = min(r['aicc'] for r in res)
print(f"{'model':30s} {'#fit':>5s} {'RSS':>9s} {'r':>7s} {'MAE':>8s} {'AICc':>9s} {'dAICc':>8s} {'LOOCV':>8s}")
print("-"*90)
for r in res:
    print(f"{r['name']:30s} {r['k']:5d} {r['rss']:9.4f} {r['r']:7.3f} {r['mae']:8.4f} "
          f"{r['aicc']:9.2f} {r['aicc']-best:8.2f} {r['loocv']:8.4f}")
print("\n(observation units; residual = scaled model - observed phytoplankton carbon)")
