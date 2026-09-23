"""
fig13_modelselection.py  ->  fig13_modelselection.png

Two-panel diagnostic comparison of the two-type model against the one-type baseline
and a descriptive single-harmonic regression, on the CMEMS monthly climatology.

Every model is fitted at its OWN least-squares optimum and then scaled to the data,
a = <Pmod,obs>/<Pmod,Pmod>, so residuals are in observation units.  The one-type
baseline is therefore shown at its best achievable fit rather than at a suboptimal
optimum; this is why it appears nearly flat, which is the temperature-independent
single-type plateau of the equilibrium analysis.

(a) monthly cycle: observations, two-type, one-type, harmonic
(b) modelled vs observed, with the 1:1 line
"""
import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import model

rows = list(csv.DictReader(open('station_climatology_point.csv')))
sst  = np.array([float(r['sst'])  for r in rows])
obs  = np.array([float(r['phyc']) for r in rows])
n = 12

DC  = (np.arange(12)+0.5)*365/12.0
dcx = np.concatenate([[DC[-1]-365], DC, [DC[0]+365]])
sx  = np.concatenate([[sst[-1]], sst, [sst[0]]])
Tfun = lambda t: float(np.interp(t % 365.0, dcx, sx))
P0 = model.make_p()

def rhs_factory(o1, o2, two=True):
    def rhs(t, y):
        N,P1,P2,Z,D = [max(x,1e-12) for x in y]; T = Tfun(t)
        th1 = model.theta(T,o1,P0['w1'],P0['wskew1']); th2 = model.theta(T,o2,P0['w2'],P0['wskew2'])
        rh  = model.rho(T,P0['rho0'],P0['Q10'],P0['Tref'])
        s=P0['switch']; a=P1**s; b=P2**s; tot=a+b+1e-12
        phi1, phi2 = (a/tot, b/tot) if two else (1.0, 0.0)
        gP1 = phi1*model.g(P1,P0['KP1']); gP2 = phi2*model.g(P2,P0['KP2'])
        gr1 = P0['mu1']*th1*model.f(N,P0['KN1'])*P1
        gr2 = (P0['mu2']*th2*model.f(N,P0['KN2'])*P2) if two else 0.0
        rm  = rh*D
        return [P0['I0']-P0['lN']*N-gr1-gr2+rm,
                gr1-P0['mZ']*gP1*Z-P0['d1']*P1,
                (gr2-P0['mZ']*gP2*Z-P0['d2']*P2) if two else -P2,
                P0['e']*P0['mZ']*(gP1+gP2)*Z-P0['dZ']*Z,
                P0['d1']*P1+P0['d2']*P2+(1-P0['e'])*P0['mZ']*(gP1+gP2)*Z+P0['dZ']*Z-rm-P0['lD']*D]
    return rhs

def monthly(o1, o2, two=True, years=6):
    y0 = [1,0.5,0.5,0.4,2.0] if two else [1,1.0,0.0,0.4,2.0]
    sol = solve_ivp(rhs_factory(o1,o2,two), [0,years*365], y0, method='RK45',
                    rtol=1e-6, atol=1e-8, max_step=5.0, dense_output=True)
    t0 = (years-1)*365.0; tt = np.linspace(t0, t0+365, 3660)
    Y = sol.sol(tt); Pm = Y[1]+Y[2] if two else Y[1]
    mon = np.floor(((tt-t0)/365.0)*12).astype(int) % 12
    return np.array([Pm[mon==m].mean() for m in range(12)])

def scaled(Pm):
    a = np.dot(Pm, obs)/np.dot(Pm, Pm)
    return a*Pm

def rss_of(Pm):
    return np.sum((scaled(Pm)-obs)**2)

# ---- fit each model at its own optimum ----
best2 = None
for o1 in np.arange(14.0, 17.51, 0.1):
    for o2 in np.arange(18.0, 21.51, 0.1):
        r = rss_of(monthly(o1, o2))
        if best2 is None or r < best2[0]: best2 = (r, o1, o2)
best1 = None
for o in np.arange(13.0, 23.01, 0.1):
    r = rss_of(monthly(o, o, two=False))
    if best1 is None or r < best1[0]: best1 = (r, o)

M2 = scaled(monthly(best2[1], best2[2]))
M1 = scaled(monthly(best1[1], best1[1], two=False))
tg = (np.arange(12)+0.5)/12.0
H  = np.column_stack([np.ones(n), np.cos(2*np.pi*tg), np.sin(2*np.pi*tg)])
MH = H @ np.linalg.lstsq(H, obs, rcond=None)[0]

def rep(tag, M):
    print(f"  {tag:26s} RSS={np.sum((M-obs)**2):.4f}  r={np.corrcoef(M,obs)[0,1]:.3f}  "
          f"range={M.max()/M.min():.2f}x")
print(f"two-type optimum  Topt=({best2[1]:.2f},{best2[2]:.2f})")
print(f"one-type optimum  Topt={best1[1]:.2f}")
rep("two-type", M2); rep("one-type", M1); rep("harmonic", MH)
print(f"  {'observations':26s} {'':21s} range={obs.max()/obs.min():.2f}x")

# ---------------- figure ----------------
plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8,
                     'mathtext.fontset': 'cm', 'font.family': 'serif'})
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9))
mn = np.arange(1, 13)

ax[0].plot(mn, obs, 'o', color='0.15', ms=5, label='CMEMS reanalysis')
ax[0].plot(mn, M2, '-',  color='#1a9850', lw=2.2, label='two-type (distinct optima)')
ax[0].plot(mn, M1, '--', color='0.55',   lw=1.8, label='one-type (own optimum)')
ax[0].plot(mn, MH, ':',  color='#2a78d6', lw=1.8, label='single harmonic')
ax[0].set_xlabel('month'); ax[0].set_ylabel('phytoplankton carbon (mmol C m$^{-3}$)')
ax[0].set_xticks(mn); ax[0].set_xlim(0.5, 12.5)
ax[0].set_title('(a)', loc='left', fontsize=11)
ax[0].spines['top'].set_visible(False); ax[0].spines['right'].set_visible(False)

lim = [min(obs.min(), M1.min(), M2.min(), MH.min())*0.93,
       max(obs.max(), M1.max(), M2.max(), MH.max())*1.05]
ax[1].plot(lim, lim, '-', color='0.75', lw=1.0, zorder=0)
ax[1].plot(obs, M2, 'o', color='#1a9850', ms=6, label='two-type')
ax[1].plot(obs, M1, 's', color='0.55',   ms=5, label='one-type')
ax[1].plot(obs, MH, '^', color='#2a78d6', ms=5, label='harmonic')
ax[1].set_xlabel('observed'); ax[1].set_ylabel('modelled')
ax[1].set_xlim(lim); ax[1].set_ylim(lim); ax[1].set_aspect('equal')
ax[1].set_title('(b)', loc='left', fontsize=11)
ax[1].spines['top'].set_visible(False); ax[1].spines['right'].set_visible(False)

# legends strictly below the axes so they cannot overlap the data
ax[0].legend(frameon=False, fontsize=9, ncol=2, loc='upper center',
             bbox_to_anchor=(0.5, -0.22), columnspacing=1.4, handlelength=1.8)
ax[1].legend(frameon=False, fontsize=9, ncol=3, loc='upper center',
             bbox_to_anchor=(0.5, -0.22), columnspacing=1.4, handlelength=1.6)

fig.subplots_adjust(bottom=0.34, top=0.92, wspace=0.32)
plt.savefig('fig13_modelselection.png', dpi=200, bbox_inches='tight')
print('\nwrote fig13_modelselection.png')
