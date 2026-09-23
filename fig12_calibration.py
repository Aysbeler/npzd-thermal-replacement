"""
fig12_calibration.py  ->  fig12_calibration.png

Confrontation with the CMEMS reanalysis climatology at the Izmir/eastern-Aegean cell.
(a) SST forcing with the fitted optima marked; observed nitrate shown for context only,
    it is not used as model forcing.  (b) Model P1+P2 against scaled phytoplankton carbon;
    the individual P1, P2 curves are model-inferred and not resolved by the reanalysis.
(c) Reanalysis chlorophyll climatology, with the interquartile range computed across the
    twenty-two available years rather than assumed.
"""
import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model

import pandas as pd
rows = list(csv.DictReader(open('station_climatology_point.csv')))
# chlorophyll quartiles across the 22 complete years, from the year-resolved extraction
_cy = pd.read_csv('station_chl_yearly.csv')
CHL_Q1 = _cy.groupby('month').chl.quantile(0.25).values
CHL_MED = _cy.groupby('month').chl.quantile(0.50).values
CHL_Q3 = _cy.groupby('month').chl.quantile(0.75).values
sst  = np.array([float(r['sst'])  for r in rows])
no3  = np.array([float(r['no3'])  for r in rows])
chl  = np.array([float(r['chl'])  for r in rows])
obs  = np.array([float(r['phyc']) for r in rows])
O1, O2 = 15.70, 19.20                      # fitted optima

DC  = (np.arange(12)+0.5)*365/12.0
dcx = np.concatenate([[DC[-1]-365], DC, [DC[0]+365]])
sx  = np.concatenate([[sst[-1]], sst, [sst[0]]])
Tfun = lambda t: float(np.interp(t % 365.0, dcx, sx))
P0 = model.make_p()

def rhs(t, y):
    N,P1,P2,Z,D = [max(x,1e-12) for x in y]; T = Tfun(t)
    th1 = model.theta(T,O1,P0['w1'],P0['wskew1']); th2 = model.theta(T,O2,P0['w2'],P0['wskew2'])
    rh  = model.rho(T,P0['rho0'],P0['Q10'],P0['Tref'])
    s=P0['switch']; a=P1**s; b=P2**s; tot=a+b+1e-12
    gP1=(a/tot)*model.g(P1,P0['KP1']); gP2=(b/tot)*model.g(P2,P0['KP2'])
    gr1=P0['mu1']*th1*model.f(N,P0['KN1'])*P1; gr2=P0['mu2']*th2*model.f(N,P0['KN2'])*P2
    return [P0['I0']-P0['lN']*N-gr1-gr2+rh*D, gr1-P0['mZ']*gP1*Z-P0['d1']*P1,
            gr2-P0['mZ']*gP2*Z-P0['d2']*P2, P0['e']*P0['mZ']*(gP1+gP2)*Z-P0['dZ']*Z,
            P0['d1']*P1+P0['d2']*P2+(1-P0['e'])*P0['mZ']*(gP1+gP2)*Z+P0['dZ']*Z-rh*D-P0['lD']*D]

YEARS = 6
sol = solve_ivp(rhs, [0, YEARS*365], [1,0.5,0.5,0.4,2.0], method='RK45',
                rtol=1e-8, atol=1e-10, max_step=5.0, dense_output=True)
t0 = (YEARS-1)*365.0; tt = np.linspace(t0, t0+365, 3660)
Y = sol.sol(tt); mon = np.floor(((tt-t0)/365.0)*12).astype(int) % 12
P1m = np.array([Y[1][mon==m].mean() for m in range(12)])
P2m = np.array([Y[2][mon==m].mean() for m in range(12)])
Pm  = P1m + P2m
c   = np.dot(Pm, obs)/np.dot(Pm, Pm)          # model -> observation units
r   = np.corrcoef(Pm, obs)[0,1]
print(f"c_obs (obs per model unit) = {c:.4f}   1/c = {1/c:.4f}   r = {r:.4f}")

M = np.arange(1,13)
plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8,
                     'mathtext.fontset': 'cm', 'font.family': 'serif'})
fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.7))

ax[0].plot(M, sst, '-o', color='#eda100', lw=2.0, ms=4, label='SST')
for O, col, lab in [(O1,'#2a78d6',r'$T_{\rm opt}^{(1)}$'), (O2,'#e34948',r'$T_{\rm opt}^{(2)}$')]:
    ax[0].axhline(O, color=col, ls='--', lw=1.1, label=lab)
axn = ax[0].twinx(); axn.plot(M, no3, ':', color='0.55', lw=1.4)
axn.set_ylabel(r'nitrate (context only)', color='0.45', fontsize=9)
axn.tick_params(axis='y', colors='0.45', labelsize=9); axn.spines['top'].set_visible(False)
ax[0].set_ylabel(r'SST ($^\circ$C)'); ax[0].set_title('(a) forcing', loc='left', fontsize=10)

ax[1].plot(M, obs, 'o', color='0.15', ms=5, label='CMEMS $P$ carbon')
ax[1].plot(M, c*Pm,  '-', color='#1a9850', lw=2.2, label=r'model $P_1+P_2$')
ax[1].plot(M, c*P1m, '--', color='#2a78d6', lw=1.5, label=r'model $P_1$')
ax[1].plot(M, c*P2m, '--', color='#e34948', lw=1.5, label=r'model $P_2$')
ax[1].set_ylabel(r'mmol\,C\,m$^{-3}$'.replace('\\,',' '))
ax[1].set_title(f'(b) fit, $r={r:.2f}$', loc='left', fontsize=10)

ax[2].plot(M, CHL_MED, '-s', color='#199e70', lw=2.0, ms=4)
ax[2].fill_between(M, CHL_Q1, CHL_Q3, color='#199e70', alpha=0.20, lw=0)
print(f"chl IQR/median, mean over months = {np.mean((CHL_Q3-CHL_Q1)/CHL_MED):.3f}")
ax[2].set_ylabel(r'chlorophyll'); ax[2].set_title('(c) chlorophyll climatology', loc='left', fontsize=10)

for a in ax:
    a.set_xlabel('month'); a.set_xticks(M); a.set_xlim(0.5,12.5)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
ax[0].legend(frameon=False, fontsize=8, ncol=3, loc='upper center',
             bbox_to_anchor=(0.5,-0.24), columnspacing=1.0, handlelength=1.4)
ax[1].legend(frameon=False, fontsize=8, ncol=2, loc='upper center',
             bbox_to_anchor=(0.5,-0.24), columnspacing=1.0, handlelength=1.4)
fig.subplots_adjust(bottom=0.34, top=0.90, wspace=0.38)
plt.savefig('fig12_calibration.png', dpi=200, bbox_inches='tight')
print('wrote fig12_calibration.png')
