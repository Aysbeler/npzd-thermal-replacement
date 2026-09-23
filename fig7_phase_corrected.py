"""
fig7_phase_corrected.py  ->  fig7_phase_corrected.pdf

Projected phase-space trajectories in the (P1+P2, Z) plane at T = 19 C, with arrows
indicating the direction of time: (a) below the enrichment Hopf (iota0 = 0.5), where the
trajectory spirals into the coexistence equilibrium, and (b) above it (iota0 = 0.85),
where it approaches a limit cycle.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model

T = 19.0
plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8,
                     'mathtext.fontset': 'cm', 'font.family': 'serif'})
fig, ax = plt.subplots(1, 2, figsize=(10, 4.0))

for k, (I0, lab) in enumerate([(0.50, r'below the Hopf, $\iota_0=0.5$'),
                               (0.85, r'above the Hopf, $\iota_0=0.85$')]):
    p = model.make_p(T=T, I0=I0)
    sol = solve_ivp(model.make_rhs(p), [0, 6000], [1.0, 0.8, 0.8, 0.4, 2.0],
                    method='RK45', rtol=1e-10, atol=1e-12, dense_output=True)
    t = np.linspace(0, 6000, 60000); Y = sol.sol(t)
    P, Z = Y[1]+Y[2], Y[3]
    a = ax[k]
    a.plot(P, Z, color='0.62', lw=0.7, zorder=1)                       # transient
    m = t > 0.9*t[-1]
    a.plot(P[m], Z[m], color='#2a78d6', lw=2.0, zorder=3)              # attractor
    for frac in (0.06, 0.16, 0.34):
        i = int(frac*len(t))
        a.annotate('', xy=(P[i+40], Z[i+40]), xytext=(P[i], Z[i]),
                   arrowprops=dict(arrowstyle='-|>', color='0.35', lw=0.9), zorder=4)
    a.plot(P[-1], Z[-1], 'o', color='#e34948', ms=5, zorder=5)
    amp = P[m].max()-P[m].min()
    print(f"iota0={I0}: attractor extent in P1+P2 = {amp:.4f}"
          + ("  (equilibrium)" if amp < 1e-3 else "  (limit cycle)"))
    a.set_xlabel(r'$P_1+P_2$'); a.set_title(f'({"ab"[k]}) {lab}', loc='left', fontsize=10)
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
ax[0].set_ylabel(r'zooplankton $Z$')
fig.subplots_adjust(bottom=0.16, top=0.90, wspace=0.24)
plt.savefig('fig7_phase_corrected.pdf', bbox_inches='tight')
print('wrote fig7_phase_corrected.pdf')
