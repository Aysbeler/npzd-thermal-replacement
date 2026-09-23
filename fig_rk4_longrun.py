"""
fig_rk4_longrun.py  ->  fig_rk4_longrun.png

Time evolution of (N,P1,P2,Z,D) in the three regimes at iota0 = 0.5, with D on the right axis:
cold-type dominance (T = 13 C), coexistence (T = 19 C), warm-type dominance (T = 24 C).
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model

I0, TEND = 0.5, 600.0
CASES = [(13.0, 'cold-type dominance'), (19.0, 'coexistence'), (24.0, 'warm-type dominance')]
COL = {'N': '0.35', 'P1': '#2a78d6', 'P2': '#e34948', 'Z': '#199e70', 'D': '#BA7517'}

plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8,
                     'mathtext.fontset': 'cm', 'font.family': 'serif'})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True)

for k, (T, title) in enumerate(CASES):
    p = model.make_p(T=T, I0=I0)
    sol = solve_ivp(model.make_rhs(p), [0, TEND], [1.0, 0.8, 0.8, 0.4, 0.4],
                    method='RK45', rtol=1e-9, atol=1e-11, dense_output=True)
    t = np.linspace(0, TEND, 3000); Y = sol.sol(t)
    a = ax[k]
    a.plot(t, Y[0], color=COL['N'],  lw=1.6, label=r'$N$')
    a.plot(t, Y[1], color=COL['P1'], lw=2.0, label=r'$P_1$')
    a.plot(t, Y[2], color=COL['P2'], lw=2.0, label=r'$P_2$')
    a.plot(t, Y[3], color=COL['Z'],  lw=1.8, label=r'$Z$')
    a2 = a.twinx()
    a2.plot(t, Y[4], color=COL['D'], lw=1.4, ls='--', label=r'$D$ (right)')
    a2.tick_params(axis='y', colors=COL['D'], labelsize=9)
    a2.spines['top'].set_visible(False)
    if k == 2:
        a2.set_ylabel(r'detritus $D$', color=COL['D'])
    a.set_title(f'({"abc"[k]}) {title},  $T={T:.0f}^\\circ$C', loc='left', fontsize=10)
    a.set_xlabel('time (d)')
    a.spines['top'].set_visible(False)
    if k == 0: a.set_ylabel('concentration')
    print(f"T={T}: terminal P1={Y[1,-1]:.4f} P2={Y[2,-1]:.4f} Z={Y[3,-1]:.4f}")

h1, l1 = ax[0].get_legend_handles_labels()
fig.legend(h1 + [plt.Line2D([0],[0], color=COL['D'], lw=1.4, ls='--')],
           l1 + [r'$D$ (right axis)'], frameon=False, fontsize=9, ncol=5,
           loc='lower center', bbox_to_anchor=(0.5, -0.06))
fig.subplots_adjust(bottom=0.30, top=0.90, wspace=0.34)
plt.savefig('fig_rk4_longrun.png', dpi=200, bbox_inches='tight')
print('wrote fig_rk4_longrun.png')
