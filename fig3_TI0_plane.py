"""
fig3_TI0_plane.py  ->  fig3_TI0_plane.png

Two-parameter (T, iota0) bifurcation diagram.  Solid: the transcritical boundaries T1, T2
enclosing the coexistence wedge, computed from the invasion criterion at each iota0.
Dashed: the enrichment Hopf boundary, located by bisection on the onset of sustained
oscillation of the coexistence state.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model, mc_edges

I0s = np.arange(0.20, 1.401, 0.05)
T1s, T2s = [], []
for I0 in I0s:
    try:
        a, b = mc_edges.edges(model.make_p(I0=I0))
    except Exception:
        a = b = np.nan
    T1s.append(a); T2s.append(b)
T1s, T2s = np.array(T1s), np.array(T2s)

def oscillates(T, I0, tend=6000):
    p = model.make_p(T=T, I0=I0)
    sol = solve_ivp(model.make_rhs(p), [0, tend], [1.0, 0.8, 0.8, 0.4, 0.4],
                    method='LSODA', rtol=1e-10, atol=1e-12, dense_output=True)
    tt = np.linspace(0.75*tend, tend, 1500); Y = sol.sol(tt)
    amp = max(Y[1].max()-Y[1].min(), Y[2].max()-Y[2].min())
    return amp > 1e-3

# Hopf boundary: for each T inside the wedge, bisect on iota0 for the onset of oscillation
Tgrid = np.arange(15.5, 20.51, 0.5)
I0H_of_T = []
for T in Tgrid:
    lo, hi = 0.3, 2.0
    if oscillates(T, lo) or not oscillates(T, hi):
        I0H_of_T.append(np.nan); continue
    for _ in range(12):
        mid = 0.5*(lo+hi)
        if oscillates(T, mid): hi = mid
        else:                  lo = mid
    I0H_of_T.append(0.5*(lo+hi))
I0H_of_T = np.array(I0H_of_T)
k = np.argmin(np.abs(Tgrid-19.0))
print(f"iota0^H at T=19: {I0H_of_T[k]:.3f}   (manuscript reports 0.753)")


plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8,
                     'mathtext.fontset': 'cm', 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ok = np.isfinite(T1s) & np.isfinite(T2s)
ax.fill_betweenx(I0s[ok], T1s[ok], T2s[ok], color='#eda100', alpha=0.16, lw=0)
ax.plot(T1s[ok], I0s[ok], '-',  color='#2a78d6', lw=2.2, label=r'$T_1$ (transcritical)')
ax.plot(T2s[ok], I0s[ok], '-',  color='#e34948', lw=2.2, label=r'$T_2$ (transcritical)')
ax.plot(T1s[ok], I0s[ok], 'o',  color='#2a78d6', ms=3)
ax.plot(T2s[ok], I0s[ok], 'o',  color='#e34948', ms=3)
okH = np.isfinite(I0H_of_T)
ax.plot(Tgrid[okH], I0H_of_T[okH], '--', color='#7a5195', lw=2.0,
        label=r'enrichment Hopf $\iota_0^{H}(T)$')
# codimension-two point located in Section 6.3 by solving the defining system
ax.plot([20.153834], [0.520075], marker='*', ms=15, color='k', mfc='#ffd24d',
        mew=1.1, zorder=6, label=r'transcritical--Hopf point')
ax.annotate('coexistence', xy=(0.5*(np.nanmin(T1s)+np.nanmax(T2s)), I0s.mean()),
            ha='center', fontsize=10, color='0.35')
ax.set_xlabel(r'temperature $T$ ($^\circ$C)')
ax.set_ylabel(r'nutrient input $\iota_0$')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(frameon=False, fontsize=9, ncol=3, loc='upper center',
          bbox_to_anchor=(0.5, -0.20), columnspacing=1.3, handlelength=1.6)
fig.subplots_adjust(bottom=0.30, top=0.95)
plt.savefig('fig3_TI0_plane.png', dpi=200, bbox_inches='tight')
print('wrote fig3_TI0_plane.png')
