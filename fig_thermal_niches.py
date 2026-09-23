"""
fig_thermal_niches.py  ->  fig_thermal_niches.png

Thermal performance curves theta_i(T) of the two phytoplankton types (eq:theta),
with the reference parameters of Table 1 (identical to model.py):
  Topt1=15, Topt2=25, w=7 (below optimum), w^-=4 (above optimum).
The shaded band spans the interval between the two thermal optima,
i.e. the origin of the competitive coexistence window.
"""
import numpy as np
import matplotlib.pyplot as plt

Topt1, Topt2 = 15.0, 25.0
w, wskew = 7.0, 4.0

def theta(T, Topt, w, wskew):
    T = np.asarray(T, float)
    return np.where(T <= Topt,
                    np.exp(-((T - Topt) / w) ** 2),
                    np.exp(-((T - Topt) / wskew) ** 2))

plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8,
                     'mathtext.fontset': 'cm', 'font.family': 'serif'})
fig, ax = plt.subplots(figsize=(6.2, 3.6))

T = np.linspace(10, 30, 500)
ax.fill_betweenx([0, 1.05], Topt1, Topt2, color='0.85', alpha=0.35, zorder=0)
ax.plot(T, theta(T, Topt1, w, wskew), color='#2a78d6', lw=2.4, label=r'$P_1$ cold-adapted')
ax.plot(T, theta(T, Topt2, w, wskew), color='#e34948', lw=2.4, label=r'$P_2$ warm-adapted')
ax.axvline(Topt1, color='#2a78d6', ls='--', lw=0.9, alpha=0.6)
ax.axvline(Topt2, color='#e34948', ls='--', lw=0.9, alpha=0.6)

ax.set_xlabel(r'temperature $T$ ($^\circ$C)')
ax.set_ylabel(r'thermal performance $\theta_i(T)$')
ax.set_ylim(0, 1.05); ax.set_xlim(10, 30)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

# legend below the axes -> never overlaps the curves
ax.legend(frameon=False, fontsize=10, ncol=2, loc='upper center',
          bbox_to_anchor=(0.5, -0.22), columnspacing=1.8, handlelength=1.6)

fig.subplots_adjust(bottom=0.30, top=0.95)
plt.savefig('fig_thermal_niches.png', dpi=200, bbox_inches='tight')
print('wrote fig_thermal_niches.png')
