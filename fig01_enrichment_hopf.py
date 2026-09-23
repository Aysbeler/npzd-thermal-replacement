import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8,
                     'mathtext.fontset': 'cm', 'font.family': 'serif'})
import warnings; warnings.filterwarnings('ignore')
import model                      # isolated namespace: model.make_p, model.make_rhs, model.C1 ...

C1, C2 = model.C1, model.C2
T   = 19.0
I0H = 0.753                        # enrichment Hopf (Prop 6.2 / Table 3)

# ---------- equilibrium branch by numerical continuation ----------
def eq_at(I0, seed):
    p = model.make_p(T=T, I0=I0, switch=2.0)
    rhs = model.make_rhs(p)
    x, info, ier, msg = fsolve(lambda y: rhs(0.0, y), seed, full_output=True)
    return np.maximum(x, 1e-12), ier

I0s  = np.linspace(0.35, 1.60, 260)
i050 = int(np.argmin(np.abs(I0s - 0.5)))
P1eq = np.full_like(I0s, np.nan)
P2eq = np.full_like(I0s, np.nan)
seed0 = np.array([0.845, 0.814, 1.914, 1.991, 7.972])   # known coexistence eq at I0=0.5, T=19

# march upward from 0.5
seed = seed0.copy()
for k in range(i050, len(I0s)):
    x, ier = eq_at(I0s[k], seed)
    if ier == 1: P1eq[k], P2eq[k] = x[1], x[2]; seed = x
# march downward from 0.5
seed = seed0.copy()
for k in range(i050 - 1, -1, -1):
    x, ier = eq_at(I0s[k], seed)
    if ier == 1: P1eq[k], P2eq[k] = x[1], x[2]; seed = x

xH, _ = eq_at(I0H, seed0)
yH = xH[2]                         # P2 branch value at the Hopf (~1.9)

# ---------- limit-cycle envelopes above the Hopf ----------
def envelope(I0):
    p = model.make_p(T=T, I0=I0, switch=2.0)
    rhs = model.make_rhs(p)
    sol = solve_ivp(rhs, [0, 5000], [1.0, 0.8, 0.8, 0.4, 0.4], method='RK45',
                    rtol=1e-7, atol=1e-9, t_eval=np.linspace(3500, 5000, 2500))
    P1, P2 = sol.y[1], sol.y[2]
    return P1.min(), P1.max(), P2.min(), P2.max()

I0osc = np.linspace(I0H+0.01, 1.60, 42)          # coarse grid for the (slow) envelope
env = np.array([envelope(I0) for I0 in I0osc])
P1lo, P1hi, P2lo, P2hi = env[:, 0], env[:, 1], env[:, 2], env[:, 3]

# ---------- plot ----------
below = I0s <= I0H
above = I0s >= I0H
fig, ax = plt.subplots(figsize=(7.4, 5.0))

ax.plot(I0s[below], P1eq[below], '-',  color=C1, lw=2.2, label=r'$P_1$')
ax.plot(I0s[below], P2eq[below], '-',  color=C2, lw=2.2, label=r'$P_2$')
ax.plot(I0s[above], P1eq[above], '--', color=C1, lw=1.8)
ax.plot(I0s[above], P2eq[above], '--', color=C2, lw=1.8)
ax.fill_between(I0osc, P1lo, P1hi, color=C1, alpha=0.18, linewidth=0)
ax.fill_between(I0osc, P2lo, P2hi, color=C2, alpha=0.18, linewidth=0)

ax.axvline(I0H, ls=':', color='0.35', lw=1.2)
ax.plot([I0H], [yH], 'o', color='black', ms=7, zorder=5)

# annotation -- REAL newline, corrected value 0.753
ax.annotate("supercritical Hopf\n" r"$\iota_0^{H}\approx 0.753,\ \ell_1<0$",
            xy=(I0H, yH), xytext=(0.98, 4.25),
            fontsize=10, ha='left', va='center',
            arrowprops=dict(arrowstyle='->', lw=1.1, color='black'))

ax.set_xlabel(r'Nutrient input $\iota_0$ (dimensionless)')
ax.set_ylabel('Phytoplankton biomass (dimensionless)')
ax.set_title(r'Enrichment Hopf bifurcation at $T=19^\circ$C')
ax.set_xlim(0.35, 1.60); ax.set_ylim(0, 5)
ax.legend(loc='upper left', frameon=False)

plt.tight_layout()
plt.savefig('hoph.png', dpi=200, bbox_inches='tight')
plt.savefig('fig3_enrichment_hopf.png', dpi=200, bbox_inches='tight')
print("OK  I0H=%.3f  yH(P2)=%.3f  osc_pts=%d  eqNaN=%d" %
      (I0H, yH, len(I0osc), int(np.isnan(P1eq).sum())))
print("eq@0.5: P1=%.3f P2=%.3f (paper 0.814/1.914)" %
      (P1eq[i050], P2eq[i050]))
print("P1 env [%.2f,%.2f]  P2 env [%.2f,%.2f]" %
      (P1lo.min(), P1hi.max(), P2lo.min(), P2hi.max()))
