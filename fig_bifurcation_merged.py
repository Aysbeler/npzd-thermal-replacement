"""
fig_bifurcation_merged.py  ->  fig_bifurcation_merged.png

Bifurcation structure in temperature at iota0 = 0.5.
(a) full sweep: cold dominance, coexistence window bounded by TC1 and TC2, warm dominance,
    and the warming Hopf.  (b) detail of the window and the P1 -> P2 handover.
Plotting follows the standard convention: stable equilibria solid, unstable equilibria dashed,
and periodic orbits as their min-max envelope. Past the warming Hopf the warm-type equilibrium E_2
is continued as an unstable branch by Newton iteration on the boundary balance rather than by
integration, which cannot reach it, and is drawn dashed alongside the envelope of the cycle that
replaces it.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model, mc_edges

I0 = 0.5
p0 = model.make_p(I0=I0)
T1, T2 = mc_edges.edges(p0)
print(f"transcritical edges: T1={T1:.2f}  T2={T2:.2f}")

def endstate(T, tend=20000, y0=(1.0, 0.8, 0.8, 0.4, 0.4)):
    p = model.make_p(T=T, I0=I0)
    sol = solve_ivp(model.make_rhs(p), [0, tend], list(y0), method='LSODA',
                    rtol=1e-10, atol=1e-12, dense_output=True)
    tt = np.linspace(0.75*tend, tend, 4000)
    Y = sol.sol(tt)
    return Y[1], Y[2]                       # P1(t), P2(t) on the late window

def E2_branch(T):
    """Warm-type boundary equilibrium E_2, stable or not, from the scalar balance."""
    from scipy.optimize import brentq
    p = model.make_p(T=T, I0=I0)
    P2 = p['KP2']*p['dZ']/(p['e']*p['mZ'] - p['dZ'])
    rho = model.rho(T, p['rho0'], p['Q10'], p['Tref'])
    c = p['lD']/(rho + p['lD'])
    th2 = model.theta(T, p['Topt2'], p['w2'], p['wskew2'])
    G2 = lambda N: p['mu2']*th2*model.f(N, p['KN2'])
    f = lambda N: I0 - p['lN']*N - c*P2*G2(N)
    hi = I0/p['lN']
    if not (f(1e-12) > 0 > f(hi)):
        return None, None
    N = brentq(f, 1e-12, hi, xtol=1e-14)
    Z = p['e']*P2/p['dZ']*(G2(N) - p['d2'])
    if Z <= 0:
        return None, None
    x = np.array([N, 0.0, P2, Z, P2*G2(N)/(rho + p['lD'])])
    rhs = model.make_rhs(p)
    J = np.empty((5, 5)); h = 1e-7
    for k in range(5):
        xp = x.copy(); xm = x.copy(); xp[k] += h; xm[k] -= h
        J[:, k] = (np.asarray(rhs(0, xp)) - np.asarray(rhs(0, xm)))/(2*h)
    th1 = model.theta(T, p['Topt1'], p['w1'], p['wskew1'])
    J[1, :] = 0.0; J[1, 1] = p['mu1']*th1*model.f(N, p['KN1']) - p['d1']
    return P2, np.linalg.eigvals(J).real.max() < 0


Ts = np.arange(12.0, 30.01, 0.25)
P1lo, P1hi, P2lo, P2hi = [], [], [], []
for T in Ts:
    a, b = endstate(T)
    P1lo.append(a.min()); P1hi.append(a.max())
    P2lo.append(b.min()); P2hi.append(b.max())
P1lo, P1hi = np.array(P1lo), np.array(P1hi)
P2lo, P2hi = np.array(P2lo), np.array(P2hi)
osc = (P1hi - P1lo > 1e-3) | (P2hi - P2lo > 1e-3)
TH = Ts[osc].min() if osc.any() else np.nan
print(f"oscillation onset on this grid: T ~ {TH}")

plt.rcParams.update({'font.size': 10, 'axes.linewidth': 0.8,
                     'mathtext.fontset': 'cm', 'font.family': 'serif'})
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))

def draw(a, Tsub, m):
    lo1, hi1, lo2, hi2, o = P1lo[m], P1hi[m], P2lo[m], P2hi[m], osc[m]
    for lo, hi, oo, col, lab in [(lo1, hi1, o, '#2a78d6', r'$P_1$ cold'),
                                 (lo2, hi2, o, '#e34948', r'$P_2$ warm')]:
        # equilibrium branch: solid where the attractor is a point
        eq = 0.5*(lo + hi)
        eq[oo] = np.nan
        a.plot(Tsub, eq, '-', color=col, lw=2.2, label=lab)
        # periodic branch: min and max of the cycle, with the envelope shaded
        cyc_lo = lo.copy(); cyc_lo[~oo] = np.nan
        cyc_hi = hi.copy(); cyc_hi[~oo] = np.nan
        a.fill_between(Tsub, lo, hi, where=oo, color=col, alpha=0.20, lw=0)
        a.plot(Tsub, cyc_lo, '-', color=col, lw=1.3, zorder=5)
        a.plot(Tsub, cyc_hi, '-', color=col, lw=1.3, zorder=5)
    # the warm-type equilibrium continued past the warming Hopf, where it is unstable and the
    # cycle has replaced it as the attractor; below T2 the interior branch is the attractor and
    # E_2 is not the relevant object, so we draw it only where it is both unstable and dominant
    unst_T, unst_P = [], []
    for TT in Tsub:
        if TT < T2:
            continue
        val, stable = E2_branch(TT)
        if val is not None and not bool(stable):
            unst_T.append(TT); unst_P.append(val)
    if unst_T:
        a.plot(unst_T, unst_P, '--', color='#5a1010', lw=1.8, zorder=6,
               dashes=(5, 3), label=r'$E_2$ unstable')
    for Tc, nm in [(T1, r'TC$_1$'), (T2, r'TC$_2$')]:
        a.axvline(Tc, color='0.45', lw=0.9, ls=':')
        a.annotate(nm, xy=(Tc, a.get_ylim()[1]), xytext=(0, -11),
                   textcoords='offset points', ha='center', fontsize=9, color='0.35')
    a.set_xlabel(r'temperature $T$ ($^\circ$C)')
    a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)

m = np.ones_like(Ts, bool)
ax[0].set_ylim(0, max(P1hi.max(), P2hi.max())*1.12)
draw(ax[0], Ts, m)
ax[0].set_ylabel('phytoplankton biomass')
ax[0].set_title('(a)', loc='left', fontsize=11)
if np.isfinite(TH):
    ax[0].axvline(TH, color='#7a5195', lw=0.9, ls='-.')
    ax[0].annotate('Hopf', xy=(TH, ax[0].get_ylim()[1]), xytext=(3, -11),
                   textcoords='offset points', fontsize=9, color='#7a5195')

md = (Ts >= T1-1.2) & (Ts <= T2+1.2)
ax[1].set_ylim(0, max(P1hi[md].max(), P2hi[md].max())*1.15)
draw(ax[1], Ts[md], md)
ax[1].set_title('(b) detail of the window', loc='left', fontsize=11)

ax[0].legend(frameon=False, fontsize=9, ncol=2, loc='upper center',
             bbox_to_anchor=(0.5, -0.22), columnspacing=1.6, handlelength=1.8)
fig.subplots_adjust(bottom=0.30, top=0.92, wspace=0.22)
plt.savefig('fig_bifurcation_merged.png', dpi=200, bbox_inches='tight')
print('wrote fig_bifurcation_merged.png')
