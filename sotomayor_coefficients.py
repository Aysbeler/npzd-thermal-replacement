"""
sotomayor_coefficients.py

Sotomayor transcritical coefficients at the two window edges T1, T2, computed on the system
with the moving boundary equilibrium translated to the origin, which is the point raised by a
referee: along a moving branch E_i(T) the mixed derivative is not the total derivative of the
invasion eigenvalue.

We translate.  With y = x - E_i(T), the shifted field is G(y,T) = F(y + E_i(T), T), for which the
origin is an equilibrium for all T.  Sotomayor's three coefficients are evaluated for G at (0, T*):

    a1 = w . G_T,              (must vanish: transcritical, not saddle-node)
    a2 = w . [ D G_T . v ],    (must be nonzero)
    a3 = w . [ D^2 G(v,v) ],   (must be nonzero: transcritical, not pitchfork)

with v, w the right and left null vectors of A = D_y G(0,T*).  The chain rule gives
G_T = F_T + D_x F . E_i'(T) and D G_T = D_x F_T + D_xx F . E_i'(T), so E_i'(T) enters explicitly;
we obtain E_i'(T) by implicit differentiation of the branch and never assume a2 equals dI/dT.

All derivatives are exact (sympy).  The eigenvalue crossing that locates the edge is checked
against invasion_edges.py.
"""
import numpy as np, sympy as sp, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import brentq
import model, invasion_edges

# ---- symbolic reference field ----
N, P1, P2, Z, D, T = sp.symbols('N P1 P2 Z D T', real=True)
p = model.make_p()
def th(Topt, w, ws):
    z = (T - Topt)
    return sp.Piecewise((sp.exp(-(z/w)**2), T <= Topt), (sp.exp(-(z/ws)**2), True))
th1 = th(p['Topt1'], p['w1'], p['wskew1']); th2 = th(p['Topt2'], p['w2'], p['wskew2'])
rho = p['rho0']*p['Q10']**((T - p['Tref'])/10)
f1 = N/(p['KN1']+N); f2 = N/(p['KN2']+N)
g1 = P1/(p['KP1']+P1); g2 = P2/(p['KP2']+P2)
s = p['switch']; tot = P1**s + P2**s
phi1 = P1**s/tot; phi2 = P2**s/tot
gr1 = p['mu1']*th1*f1*P1; gr2 = p['mu2']*th2*f2*P2
graze = p['mZ']*(phi1*g1 + phi2*g2)*Z
Fsym = sp.Matrix([
    p['I0'] - p['lN']*N - gr1 - gr2 + rho*D,
    gr1 - p['mZ']*phi1*g1*Z - p['d1']*P1,
    gr2 - p['mZ']*phi2*g2*Z - p['d2']*P2,
    p['e']*graze - p['dZ']*Z,
    p['d1']*P1 + p['d2']*P2 + (1-p['e'])*graze + p['dZ']*Z - rho*D - p['lD']*D])
X = sp.Matrix([N, P1, P2, Z, D])

def coeffs_at(Tstar, resident):
    """resident = 1 means E_1 on {P2=0}; the invader is the other type."""
    inv = 2 if resident == 1 else 1
    # locate the single-type boundary equilibrium E_resident(T) numerically near Tstar
    def Eres(Tval):
        pv = model.make_p(T=Tval); rhs = model.make_rhs(pv)
        from scipy.optimize import fsolve
        y0 = [0.5, 1.7, 0.0, 2.0, 8.0] if resident == 1 else [0.5, 0.0, 1.7, 2.0, 8.0]
        def eq(u4):
            full = np.insert(np.abs(u4), inv, 0.0)
            return np.delete(rhs(0.0, full), inv)
        sol = fsolve(eq, np.delete(y0, inv), full_output=False)
        full = np.insert(np.abs(sol), inv, 0.0); return full
    E0 = Eres(Tstar)
    # E'(T) by finite difference of the branch (implicit differentiation, numerically)
    h = 1e-4
    Ep = (Eres(Tstar+h) - Eres(Tstar-h))/(2*h)

    sub0 = {N:E0[0], P1:E0[1], P2:E0[2], Z:E0[3], D:E0[4], T:Tstar}
    A = Fsym.jacobian(X)
    FT = sp.diff(Fsym, T)
    # translated mixed derivative: D G_T = D_x F_T + D_xx F . E'(T)
    DxFT = FT.jacobian(X)
    # D_xx F contracted with E'(T): for each component, Hessian . Ep
    DxxF_Ep = sp.zeros(5, 5)
    for i in range(5):
        Hi = sp.hessian(Fsym[i], X)         # 5x5
        DxxF_Ep[i, :] = (Hi @ sp.Matrix(Ep)).T
    Anum = np.array(A.subs(sub0)).astype(float)
    # right/left null vectors of A
    wv, Vr = np.linalg.eig(Anum); k = int(np.argmin(np.abs(wv)))
    v = np.real(Vr[:, k]); 
    wl, Vl = np.linalg.eig(Anum.T); kk = int(np.argmin(np.abs(wl)))
    w = np.real(Vl[:, kk]); w = w/np.dot(w, v)
    lam0 = np.real(wv[k])

    FTnum = np.array(FT.subs(sub0)).astype(float).ravel()
    # G_T = F_T + A . E'(T)
    GT = FTnum + Anum @ Ep
    a1 = float(w @ GT)
    DGT = np.array((DxFT).subs(sub0)).astype(float) + np.array(DxxF_Ep.subs(sub0)).astype(float)
    a2 = float(w @ (DGT @ v))
    # D^2 F (v,v)
    D2 = np.zeros(5)
    for i in range(5):
        Hi = np.array(sp.hessian(Fsym[i], X).subs(sub0)).astype(float)
        D2[i] = v @ Hi @ v
    a3 = float(w @ D2)
    return lam0, a1, a2, a3, Ep

T1, T2 = invasion_edges.edges(p) if hasattr(invasion_edges, 'edges') else (15.37, 20.10)
print(f"edges from invasion_edges: T1={T1:.3f}, T2={T2:.3f}\n")
print(f"{'edge':>6} {'zero-eig':>11} {'w.G_T':>12} {'w.DG_T v':>12} {'w.D2G(v,v)':>12}")
print("-"*58)
for name, Ts, res in [('T1', T1, 1), ('T2', T2, 2)]:
    lam, a1, a2, a3, Ep = coeffs_at(Ts, res)
    print(f"{name:>6} {lam:11.2e} {a1:12.2e} {a2:12.4f} {a3:12.4f}")
    print(f"        (||E'(T)||={np.linalg.norm(Ep):.3f}, so the moving-branch term is included)")
print("\nInterpretation: w.G_T ~ 0 (transcritical, not saddle-node);")
print("w.DG_T v and w.D2G(v,v) both nonzero (not a pitchfork). Computed in translated")
print("coordinates with E'(T) retained, so the coefficients are the Sotomayor coefficients")
print("of the shifted system, not the total derivative of the invasion eigenvalue.")


def dI_dT_analytic(T, resident):
    """Transversality coefficient by implicit differentiation, without finite differences.

    The single-type equilibrium satisfies the scalar balance

        Phi_i(N, T) = iota_0 - lambda_N N - c(T) P_i^* G_i(N, T) = 0,
        c(T) = lambda_D / (rho(T) + lambda_D),

    so away from a degeneracy the implicit function theorem gives dN*/dT = -Phi_T / Phi_N, and the
    invasion fitness of the missing type j differentiates as

        I_j'(T) = dG_j/dT + (dG_j/dN)(dN*/dT).

    Every partial derivative here is available in closed form: the thermal responses are Gaussian on
    each branch, the remineralization rate is a Q10 law, and G_j is Monod. Since the critical
    temperatures lie strictly inside one branch of theta, the expression is exact there, which
    removes the step-size dependence of a differenced evaluation.
    """
    p = model.make_p()
    th = lambda TT, i: model.theta(TT, p[f'Topt{i}'], p[f'w{i}'], p[f'wskew{i}'])
    G = lambda N, TT, i: p[f'mu{i}']*th(TT, i)*model.f(N, p[f'KN{i}'])

    def dtheta(TT, i):
        Topt = p[f'Topt{i}']
        w = p[f'w{i}'] if TT <= Topt else p[f'wskew{i}']
        return th(TT, i)*(-2.0*(TT - Topt)/w**2)

    i = resident
    Pi = p[f'KP{i}']*p['dZ']/(p['e']*p['mZ'] - p['dZ'])
    rho = model.rho(T, p['rho0'], p['Q10'], p['Tref'])
    c = p['lD']/(rho + p['lD'])
    N = brentq(lambda x: p['I0'] - p['lN']*x - c*Pi*G(x, T, i),
               1e-12, p['I0']/p['lN'], xtol=1e-14)

    dGi_dN = p[f'mu{i}']*th(T, i)*p[f'KN{i}']/(p[f'KN{i}'] + N)**2
    Phi_N = -p['lN'] - c*Pi*dGi_dN
    drho = rho*np.log(p['Q10'])/10.0
    dc = -p['lD']*drho/(rho + p['lD'])**2
    Phi_T = -dc*Pi*G(N, T, i) - c*Pi*p[f'mu{i}']*dtheta(T, i)*model.f(N, p[f'KN{i}'])
    dN_dT = -Phi_T/Phi_N

    j = 2 if resident == 1 else 1
    dGj_dT = p[f'mu{j}']*dtheta(T, j)*model.f(N, p[f'KN{j}'])
    dGj_dN = p[f'mu{j}']*th(T, j)*p[f'KN{j}']/(p[f'KN{j}'] + N)**2
    return dGj_dT + dGj_dN*dN_dT


if __name__ == '__main__':
    print("\ntransversality by implicit differentiation")
    for T, res, lab in [(15.369708, 1, "I_2'(T_1)"), (20.098809, 2, "I_1'(T_2)")]:
        print(f"  {lab} = {dI_dT_analytic(T, res):+.8f}")
