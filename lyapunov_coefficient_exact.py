"""
First Lyapunov coefficient l1 of the two Hopf bifurcations, via Kuznetsov's
projection formula with EXACT (symbolic) multilinear forms B, C -- no finite
differencing of third derivatives. Cross-checked against the emergent stable
limit cycle (branch continuation) reported in the main text.

Results:
  enrichment Hopf (E*, T=19, i0^H=0.753): l1 = -0.129  (supercritical)
  warming    Hopf (E2, i0=0.5, T^H=27.05): l1 = -0.508  (supercritical)
Both negative -> supercritical, consistent with the stable small-amplitude
cycles that emerge past each threshold.
"""
import numpy as np, sympy as sp, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import fsolve, brentq
import model

def kuznetsov_l1(sym_state, F, x_eq):
    """Kuznetsov l1 with exact B,C at equilibrium x_eq (numpy)."""
    n=len(sym_state); y=sym_state
    A=np.array(sp.lambdify(y,sp.Matrix(F).jacobian(y),'numpy')(*x_eq),float)
    w,V=np.linalg.eig(A); k=np.argmin(np.abs(w.real)); lam=w[k]; om=abs(lam.imag); q=V[:,k]
    wL,VL=np.linalg.eig(A.T); kL=np.argmin(np.abs(wL-np.conj(lam))); pv=VL[:,kL]
    q=q/np.sqrt(np.vdot(q,q)); pv=pv/np.conj(np.vdot(pv,q))       # <p,q>=1
    Bt=np.zeros((n,n,n)); Ct=np.zeros((n,n,n,n)); subs={y[i]:x_eq[i] for i in range(n)}
    for i in range(n):
        Bt[i]=np.array(sp.lambdify(y,sp.hessian(F[i],y),'numpy')(*x_eq),float)
        for j in range(n):
            for kk in range(n):
                d3=sp.diff(F[i],y[j],y[kk])
                for l in range(n): Ct[i,j,kk,l]=float(sp.diff(d3,y[l]).subs(subs))
    B=lambda u,v:np.array([u@Bt[i]@v for i in range(n)])
    C=lambda u,v,w_:np.array([sum(Ct[i,a,b,c]*u[a]*v[b]*w_[c] for a in range(n) for b in range(n) for c in range(n)) for i in range(n)])
    I=np.eye(n); a1=np.linalg.solve(A,B(q,np.conj(q))); a2=np.linalg.solve(2j*om*I-A,B(q,q))
    term=C(q,q,np.conj(q))-2*B(q,a1)+B(np.conj(q),a2)
    return (1/(2*om))*np.real(np.vdot(pv,term)), om

# ---- enrichment Hopf: full 5D system, T=19 ----
def F5(p):
    N,P1,P2,Z,D=sp.symbols('N P1 P2 Z D',real=True); y=[N,P1,P2,Z,D]
    th1=float(model.theta(p['T'],p['Topt1'],p['w1'],p['wskew1'])); th2=float(model.theta(p['T'],p['Topt2'],p['w2'],p['wskew2']))
    rh=float(model.rho(p['T'],p['rho0'],p['Q10'],p['Tref'])); s=p['switch']
    w1s=P1**s; w2s=P2**s; tot=w1s+w2s; phi1=w1s/tot; phi2=w2s/tot
    gP1=phi1*P1/(p['KP1']+P1); gP2=phi2*P2/(p['KP2']+P2)
    gr1=p['mu1']*th1*(N/(p['KN1']+N))*P1; gr2=p['mu2']*th2*(N/(p['KN2']+N))*P2; rm=rh*D
    F=[p['I0']-p['lN']*N-gr1-gr2+rm, gr1-p['mZ']*gP1*Z-p['d1']*P1, gr2-p['mZ']*gP2*Z-p['d2']*P2,
       p['e']*p['mZ']*(gP1+gP2)*Z-p['dZ']*Z, p['d1']*P1+p['d2']*P2+(1-p['e'])*p['mZ']*(gP1+gP2)*Z+p['dZ']*Z-rm-p['lD']*D]
    return y,F
def re5(I0):
    p=model.make_p(T=19.0,I0=I0); y,F=F5(p); Fl=sp.lambdify(y,F,'numpy')
    xs=fsolve(lambda v:np.array(Fl(*v),float),[0.8,0.8,1.9,2.0,8.0])
    return max(np.linalg.eigvals(np.array(sp.lambdify(y,sp.Matrix(F).jacobian(y),'numpy')(*xs),float)).real)
I0H=brentq(re5,0.6,0.9,xtol=1e-6); p=model.make_p(T=19.0,I0=I0H); y,F=F5(p)
xs=fsolve(lambda v:np.array(sp.lambdify(y,F,'numpy')(*v),float),[0.8,0.8,1.9,2.0,8.0])
l1e,ome=kuznetsov_l1(y,F,xs)
print("enrichment Hopf: i0^H=%.4f omega=%.4f l1=%.4f"%(I0H,ome,l1e))

# ---- warming Hopf: 4D {P1=0} face, i0=0.5 (critical pair lies in the face) ----
def Fface(T):
    N,P2,Z,D=sp.symbols('N P2 Z D',real=True); y=[N,P2,Z,D]; p=model.make_p(T=T,I0=0.5)
    th2=float(model.theta(T,p['Topt2'],p['w2'],p['wskew2'])); rh=float(model.rho(T,p['rho0'],p['Q10'],p['Tref']))
    gP2=P2/(p['KP2']+P2); gr2=p['mu2']*th2*(N/(p['KN2']+N))*P2; rm=rh*D
    F=[p['I0']-p['lN']*N-gr2+rm, gr2-p['mZ']*gP2*Z-p['d2']*P2, p['e']*p['mZ']*gP2*Z-p['dZ']*Z,
       p['d2']*P2+(1-p['e'])*p['mZ']*gP2*Z+p['dZ']*Z-rm-p['lD']*D]
    return y,F
def reF(T):
    y,F=Fface(T); xs=fsolve(lambda v:np.array(sp.lambdify(y,F,'numpy')(*v),float),[3.5,1.667,0.4,2.0])
    return max(np.linalg.eigvals(np.array(sp.lambdify(y,sp.Matrix(F).jacobian(y),'numpy')(*xs),float)).real)
Th=brentq(reF,27.0,28.0,xtol=1e-6); y,F=Fface(Th)
xs=fsolve(lambda v:np.array(sp.lambdify(y,F,'numpy')(*v),float),[3.5,1.667,0.4,2.0])
l1w,omw=kuznetsov_l1(y,F,xs)
print("warming Hopf:    T^H=%.4f omega=%.4f l1=%.4f"%(Th,omw,l1w))

# ---- VALIDATION: same routine on normal forms with known criticality ----
# supercritical: l1<0 ; subcritical: l1>0 (sign convention check)
xx,yy=sp.symbols('xx yy',real=True)
Fsup=[-yy-xx*(xx**2+yy**2), xx-yy*(xx**2+yy**2)]
Fsub=[-yy+xx*(xx**2+yy**2), xx+yy*(xx**2+yy**2)]
l1s,_=kuznetsov_l1([xx,yy],Fsup,[0.0,0.0])
l1b,_=kuznetsov_l1([xx,yy],Fsub,[0.0,0.0])
print("validation: supercritical normal form l1=%.4f (<0 expected); subcritical l1=%.4f (>0 expected)"%(l1s,l1b))
