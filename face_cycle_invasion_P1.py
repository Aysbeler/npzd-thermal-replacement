import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model
def face_rhs_P1zero(t,y,p,T):  # {P1=0}: N-P2-Z-D, phi2=1
    N,P2,Z,D=[max(x,1e-12) for x in y]
    th2=model.theta(T,p['Topt2'],p['w2'],p['wskew2']); rh=model.rho(T,p['rho0'],p['Q10'],p['Tref'])
    fN2=model.f(N,p['KN2']); gP2=model.g(P2,p['KP2'])
    gr2=p['mu2']*th2*fN2*P2; rm=rh*D
    return [p['I0']-p['lN']*N-gr2+rm, gr2-p['mZ']*gP2*Z-p['d2']*P2,
            p['e']*p['mZ']*gP2*Z-p['dZ']*Z, p['d2']*P2+(1-p['e'])*p['mZ']*gP2*Z+p['dZ']*Z-rm-p['lD']*D]
def G1(N,T,p): return p['mu1']*model.theta(T,p['Topt1'],p['w1'],p['wskew1'])*model.f(N,p['KN1'])
print("=== {P1=0} yuzu (sicak tip yalniz): cevrim + <I1> (soguk tip invasion) ===")
for T in [15.5,16.0,17.0,17.7,18.0,19.0]:
    p=model.make_p(T=T,I0=0.5)
    s=solve_ivp(lambda t,y:face_rhs_P1zero(t,y,p,T),[0,20000],[1,0.5,0.4,2.0],
                method='RK45',rtol=1e-10,atol=1e-12,dense_output=True)
    tt=np.linspace(18000,20000,20000); Y=s.sol(tt)
    amp=Y[1].max()-Y[1].min(); I1=np.trapezoid(G1(Y[0],T,p)-p['d1'],tt)/(tt[-1]-tt[0])
    kind="CYCLE" if amp>1e-2 else "eq"
    print("  T=%4.1f: P2-amp=%.2f (%s)  <I1>=%+.4f -> %s"%(T,amp,kind,I1,"repelling" if I1>0 else "*** NOT ***"))
print("\n=== Ozet: pencere [15.37,20.10] icinde yuz-Hopf araligi ve transversal itim ===")
