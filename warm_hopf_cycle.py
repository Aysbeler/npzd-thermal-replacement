"""Warming Hopf supercriticality check: a stable, small-amplitude limit cycle
emerges on the {P1=0} face just above T^H ~ 27.05 C (i0=0.5). Amplitude should
scale ~ sqrt(T - T^H), the signature of a supercritical Hopf."""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model
p=model.make_p()
def face_rhs(t,y,T,I0=0.5):
    N,P2,Z,D=[max(x,1e-12) for x in y]
    th2=model.theta(T,25.0,7.0,4.0); rh=model.rho(T,0.05,2.0,20.0)
    fN2=N/(p['KN2']+N); gP2=P2/(p['KP2']+P2)
    gr2=p['mu2']*th2*fN2*P2; rm=rh*D
    return [I0-p['lN']*N-gr2+rm, gr2-p['mZ']*gP2*Z-p['d2']*P2,
            p['e']*p['mZ']*gP2*Z-p['dZ']*Z, p['d2']*P2+(1-p['e'])*p['mZ']*gP2*Z+p['dZ']*Z-rm-p['lD']*D]
print("T^H ~ 27.05 ; amplitude of emergent P2 cycle just above threshold:")
for T in [27.0,27.2,27.6,28.0,29.0]:
    s=solve_ivp(lambda t,y:face_rhs(t,y,T),[0,30000],[1.1,1.667,3.2,7.4],
                method='RK45',rtol=1e-10,atol=1e-12,t_eval=np.linspace(26000,30000,8000))
    amp=s.y[1].max()-s.y[1].min()
    print("  T=%.2f: P2 cycle amplitude=%.4f  sqrt(T-27.05)=%.3f  ratio=%.3f"%(
        T,amp,np.sqrt(max(T-27.05,0)),amp/np.sqrt(max(T-27.05,1e-9)) if T>27.05 else 0))
