import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import model

def face_rhs(N,P1,Z,D,p,T):
    th1=model.theta(T,p['Topt1'],p['w1'],p['wskew1']); rh=model.rho(T,p['rho0'],p['Q10'],p['Tref'])
    fN1=model.f(N,p['KN1']); gP1=model.g(P1,p['KP1'])
    gr1=p['mu1']*th1*fN1*P1; rm=rh*D
    return np.array([p['I0']-p['lN']*N-gr1+rm, gr1-p['mZ']*gP1*Z-p['d1']*P1,
                     p['e']*p['mZ']*gP1*Z-p['dZ']*Z,
                     p['d1']*P1+(1-p['e'])*p['mZ']*gP1*Z+p['dZ']*Z-rm-p['lD']*D])

def jac_face(y,p,T,h=1e-6):
    J=np.zeros((4,4)); f0=face_rhs(*y,p,T)
    for j in range(4):
        yp=y.copy(); yp[j]+=h; J[:,j]=(face_rhs(*yp,p,T)-f0)/h
    return J

for T in [19.0, 20.0, 17.7, 24.0]:
    p=model.make_p(T=T,I0=0.5)
    # find E1 face equilibrium
    sol=fsolve(lambda y: face_rhs(*y,p,T), [0.8,1.667,2.0,8.0], full_output=True)
    y=sol[0]; conv=sol[2]==1
    ev=np.linalg.eigvals(jac_face(y,p,T))
    lead=ev[np.argmax(ev.real)]
    # long integration to characterize attractor
    s=solve_ivp(lambda t,x: face_rhs(*x,p,T),[0,20000],[1,0.5,0.4,2.0],
                method='RK45',rtol=1e-10,atol=1e-12,t_eval=np.linspace(18000,20000,8000))
    P1seg=s.y[1]; amp=P1seg.max()-P1seg.min(); bounded=np.all(np.abs(s.y)<1e3)
    print("T=%4.1f: E1_face=(%.3f,%.3f,%.3f,%.3f) conv=%s"%(T,y[0],y[1],y[2],y[3],conv))
    print("   face eigenvalues real parts: %s"%np.round(sorted(ev.real,reverse=True),4))
    print("   leading=%.4f%+.4fi  -> %s ; long-run P1 amp=%.3f bounded=%s"%(
        lead.real,lead.imag,"UNSTABLE (Hopf?)" if lead.real>1e-4 else "stable",amp,bounded))
