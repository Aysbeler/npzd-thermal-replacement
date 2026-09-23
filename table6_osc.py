import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model
T=19.0; I0=1.3
def amp_from_rhs(rhs, y0=[1,0.8,0.8,0.4,2.0], Tend=8000):
    sol=solve_ivp(rhs,[0,Tend],y0,method='RK45',rtol=1e-8,atol=1e-10,
                  t_eval=np.linspace(Tend*0.75,Tend,4000))
    P2=sol.y[2]; return P2.max()-P2.min()

# reference
print("reference (skew, HollingII, linear, s=2):  amp=%.2f"%amp_from_rhs(model.make_rhs(model.make_p(T=T,I0=I0))))
# symmetric Gaussian theta: wskew=w
print("symmetric Gaussian theta:                   amp=%.2f"%amp_from_rhs(model.make_rhs(model.make_p(T=T,I0=I0,wskew1=7.0,wskew2=7.0))))
# no switching s=1
print("no switching (s=1):                         amp=%.2f"%amp_from_rhs(model.make_rhs(model.make_p(T=T,I0=I0,switch=1.0))))
# strong switching s=3
print("strong switching (s=3):                     amp=%.2f"%amp_from_rhs(model.make_rhs(model.make_p(T=T,I0=I0,switch=3.0))))

# Holling type III grazing: g(P)=P^2/(KP^2+P^2)  -- custom rhs
def rhs_hollingIII(p):
    def rhs(t,y):
        N,P1,P2,Z,D=[max(x,1e-12) for x in y]
        th1=model.theta(T,p['Topt1'],p['w1'],p['wskew1']); th2=model.theta(T,p['Topt2'],p['w2'],p['wskew2'])
        rh=model.rho(T,p['rho0'],p['Q10'],p['Tref']); fN1=model.f(N,p['KN1']); fN2=model.f(N,p['KN2'])
        s=p['switch']; a=P1**s; b=P2**s; tot=a+b+1e-12; phi1=a/tot; phi2=b/tot
        def g3(P,KP): return P**2/(KP**2+P**2)
        gP1=phi1*g3(P1,p['KP1']); gP2=phi2*g3(P2,p['KP2'])
        gr1=p['mu1']*th1*fN1*P1; gr2=p['mu2']*th2*fN2*P2; rm=rh*D
        return [p['I0']-p['lN']*N-gr1-gr2+rm, gr1-p['mZ']*gP1*Z-p['d1']*P1,
                gr2-p['mZ']*gP2*Z-p['d2']*P2, p['e']*p['mZ']*(gP1+gP2)*Z-p['dZ']*Z,
                p['d1']*P1+p['d2']*P2+(1-p['e'])*p['mZ']*(gP1+gP2)*Z+p['dZ']*Z-rm-p['lD']*D]
    return rhs
print("Holling type III grazing:                   amp=%.2f"%amp_from_rhs(rhs_hollingIII(model.make_p(T=T,I0=I0))))

# quadratic closure: dZ*Z^2
def rhs_quad(p):
    def rhs(t,y):
        N,P1,P2,Z,D=[max(x,1e-12) for x in y]
        th1=model.theta(T,p['Topt1'],p['w1'],p['wskew1']); th2=model.theta(T,p['Topt2'],p['w2'],p['wskew2'])
        rh=model.rho(T,p['rho0'],p['Q10'],p['Tref']); fN1=model.f(N,p['KN1']); fN2=model.f(N,p['KN2'])
        s=p['switch']; a=P1**s; b=P2**s; tot=a+b+1e-12; phi1=a/tot; phi2=b/tot
        gP1=phi1*model.g(P1,p['KP1']); gP2=phi2*model.g(P2,p['KP2'])
        gr1=p['mu1']*th1*fN1*P1; gr2=p['mu2']*th2*fN2*P2; rm=rh*D
        return [p['I0']-p['lN']*N-gr1-gr2+rm, gr1-p['mZ']*gP1*Z-p['d1']*P1,
                gr2-p['mZ']*gP2*Z-p['d2']*P2, p['e']*p['mZ']*(gP1+gP2)*Z-p['dZ']*Z**2,
                p['d1']*P1+p['d2']*P2+(1-p['e'])*p['mZ']*(gP1+gP2)*Z+p['dZ']*Z**2-rm-p['lD']*D]
    return rhs
print("quadratic closure (nuZ Z^2):                amp=%.2f"%amp_from_rhs(rhs_quad(model.make_p(T=T,I0=I0))))
