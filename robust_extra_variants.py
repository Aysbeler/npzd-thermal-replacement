"""
robust_extra_variants.py

Computes three additional structural-robustness variants for the coexistence-window
table (tab:robstruct), using the SAME protocol as Section 7:
  - scan T in [13,24] C, step 0.5, at iota0=0.5
  - integrate to t=6000 with RK45
  - a type is 'present' if its terminal value > 1e-2
  - window = [first T with both present, last T with both present]
  - oscillation amplitude = peak-to-peak P2 at (T,iota0)=(19,1.3), long-time segment

We FIRST reproduce the reference row ([15.5,20.0], amp 17.96) to validate the protocol,
then compute:
  (A) epsilon-background grazing  phi_i = (eps+Pi^s)/(2 eps + P1^s + P2^s)   [removes rare-prey refuge]
  (B) plankton export             extra -lP*Pi (i=1,2) and -lZ*Z
  (C) T-dependent grazing/mort.   mZ(T)=mZ*Qz^((T-Tref)/10), dZ(T)=dZ*Qz^((T-Tref)/10), Qz=2
"""
import numpy as np
from scipy.integrate import solve_ivp
exec(open('model.py').read().split("def run(")[0])   # theta, rho, f, g, make_p

def build_rhs(variant='ref', eps=0.1, lP=0.05, lZ=0.05, Qz=2.0):
    def rhs(t, y, T, I0):
        N,P1,P2,Z,D = [max(x,1e-12) for x in y]
        p = make_p(T=T, I0=I0)
        th1=theta(T,p['Topt1'],p['w1'],p['wskew1']); th2=theta(T,p['Topt2'],p['w2'],p['wskew2'])
        rh=rho(T,p['rho0'],p['Q10'],p['Tref']); fN1=f(N,p['KN1']); fN2=f(N,p['KN2'])
        s=p['switch']; w1s=P1**s; w2s=P2**s
        if variant=='eps':
            tot=2*eps+w1s+w2s; phi1=(eps+w1s)/tot; phi2=(eps+w2s)/tot
        else:
            tot=w1s+w2s+1e-12; phi1=w1s/tot; phi2=w2s/tot
        mZ=p['mZ']; dZ=p['dZ']
        if variant=='Tgraz':
            fac=Qz**((T-p['Tref'])/10.0); mZ=mZ*fac; dZ=dZ*fac
        gP1=phi1*g(P1,p['KP1']); gP2=phi2*g(P2,p['KP2'])
        gr1=p['mu1']*th1*fN1*P1; gr2=p['mu2']*th2*fN2*P2; remin=rh*D
        dN=I0-p['lN']*N-gr1-gr2+remin
        dP1=gr1-mZ*gP1*Z-p['d1']*P1
        dP2=gr2-mZ*gP2*Z-p['d2']*P2
        dZv=p['e']*mZ*(gP1+gP2)*Z-dZ*Z
        dD=p['d1']*P1+p['d2']*P2+(1-p['e'])*mZ*(gP1+gP2)*Z+dZ*Z-remin-p['lD']*D
        if variant=='export':
            dP1-=lP*P1; dP2-=lP*P2; dZv-=lZ*Z
        return [dN,dP1,dP2,dZv,dD]
    return rhs

def window(variant, **kw):
    rhs=build_rhs(variant, **kw)
    Ts=np.arange(13.0,24.01,0.5); present=[]
    for T in Ts:
        sol=solve_ivp(lambda t,y: rhs(t,y,T,0.5),[0,6000],[1.0,0.8,0.8,0.4,0.4],
                      method='RK45',rtol=1e-8,atol=1e-10)
        P1e,P2e=sol.y[1,-1],sol.y[2,-1]
        present.append((P1e>1e-2 and P2e>1e-2))
    coex=[T for T,pr in zip(Ts,present) if pr]
    return (coex[0],coex[-1]) if coex else None

def osc_amp(variant, **kw):
    rhs=build_rhs(variant, **kw)
    sol=solve_ivp(lambda t,y: rhs(t,y,19.0,1.3),[0,12000],[1.0,0.8,0.8,0.4,0.4],
                  method='RK45',rtol=1e-9,atol=1e-11,t_eval=np.linspace(9000,12000,4000))
    P2=sol.y[2]; return P2.max()-P2.min()

print("VALIDATION - reference row (target: [15.5,20.0], amp 17.96):")
w=window('ref'); a=osc_amp('ref')
print(f"  window={w}  amp={a:.2f}")
print()
print("KADEME-3 VARIANTS:")
for name,var,kw in [("epsilon-background grazing (eps=0.1)",'eps',dict(eps=0.1)),
                    ("plankton export (lP=lZ=0.05)",'export',dict(lP=0.05,lZ=0.05)),
                    ("T-dependent grazing/mortality (Qz=2)",'Tgraz',dict(Qz=2.0))]:
    w=window(var,**kw); a=osc_amp(var,**kw)
    ws=f"[{w[0]:.1f},{w[1]:.1f}]" if w else "collapsed"
    print(f"  {name:40s} window={ws}  amp={a:.2f}")
