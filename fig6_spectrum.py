import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import fsolve
import model
T=19.0
def spectrum(I0, seed):
    p=model.make_p(T=T,I0=I0); rhs=model.make_rhs(p)
    x=np.maximum(fsolve(lambda y: rhs(0.0,y), seed),1e-12)
    h=1e-6; J=np.zeros((5,5))
    for k in range(5):
        xp=x.copy(); xm=x.copy(); xp[k]+=h; xm[k]-=h
        J[:,k]=(np.array(rhs(0.0,list(xp)))-np.array(rhs(0.0,list(xm))))/(2*h)  # central
    return x, np.linalg.eigvals(J)
seed=np.array([0.845,0.814,1.914,1.991,7.972])
for I0 in (0.5,0.85):
    x,ev=spectrum(I0,seed)
    ev=ev[np.argsort(-ev.real)]
    print("\niota0=%.2f  eq P1+P2=%.3f Z=%.3f"%(I0,x[1]+x[2],x[3]))
    for e in ev:
        tag="complex pair" if abs(e.imag)>1e-6 else "real"
        print("   %+.4f %+.4fi   (%s)"%(e.real,e.imag,tag))
    ncomplex=sum(1 for e in ev if abs(e.imag)>1e-6)
    print("   => %s"%("FOCUS (has complex pair -> spirals)" if ncomplex>0 else "NODE (all real -> no spiral)"))
