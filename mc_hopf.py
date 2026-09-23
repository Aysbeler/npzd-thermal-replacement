import numpy as np
from scipy.optimize import brentq
import model
from mc_edges import theta, rho, f, G, Pstar, Nstar, edges

def rhs_vec(y, T, p):
    r = model.make_rhs(dict(p, T=T)); return np.array(r(0.0, list(y)))

def E2_state(T, p):                     # warm-type boundary eq (P1=0)
    P2 = Pstar(2,p); N = Nstar(2,T,p)
    G2 = G(2,N,T,p); Z = p['e']*P2/p['dZ']*(G2-p['d2'])
    D  = P2*G2/(rho(T,p)+p['lD'])
    return np.array([N, 0.0, P2, Z, D])

def maxRe(T, p):
    y = E2_state(T,p); h=1e-6; J=np.zeros((5,5))
    for k in range(5):
        yp=y.copy(); ym=y.copy(); yp[k]+=h; ym[k]-=h
        J[:,k]=(rhs_vec(yp,T,p)-rhs_vec(ym,T,p))/(2*h)
    return np.max(np.real(np.linalg.eigvals(J)))

def warm_hopf(p, Tlo=None, Thi=32.0, step=0.25):
    _,T2 = edges(p)
    if not np.isfinite(T2): return np.nan
    Tlo = T2+0.5 if Tlo is None else Tlo
    Ts = np.arange(Tlo, Thi, step)
    mr = np.array([maxRe(T,p) for T in Ts])
    s = np.where((mr[:-1] <= 0) & (mr[1:] > 0))[0]     # - -> + (loses stability)
    if len(s)==0: return np.nan
    k=s[0]; return brentq(lambda T: maxRe(T,p), Ts[k], Ts[k+1])

if __name__ == '__main__':
    p = model.make_p()
    print("reference warm Hopf T^H = %.2f  (paper 27.0)" % warm_hopf(p))
    print("maxRe at 25,27,29: %.4f %.4f %.4f" % (maxRe(25,p),maxRe(27,p),maxRe(29,p)))
