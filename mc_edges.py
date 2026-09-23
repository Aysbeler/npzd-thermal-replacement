import numpy as np
from scipy.optimize import brentq, fsolve
import model

def theta(T, Topt, w, wskew):
    return np.exp(-((T-Topt)/w)**2) if T <= Topt else np.exp(-((T-Topt)/wskew)**2)
def rho(T, p): return p['rho0']*p['Q10']**((T-p['Tref'])/10.0)
def f(N, K):   return N/(K+N)

def G(i, N, T, p):
    if i == 1: return p['mu1']*theta(T,p['Topt1'],p['w1'],p['wskew1'])*f(N,p['KN1'])
    else:      return p['mu2']*theta(T,p['Topt2'],p['w2'],p['wskew2'])*f(N,p['KN2'])

def Pstar(i, p):                       # predator-controlled plateau (Prop 5.2)
    KP = p['KP1'] if i==1 else p['KP2']
    return KP*p['dZ']/(p['e']*p['mZ']-p['dZ'])

def Nstar(i, T, p):                    # scalar balance Phi(N)=0 (eq 26)
    Pi = Pstar(i, p); d = p['d1'] if i==1 else p['d2']
    def Phi(N): return p['I0'] - p['lN']*N - (p['lD']/(rho(T,p)+p['lD']))*Pi*G(i,N,T,p)
    # Phi decreasing, Phi(0)=I0>0
    hi = 1.0
    while Phi(hi) > 0 and hi < 1e6: hi *= 2
    return brentq(Phi, 0.0, hi, xtol=1e-10)

def I2(T, p):  return G(2, Nstar(1,T,p), T, p) - p['d2']   # P2 invading E1
def I1(T, p):  return G(1, Nstar(2,T,p), T, p) - p['d1']   # P1 invading E2

def edges(p, Tlo=8.0, Thi=30.0, ngrid=441):
    Ts = np.linspace(Tlo, Thi, ngrid)
    i2 = np.array([I2(T,p) for T in Ts])
    i1 = np.array([I1(T,p) for T in Ts])
    def first_up(x):    # inf{T: x>0}
        s = np.where((x[:-1] <= 0) & (x[1:] > 0))[0]
        if len(s)==0: return np.nan
        k=s[0]; return brentq(lambda T: (I2(T,p)), Ts[k], Ts[k+1]) if x is i2 else brentq(lambda T: I1(T,p), Ts[k], Ts[k+1])
    # T1: I2 goes - -> +  ; T2: I1 goes + -> -
    s2 = np.where((i2[:-1] <= 0) & (i2[1:] > 0))[0]
    s1 = np.where((i1[:-1] >= 0) & (i1[1:] < 0))[0]
    T1 = brentq(lambda T: I2(T,p), Ts[s2[0]], Ts[s2[0]+1]) if len(s2) else np.nan
    T2 = brentq(lambda T: I1(T,p), Ts[s1[-1]], Ts[s1[-1]+1]) if len(s1) else np.nan
    return T1, T2

if __name__ == '__main__':
    p = model.make_p()             # reference set
    T1, T2 = edges(p)
    print("reference edges:  T1=%.3f  T2=%.3f  width=%.3f" % (T1, T2, T2-T1))
    print("paper:            T1=15.37 T2=20.10 width=4.73")
    print("Pstar1=%.3f Pstar2=%.3f (paper ~1.67)" % (Pstar(1,p), Pstar(2,p)))
