import numpy as np
from scipy.optimize import brentq
import model
from mc_edges import theta, rho, f, G, Pstar, Nstar, edges
from mc_hopf import E2_state, maxRe

RATE = ['mu1','mu2','KN1','KN2','mZ','dZ','d1','d2','rho0','e']   # beta,kappaN,gamma,nuZ,nu,rho0,e

def edges_fast(p, Tlo=8.0, Thi=30.0, ngrid=161):
    Ts=np.linspace(Tlo,Thi,ngrid)
    i2=np.array([G(2,Nstar(1,T,p),T,p)-p['d2'] for T in Ts])
    i1=np.array([G(1,Nstar(2,T,p),T,p)-p['d1'] for T in Ts])
    s2=np.where((i2[:-1]<=0)&(i2[1:]>0))[0]
    s1=np.where((i1[:-1]>=0)&(i1[1:]<0))[0]
    T1=brentq(lambda T:G(2,Nstar(1,T,p),T,p)-p['d2'],Ts[s2[0]],Ts[s2[0]+1]) if len(s2) else np.nan
    T2=brentq(lambda T:G(1,Nstar(2,T,p),T,p)-p['d1'],Ts[s1[-1]],Ts[s1[-1]+1]) if len(s1) else np.nan
    return T1,T2

def warm_hopf_fast(p,T2,Thi=32.0,step=0.3):
    if not np.isfinite(T2): return np.nan
    Ts=np.arange(T2+0.5,Thi,step); mr=np.array([maxRe(T,p) for T in Ts])
    s=np.where((mr[:-1]<=0)&(mr[1:]>0))[0]
    if len(s)==0: return np.nan
    k=s[0]; return brentq(lambda T:maxRe(T,p),Ts[k],Ts[k+1])

def sample(rng, thermal):
    p=model.make_p()
    for k in RATE: p[k]*=1+rng.uniform(-0.15,0.15)
    if thermal:
        p['Topt1']+=rng.uniform(-2,2); p['Topt2']+=rng.uniform(-2,2)
        for k in ['w1','w2','wskew1','wskew2']: p[k]*=1+rng.uniform(-0.20,0.20)
        if p['wskew1']>=p['w1']: p['w1']=p['wskew1']+0.5   # keep left-skew w>w-
        if p['wskew2']>=p['w2']: p['w2']=p['wskew2']+0.5
    try:
        T1,T2=edges_fast(p); TH=warm_hopf_fast(p,T2)
    except Exception:
        T1,T2,TH=np.nan,np.nan,np.nan
    return T1,T2,TH

def run(N=200, thermal=False, seed=2024):
    rng=np.random.default_rng(seed)
    R=np.array([sample(rng,thermal) for _ in range(N)])
    return R[:,0],R[:,1],R[:,2]           # T1,T2,TH

N=200
A1,A2,AH=run(N,thermal=False,seed=2024)   # rate only
B1,B2,BH=run(N,thermal=True, seed=2024)   # + thermal
np.savez('mc_results.npz',A1=A1,A2=A2,AH=AH,B1=B1,B2=B2,BH=BH,N=N)

def stat(x):
    v=x[np.isfinite(x)]
    return len(v),N,np.nanpercentile(v,2.5),np.nanpercentile(v,97.5)
for name,x in [('T1(rate)',A1),('T2(rate)',A2),('TH(rate)',AH),
               ('T1(therm)',B1),('T2(therm)',B2),('TH(therm)',BH)]:
    n,tot,lo,hi=stat(x); print("%-11s persist %3d/%d  95%%[%.2f, %.2f]"%(name,n,tot,lo,hi))
# window persistence = both edges found
for lbl,e1,e2 in [('window(rate)',A1,A2),('window(therm)',B1,B2)]:
    n=int(np.sum(np.isfinite(e1)&np.isfinite(e2))); print("%-13s %d/%d"%(lbl,n,N))
