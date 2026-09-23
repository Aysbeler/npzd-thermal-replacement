import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import brentq
import model
from mc_edges import G, rho as _rho
from c3_zface_true import predator_free_eq, grazer_eig_at
RATE=['mu1','mu2','mZ','dZ','d1','d2','rho0','e','KN1','KN2']
rng=np.random.default_rng(2024); N=200
def edges(p):
    Ts=np.linspace(8,30,161)
    i2=np.array([G(2,__import__('mc_edges').Nstar(1,T,p),T,p)-p['d2'] for T in Ts])
    i1=np.array([G(1,__import__('mc_edges').Nstar(2,T,p),T,p)-p['d1'] for T in Ts])
    s2=np.where((i2[:-1]<=0)&(i2[1:]>0))[0]; s1=np.where((i1[:-1]>=0)&(i1[1:]<0))[0]
    if not len(s2) or not len(s1): return None
    T1=brentq(lambda T:G(2,__import__('mc_edges').Nstar(1,T,p),T,p)-p['d2'],Ts[s2[0]],Ts[s2[0]+1])
    T2=brentq(lambda T:G(1,__import__('mc_edges').Nstar(2,T,p),T,p)-p['d1'],Ts[s1[-1]],Ts[s1[-1]+1])
    return T1,T2
ok_grazer=0; ok_admiss=0; tot=0
mineig=np.inf
for _ in range(N):
    p=model.make_p()
    for k in RATE: p[k]*=1+rng.uniform(-0.15,0.15)
    e=edges(p)
    if e is None: continue
    T1,T2=e
    if not (T2>T1): continue
    tot+=1
    Tm=0.5*(T1+T2)  # mid-window
    p['T']=Tm
    # grazer eig at predator-free winner
    cands=[]
    for i in (1,2):
        r=predator_free_eq(i,Tm,p)
        if r is not None: cands.append((r[0][0],i,r[0],r[1]))
    if cands:
        Nc,i,yf,P=min(cands,key=lambda c:c[0])
        eig=grazer_eig_at(yf,P,i,Tm,p); mineig=min(mineig,eig)
        if eig>0: ok_grazer+=1
    # admissibility e*gamma>nuZ
    if p['e']*p['mZ']>p['dZ']: ok_admiss+=1
print("valid windows: %d/%d"%(tot,N))
print("grazer-invasion eig>0 at mid-window predator-free state: %d/%d (min eig=%.4f)"%(ok_grazer,tot,mineig))
print("e*gamma>nuZ (E_i admissible): %d/%d"%(ok_admiss,tot))
