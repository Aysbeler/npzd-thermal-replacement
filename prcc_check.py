import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import brentq
from scipy.stats import rankdata
import model
from mc_edges import G, Nstar

def edges_fast(p,Tlo=8.0,Thi=30.0,ng=161):
    Ts=np.linspace(Tlo,Thi,ng)
    i2=np.array([G(2,Nstar(1,T,p),T,p)-p['d2'] for T in Ts])
    i1=np.array([G(1,Nstar(2,T,p),T,p)-p['d1'] for T in Ts])
    s2=np.where((i2[:-1]<=0)&(i2[1:]>0))[0]; s1=np.where((i1[:-1]>=0)&(i1[1:]<0))[0]
    T1=brentq(lambda T:G(2,Nstar(1,T,p),T,p)-p['d2'],Ts[s2[0]],Ts[s2[0]+1]) if len(s2) else np.nan
    T2=brentq(lambda T:G(1,Nstar(2,T,p),T,p)-p['d1'],Ts[s1[-1]],Ts[s1[-1]+1]) if len(s1) else np.nan
    return T1,T2

if __name__=='__main__':
    # LHS sample: rates +-20%, optima +-2C
    RATE=['mu1','mu2','mZ','dZ','d1','d2','rho0','e','KN1','KN2']
    OPT =['Topt1','Topt2']
    labels=RATE+OPT
    paper={'mu1':'beta1','mu2':'beta2','mZ':'gamma','dZ':'nuZ','d1':'nu1','d2':'nu2',
           'rho0':'rho0','e':'e','KN1':'kappaN1','KN2':'kappaN2','Topt1':'Topt1','Topt2':'Topt2'}
    N=120; rng=np.random.default_rng(7)
    def lhs(n,k):
        u=(np.tile(np.arange(n),(k,1)).T+rng.random((n,k)))/n
        for j in range(k): rng.shuffle(u[:,j])
        return u
    U=lhs(N,len(labels)); base=model.make_p()
    X=np.zeros((N,len(labels))); Y=np.full((N,3),np.nan)  # T1,T2,width
    for s in range(N):
        p=model.make_p()
        for j,k in enumerate(RATE): val=base[k]*(0.8+0.4*U[s,j]); p[k]=val; X[s,j]=val
        for jj,k in enumerate(OPT): val=base[k]+(-2+4*U[s,len(RATE)+jj]); p[k]=val; X[s,len(RATE)+jj]=val
        T1,T2=edges_fast(p); Y[s]=[T1,T2,(T2-T1)]
    
    def prcc(X,y):
        ok=np.isfinite(y); X=X[ok]; y=y[ok]
        Rx=np.apply_along_axis(rankdata,0,X); ry=rankdata(y)
        n,k=Rx.shape; out=np.zeros(k)
        A=np.column_stack([np.ones(n),Rx])
        for j in range(k):
            idx=[c for c in range(k) if c!=j]
            Z=np.column_stack([np.ones(n),Rx[:,idx]])
            bx=np.linalg.lstsq(Z,Rx[:,j],rcond=None)[0]; rx=Rx[:,j]-Z@bx
            by=np.linalg.lstsq(Z,ry,rcond=None)[0];       rr=ry-Z@by
            out[j]=np.corrcoef(rx,rr)[0,1]
        return out
    
    names=['T1 (lower edge)','T2 (upper edge)','window width']
    for c in range(3):
        pr=prcc(X,Y[:,c])
        order=np.argsort(-np.abs(pr))
        print("\n%s  (n=%d valid)"%(names[c],int(np.isfinite(Y[:,c]).sum())))
        for j in order[:6]:
            print("   %-9s PRCC=%+.2f"%(paper[labels[j]],pr[j]))
    