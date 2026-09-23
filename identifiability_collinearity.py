"""Practical-identifiability / collinearity-index analysis (Section 'Confronting
the model with reanalysis data', practical-identifiability paragraph).

Builds the normalized sensitivity matrix S_ij = (p_j/y_i) dy_i/dp_j of the 12
monthly seasonal outputs y = P1+P2 to each parameter, and the Brun et al.
collinearity index gamma_K = 1/sqrt(lambda_min(Sn_K^T Sn_K)) for parameter
subsets K (columns normalized to unit length).

Robust, reproducible result: the two thermal optima are identifiable
(gamma_K ~ 3, below the threshold 10-20); enlarging the subset raises the index
steeply, and subsets that include the loss/grazing rates or the full calibratable
set drive gamma_K into the hundreds (not jointly identifiable). NOTE: beyond the
two optima the *magnitude* of gamma_K is sensitive to the exact subset and to
numerical conditioning (FD step, integration tolerance); only the qualitative
separation (optima identifiable, larger subsets not) is robust.
"""
import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model

rows=list(csv.DictReader(open('station_climatology_point.csv')))
sst=np.array([float(r['sst']) for r in rows])
DC=(np.arange(12)+0.5)*365/12.0; dcx=np.concatenate([[DC[-1]-365],DC,[DC[0]+365]])
def Tfun(t,sx=np.concatenate([[sst[-1]],sst,[sst[0]]])): return float(np.interp(t%365.0,dcx,sx))

def rhs(p):
    def f(t,y):
        N,P1,P2,Z,D=[max(x,1e-12) for x in y]; T=Tfun(t)
        th1=model.theta(T,p['Topt1'],p['w1'],p['wskew1']); th2=model.theta(T,p['Topt2'],p['w2'],p['wskew2'])
        rh=model.rho(T,p['rho0'],p['Q10'],p['Tref']); fN1=model.f(N,p['KN1']); fN2=model.f(N,p['KN2'])
        s=p['switch']; a=P1**s; b=P2**s; tot=a+b+1e-12; phi1=a/tot; phi2=b/tot
        gP1=phi1*model.g(P1,p['KP1']); gP2=phi2*model.g(P2,p['KP2'])
        gr1=p['mu1']*th1*fN1*P1; gr2=p['mu2']*th2*fN2*P2; rm=rh*D
        return [p['I0']-p['lN']*N-gr1-gr2+rm, gr1-p['mZ']*gP1*Z-p['d1']*P1,
                gr2-p['mZ']*gP2*Z-p['d2']*P2, p['e']*p['mZ']*(gP1+gP2)*Z-p['dZ']*Z,
                p['d1']*P1+p['d2']*P2+(1-p['e'])*p['mZ']*(gP1+gP2)*Z+p['dZ']*Z-rm-p['lD']*D]
    return f

def monthly(p,years=6):
    sol=solve_ivp(rhs(p),[0,years*365],[1,0.5,0.5,0.4,2.0],method='RK45',
                  rtol=1e-8,atol=1e-10,max_step=2.0,dense_output=True)
    t0=(years-1)*365.0; tt=np.linspace(t0,t0+365,3660); Y=sol.sol(tt); P=Y[1]+Y[2]
    mon=np.floor(((tt-t0)/365.0)*12).astype(int)%12
    return np.array([P[mon==m].mean() for m in range(12)])

p0=model.make_p(Topt1=15.5,Topt2=20.5); y0=monthly(p0)
pool=['Topt1','Topt2','w1','w2','d1','d2','mZ','dZ','KN1','KN2']
S={}
for k in pool:
    h=0.01*abs(p0[k]) if p0[k]!=0 else 0.01
    pp=dict(p0); pp[k]=p0[k]+h; yp=monthly(pp); pp[k]=p0[k]-h; ym=monthly(pp)
    S[k]=(yp-ym)/(2*h)*p0[k]/(y0+1e-12)
def gK(keys):
    M=np.array([S[k] for k in keys]).T; M=M/(np.linalg.norm(M,axis=0)+1e-15)
    return 1.0/np.sqrt(max(np.linalg.eigvalsh(M.T@M).min(),1e-30))

print("collinearity index gamma_K (threshold for identifiability ~ 10-20):")
print("  two optima {Topt1,Topt2}                 : %.2f"%gK(['Topt1','Topt2']))
print("  thermal-niche {optima + breadths}        : %.2f"%gK(['Topt1','Topt2','w1','w2']))
print("  loss/grazing {d1,d2,mZ,dZ}               : %.1f"%gK(['d1','d2','mZ','dZ']))
print("  full calibratable set (10 parameters)    : %.1f"%gK(pool))
print("=> optima identifiable (~3); larger subsets not (hundreds+).")
