import numpy as np, csv
from scipy.integrate import solve_ivp
import model, warnings; warnings.filterwarnings('ignore')

# ---- center-cell monthly climatology ----
rows=list(csv.DictReader(open('station_climatology_point.csv')))
sst =np.array([float(r['sst'])  for r in rows])
phyc=np.array([float(r['phyc']) for r in rows])
DC=(np.arange(12)+0.5)*365/12.0                      # month-centre days
dcx=np.concatenate([[DC[-1]-365],DC,[DC[0]+365]])

def make_Tfun(sst):
    sx=np.concatenate([[sst[-1]],sst,[sst[0]]])
    return lambda t: float(np.interp(t%365.0, dcx, sx))

def seasonal_rhs(p,Tfun):
    def rhs(t,y):
        N,P1,P2,Z,D=[max(x,1e-12) for x in y]; T=Tfun(t)
        th1=model.theta(T,p['Topt1'],p['w1'],p['wskew1']); th2=model.theta(T,p['Topt2'],p['w2'],p['wskew2'])
        rh=model.rho(T,p['rho0'],p['Q10'],p['Tref']); fN1=model.f(N,p['KN1']); fN2=model.f(N,p['KN2'])
        s=p['switch']; a=P1**s; b=P2**s; tot=a+b+1e-12; phi1=a/tot; phi2=b/tot
        gP1=phi1*model.g(P1,p['KP1']); gP2=phi2*model.g(P2,p['KP2'])
        gr1=p['mu1']*th1*fN1*P1; gr2=p['mu2']*th2*fN2*P2; rm=rh*D
        return [p['I0']-p['lN']*N-gr1-gr2+rm,
                gr1-p['mZ']*gP1*Z-p['d1']*P1,
                gr2-p['mZ']*gP2*Z-p['d2']*P2,
                p['e']*p['mZ']*(gP1+gP2)*Z-p['dZ']*Z,
                p['d1']*P1+p['d2']*P2+(1-p['e'])*p['mZ']*(gP1+gP2)*Z+p['dZ']*Z-rm-p['lD']*D]
    return rhs

def model_monthly(p,Tfun,years=6):
    rhs=seasonal_rhs(p,Tfun)
    sol=solve_ivp(rhs,[0,years*365],[1,0.5,0.5,0.4,2.0],method='RK45',
                  rtol=1e-6,atol=1e-8,max_step=5.0,dense_output=True)
    t0=(years-1)*365.0; tt=np.linspace(t0,t0+365,3660)         # last year, ~0.1 d
    Y=sol.sol(tt); P=Y[1]+Y[2]                                  # P1+P2
    mon=np.floor(((tt-t0)/365.0)*12).astype(int)%12
    return np.array([P[mon==m].mean() for m in range(12)])

def fit_optima(sst,phyc,g1=np.arange(13,18.01,0.5),g2=np.arange(18,23.01,0.5),refine=True):
    Tfun=make_Tfun(sst); best=None
    def score(o1,o2):
        p=model.make_p(Topt1=o1,Topt2=o2)
        Ph=model_monthly(p,Tfun)
        c=np.sum(Ph*phyc)/np.sum(phyc**2)                       # closed-form scale
        rss=np.sum((Ph-c*phyc)**2); r=np.corrcoef(Ph,phyc)[0,1]
        mae=np.median(np.abs(Ph-c*phyc))
        return rss,c,r,mae,Ph
    for o1 in g1:
        for o2 in g2:
            rss,c,r,mae,Ph=score(o1,o2)
            if best is None or rss<best[0]: best=(rss,c,r,mae,o1,o2)
    if refine:
        o1b,o2b=best[4],best[5]
        for o1 in np.arange(o1b-0.4,o1b+0.41,0.1):
            for o2 in np.arange(o2b-0.4,o2b+0.41,0.1):
                rss,c,r,mae,Ph=score(o1,o2)
                if rss<best[0]: best=(rss,c,r,mae,round(o1,2),round(o2,2))
    return best  # rss,c_obs,r,mae,Topt1,Topt2

if __name__=='__main__':
    rss,c,r,mae,o1,o2=fit_optima(sst,phyc)
    print("CENTER CELL 38.479N 26.375E")
    print("  fitted Topt1=%.2f Topt2=%.2f"%(o1,o2))
    print("  c_obs = %.4f"%c)
    print("  r = %.4f   RSS = %.4f   MAE = %.4f"%(r,rss,mae))

