import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model
from calib_cobs import make_Tfun, model_monthly   # two-type seasonal integrator

rows=list(csv.DictReader(open('station_climatology_point.csv')))
sst =np.array([float(r['sst'])  for r in rows])
phyc=np.array([float(r['phyc']) for r in rows])
Tfun=make_Tfun(sst)

# ---- one-type N-P-Z-D seasonal model ----
def onetype_rhs(p,Tfun):
    def rhs(t,y):
        N,P,Z,D=[max(x,1e-12) for x in y]; T=Tfun(t)
        th=model.theta(T,p['Topt'],p['w'],p['wskew']); rh=model.rho(T,p['rho0'],p['Q10'],p['Tref'])
        gr=p['mu']*th*model.f(N,p['KN'])*P; gz=p['mZ']*model.g(P,p['KP'])*Z
        return [p['I0']-p['lN']*N-gr+rh*D,
                gr-gz-p['d']*P,
                p['e']*gz-p['dZ']*Z,
                p['d']*P+(1-p['e'])*gz+p['dZ']*Z-rh*D-p['lD']*D]
    return rhs

def onetype_monthly(Topt,KN,years=6):
    b=model.make_p()
    p=dict(Topt=Topt,w=b['w1'],wskew=b['wskew1'],rho0=b['rho0'],Q10=b['Q10'],Tref=b['Tref'],
           mu=b['mu1'],KN=KN,mZ=b['mZ'],KP=b['KP1'],d=b['d1'],e=b['e'],dZ=b['dZ'],
           lN=b['lN'],lD=b['lD'],I0=b['I0'])
    sol=solve_ivp(onetype_rhs(p,Tfun),[0,years*365],[1,0.6,0.4,2.0],method='RK45',
                  rtol=1e-6,atol=1e-8,max_step=5.0,dense_output=True)
    t0=(years-1)*365.0; tt=np.linspace(t0,t0+365,3660)
    P=sol.sol(tt)[1]; mon=np.floor(((tt-t0)/365.0)*12).astype(int)%12
    return np.array([P[mon==m].mean() for m in range(12)])

# ---- unit-max normalized shape metrics ----
def metrics(Ph,obs):
    Pn=Ph/Ph.max(); On=obs/obs.max()
    rss=np.sum((Pn-On)**2); mae=np.median(np.abs(Pn-On)); r=np.corrcoef(Ph,obs)[0,1]
    return rss,r,mae

if __name__=='__main__':
    # two-type at stated optima (15.5,20.5)
    Ph2=model_monthly(model.make_p(Topt1=15.5,Topt2=20.5),Tfun)
    rss2,r2,mae2=metrics(Ph2,phyc)
    
    # one-type: fit single Topt over grid, try KN in {0.5,0.85,1.2}, keep best normalized RSS
    best=None
    for KN in (0.5,0.85,1.2):
        for Topt in np.arange(13,23.01,0.5):
            Ph=onetype_monthly(Topt,KN)
            rss,r,mae=metrics(Ph,phyc)
            if best is None or rss<best[0]: best=(rss,r,mae,Topt,KN,Ph)
    rss1,r1,mae1,Topt1b,KN1b,Ph1=best
    
    print("Convention: both series normalized to unit maximum (dimensionless shape comparison)")
    print("2 fitted = single Topt + scale (one-type); 3 fitted = two optima + scale (two-type)\n")
    print("one-type  best Topt=%.1f (KN=%.2f):  RSS=%.3f  r=%.3f  MAE=%.3f"%(Topt1b,KN1b,rss1,r1,mae1))
    print("two-type  (15.5,20.5):               RSS=%.3f  r=%.3f  MAE=%.3f"%(rss2,r2,mae2))
    print("\nphyc peak M%d | one-type peak M%d | two-type peak M%d"%(
        int(np.argmax(phyc))+1,int(np.argmax(Ph1))+1,int(np.argmax(Ph2))+1))
    np.savez('table7.npz',rss1=rss1,r1=r1,mae1=mae1,rss2=rss2,r2=r2,mae2=mae2,Topt1=Topt1b,KN1=KN1b)
    