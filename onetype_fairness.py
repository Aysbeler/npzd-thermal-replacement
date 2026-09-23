import numpy as np, csv, warnings; warnings.filterwarnings('ignore')
from table7_recompute import onetype_monthly
from calib_cobs import make_Tfun, model_monthly
import model
rows=list(csv.DictReader(open('station_climatology_point.csv')))
sst=np.array([float(r['sst']) for r in rows]); phyc=np.array([float(r['phyc']) for r in rows])
Tfun=make_Tfun(sst)
def obs_metrics(Ph,obs):
    c=np.sum(obs*Ph)/np.sum(Ph**2); resid=obs-c*Ph
    return np.sum(resid**2), np.corrcoef(Ph,obs)[0,1], np.median(np.abs(resid))
# two-type reference
Ph2=model_monthly(model.make_p(Topt1=15.5,Topt2=20.5),Tfun)
rss2,r2,mae2=obs_metrics(Ph2,phyc)
print("TWO-TYPE:                    r=%.3f RSS=%.3f MAE=%.3f"%(r2,rss2,mae2))
print("\nONE-TYPE best fit (max correlation) for each half-saturation choice:")
for KN,lbl in [(1.2,'cold-type kN,1'),(0.5,'warm-type kN,2'),(0.85,'averaged kN')]:
    best=None
    for Topt in np.arange(12,23.01,0.5):
        Ph=onetype_monthly(Topt,KN); r=np.corrcoef(Ph,phyc)[0,1]
        rss,rr,mae=obs_metrics(Ph,phyc)
        if best is None or r>best[0]: best=(r,Topt,rss,mae)
    r,Topt,rss,mae=best
    verdict="two-type WINS" if r2>r else "one-type better?!"
    print("  KN=%.2f (%-15s): best r=%.3f (Topt=%.1f) RSS=%.3f MAE=%.3f  -> %s"%(
        KN,lbl,r,Topt,rss,mae,verdict))
