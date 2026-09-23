import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
import model
def face_rhs(t,y,p,T,which):
    N,P,Z,D=[max(x,1e-12) for x in y]
    i='1' if which==2 else '2'  # which=2 -> {P2=0}, survivor P1
    Topt=p['Topt1'] if which==2 else p['Topt2']; w=p['w1'] if which==2 else p['w2']
    ws=p['wskew1'] if which==2 else p['wskew2']; KN=p['KN1'] if which==2 else p['KN2']
    KP=p['KP1'] if which==2 else p['KP2']; d=p['d1'] if which==2 else p['d2']
    th=model.theta(T,Topt,w,ws); rh=model.rho(T,p['rho0'],p['Q10'],p['Tref'])
    gr=p['mu1' if which==2 else 'mu2']*th*model.f(N,KN)*P; gP=model.g(P,KP); rm=rh*D
    return [p['I0']-p['lN']*N-gr+rm, gr-p['mZ']*gP*Z-d*P,
            p['e']*p['mZ']*gP*Z-p['dZ']*Z, d*P+(1-p['e'])*p['mZ']*gP*Z+p['dZ']*Z-rm-p['lD']*D]
def peak_signature(which,T):
    p=model.make_p(T=T,I0=0.5)
    s=solve_ivp(lambda t,y:face_rhs(t,y,p,T,which),[0,25000],[1,0.5,0.4,2.0],
                method='RK45',rtol=1e-10,atol=1e-12,dense_output=True)
    tt=np.linspace(20000,25000,50000); P=s.sol(tt)[1]
    amp=P.max()-P.min()
    if amp<1e-2: return amp,0,0.0,"equilibrium"
    pk,_=find_peaks(P,prominence=amp*0.05)
    if len(pk)<3: return amp,len(pk),0.0,"few-peaks"
    h=P[pk]; cv=h.std()/h.mean()  # peak-height spread: ~0 simple cycle, bimodal=period-doubling
    # also period regularity
    per=np.diff(tt[pk]); pcv=per.std()/per.mean()
    kind="SIMPLE limit cycle" if (cv<0.02 and pcv<0.02) else ("period-doubling?" if cv>0.05 else "torus/QP?")
    return amp,len(pk),cv,kind
print("=== {P2=0} yuzu (soguk tip yalniz) pencere boyunca ===")
for T in [18.5,19.0,19.5,20.0]:
    a,n,cv,k=peak_signature(2,T); print("  T=%4.1f amp=%.2f peaks=%d heightCV=%.4f -> %s"%(T,a,n,cv,k))
print("=== {P1=0} yuzu (sicak tip yalniz) pencere boyunca ===")
for T in [15.5,16.0,17.0,17.7,18.0,19.0]:
    a,n,cv,k=peak_signature(1,T); print("  T=%4.1f amp=%.2f peaks=%d heightCV=%.4f -> %s"%(T,a,n,cv,k))
