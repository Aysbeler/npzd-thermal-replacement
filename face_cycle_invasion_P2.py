import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model

def face_rhs(t,y,p,T):
    N,P1,Z,D=[max(x,1e-12) for x in y]
    th1=model.theta(T,p['Topt1'],p['w1'],p['wskew1']); rh=model.rho(T,p['rho0'],p['Q10'],p['Tref'])
    fN1=model.f(N,p['KN1']); gP1=model.g(P1,p['KP1'])
    gr1=p['mu1']*th1*fN1*P1; rm=rh*D
    return [p['I0']-p['lN']*N-gr1+rm, gr1-p['mZ']*gP1*Z-p['d1']*P1,
            p['e']*p['mZ']*gP1*Z-p['dZ']*Z, p['d1']*P1+(1-p['e'])*p['mZ']*gP1*Z+p['dZ']*Z-rm-p['lD']*D]

def G2(N,T,p):  # warm-type realized growth
    return p['mu2']*model.theta(T,p['Topt2'],p['w2'],p['wskew2'])*model.f(N,p['KN2'])

print("=== Face limit cycle uzerinde zaman-ortalamali P2 invasion fitness <I2> ===")
print("(>0 ise sinir cevrimi transversal itici -> persistence korunur)")
for T in [18.0,18.5,19.0,19.5,20.0]:
    p=model.make_p(T=T,I0=0.5)
    # settle onto face cycle, then integrate one long window on the cycle
    s=solve_ivp(lambda t,y:face_rhs(t,y,p,T),[0,20000],[1,0.5,0.4,2.0],
                method='RK45',rtol=1e-10,atol=1e-12,dense_output=True)
    tt=np.linspace(18000,20000,20000); Y=s.sol(tt); N=Y[0]
    I2_inst=G2(N,T,p)-p['d2']            # instantaneous invasion fitness of P2 along cycle
    I2_avg=np.trapezoid(I2_inst,tt)/(tt[-1]-tt[0])
    amp=Y[1].max()-Y[1].min()
    # compare with equilibrium-based I2 at E1 (what paper currently uses)
    print("  T=%4.1f: cycle P1-amp=%.2f  <I2>_cycle=%+.4f  -> %s"%(
        T,amp,I2_avg,"REPELLING (persist ok)" if I2_avg>0 else "*** NOT repelling ***"))

print("\n=== Kontrol: tam 5B sistem T=19,i0=0.5'te ic dengeye mi gidiyor (sinirdan uzak)? ===")
p=model.make_p(T=19.0,I0=0.5); rhs=model.make_rhs(p)
s=solve_ivp(rhs,[0,8000],[1,0.8,0.8,0.4,2.0],method='RK45',rtol=1e-9,atol=1e-11,t_eval=[8000])
y=s.y[:,-1]; print("  son durum (N,P1,P2,Z,D)=(%.3f,%.3f,%.3f,%.3f,%.3f)  min bileşen=%.3f"%(
    y[0],y[1],y[2],y[3],y[4],min(y)))
