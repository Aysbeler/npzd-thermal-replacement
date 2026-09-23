"""
mld_null.py

Tests whether the seasonal fit of the two-type thermal model can be reproduced by a competing
seasonal driver that carries no thermal-niche mechanism.

The strongest objection to the data confrontation is that temperature covaries with light,
mixed-layer depth and nutrient entrainment, so a good fit need not implicate thermal niches.
The Copernicus physical reanalysis supplies the mixed-layer thickness at the same cell, and it is
a strong seasonal signal in near-antiphase with temperature: about 54 m in winter against 12 m in
summer, a factor of 4.6.  Deep winter mixing entrains nutrients and dilutes biomass, so a single
phytoplankton type forced by the mixed layer is a mechanistically plausible competitor.

We therefore build a one-type NPZD model with no thermal response at all, theta = 1, in which the
mixed layer drives the nutrient supply and the dilution loss,

    iota_0(t) = iota_0 * h(t)/mean(h),        d_1(t) = d_1 * h(t)/mean(h),

so that deep mixing both supplies nutrient and removes phytoplankton.  A second variant drives
only the supply.  Both are fitted to the same twelve climatological points by least squares on
the observation scale, exactly as the thermal models are, and compared on the same footing.

If the mixed-layer null matched the two-type thermal model, the seasonal cycle would not
discriminate between a thermal mechanism and a mixing mechanism, and the empirical section could
claim no support at all for the thermal reading.  The point of the test is to find out.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model

P0 = model.make_p()
DC = (np.arange(12) + 0.5)*365/12.0

# monthly climatologies at the study cell (2001-2022)
MLD = np.array([52.1, 49.0, 44.4, 30.9, 14.8, 12.1, 11.9, 11.9, 14.1, 34.0, 53.5, 54.2])


def _load_obs():
    import csv
    rows = list(csv.DictReader(open('station_climatology_point.csv')))
    sst = np.array([float(r['sst']) for r in rows])
    obs = np.array([float(r['phyc']) for r in rows])
    return sst, obs


SST, OBS = _load_obs()
H = MLD/MLD.mean()


def periodic(x, arr):
    return np.interp(np.mod(x, 365.0), DC, arr, period=365.0)


def rhs_mld(t, y, dilute=True):
    """One-type NPZD with no thermal response, forced by the mixed layer."""
    N, P, Z, D = [max(v, 1e-12) for v in y]
    h = periodic(t, H)
    rho = model.rho(20.0, P0['rho0'], P0['Q10'], P0['Tref'])      # fixed, no thermal effect
    gr = P0['mu1']*model.f(N, P0['KN1'])*P                        # theta = 1
    graz = P0['mZ']*model.g(P, P0['KP1'])*Z
    loss = P0['d1']*(h if dilute else 1.0)*P
    return [P0['I0']*h - P0['lN']*N - gr + rho*D,
            gr - graz - loss,
            P0['e']*graz - P0['dZ']*Z,
            loss + (1-P0['e'])*graz + P0['dZ']*Z - rho*D - P0['lD']*D]


def monthly(dilute=True, years=8):
    s = solve_ivp(lambda t, y: rhs_mld(t, y, dilute), [0, years*365], [1.0, 1.0, 0.4, 2.0],
                  method='RK45', rtol=1e-7, atol=1e-9, max_step=5.0, dense_output=True)
    t0 = (years-1)*365.0
    tt = np.linspace(t0, t0+365, 3660)
    P = s.sol(tt)[1]
    mon = (np.floor(((tt-t0)/365.0)*12).astype(int)) % 12
    return np.array([P[mon == m].mean() for m in range(12)])


def score(pred, k_fitted):
    """Least-squares scale, then correlation, RSS and AICc on the observation scale."""
    if pred.max() < 1e-9:
        return None
    c = (pred @ OBS)/(pred @ pred)
    fit = c*pred
    rss = float(np.sum((fit - OBS)**2))
    r = float(np.corrcoef(fit, OBS)[0, 1])
    n, K = 12, k_fitted + 1 + 1          # +1 for the scale, +1 for the residual variance
    aicc = n*np.log(rss/n) + 2*K + 2*K*(K+1)/(n - K - 1)
    return r, rss, aicc, float(fit.max()/max(fit.min(), 1e-12))


if __name__ == '__main__':
    print("mixed-layer forced nulls against the observed climatology")
    print("=" * 72)
    print(f"MLD climatology (m): {MLD}")
    print(f"winter/summer ratio: {MLD.max()/MLD.min():.1f}")
    print(f"correlation of MLD with SST: {np.corrcoef(MLD, SST)[0,1]:+.3f}\n")

    print(f"{'model':40s} {'r':>7} {'RSS':>9} {'AICc':>9} {'range':>7}")
    print("-"*72)
    for dil, lab in ((True, 'MLD-forced supply and dilution, no theta'),
                     (False, 'MLD-forced supply only, no theta')):
        out = score(monthly(dil), k_fitted=0)
        if out is None:
            print(f"{lab:40s}   collapsed")
            continue
        r, rss, aicc, rng = out
        print(f"{lab:40s} {r:7.3f} {rss:9.4f} {aicc:9.2f} {rng:6.2f}x")
    print(f"{'observed range':40s} {'':>7} {'':>9} {'':>9} "
          f"{OBS.max()/OBS.min():6.2f}x")
    print("\nFor comparison the two-type thermal model attains r = 0.95 with a seasonal")
    print("range of 1.96x, against an observed 1.97x (Table 8).")
