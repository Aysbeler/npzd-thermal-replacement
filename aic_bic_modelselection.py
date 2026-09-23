"""Information-criterion comparison of the one- vs two-type fit to the CMEMS
seasonal climatology (Section 'Confronting the model with reanalysis data').
Reproduces the values quoted in the text: Delta AICc ~ 36, Delta BIC ~ 38.

Inputs are the residual sums of squares reported in Table (one-vs-two-type),
obtained under identical temperature forcing with least-squares scaling:
  one-type: RSS = 0.60, k = 2 fitted parameters (single optimum + scale)
  two-type: RSS = 0.02, k = 3 fitted parameters (two optima  + scale)
on n = 12 monthly climatology points. Gaussian-error information criteria:
  AIC  = n ln(RSS/n) + 2K,   AICc = AIC + 2K(K+1)/(n-K-1),  BIC = n ln(RSS/n) + K ln n
with K = k + 1 (the residual variance counts as a parameter).
"""
import numpy as np
n = 12
models = {'one-type': dict(rss=0.60, k=2), 'two-type': dict(rss=0.02, k=3)}
out = {}
for name, m in models.items():
    K = m['k'] + 1
    aic  = n*np.log(m['rss']/n) + 2*K
    aicc = aic + 2*K*(K+1)/(n-K-1)
    bic  = n*np.log(m['rss']/n) + K*np.log(n)
    out[name] = dict(AIC=aic, AICc=aicc, BIC=bic)
    print(f"{name:9s} k={m['k']} RSS={m['rss']:.2f}  AIC={aic:8.2f}  AICc={aicc:8.2f}  BIC={bic:8.2f}")
dAICc = out['one-type']['AICc'] - out['two-type']['AICc']
dBIC  = out['one-type']['BIC']  - out['two-type']['BIC']
print(f"\nDelta AICc (one - two) = {dAICc:.2f}   (>10 = strong support for two-type)")
print(f"Delta BIC  (one - two) = {dBIC:.2f}")
print("Caveat: the 12 monthly points are seasonally autocorrelated, so the")
print("effective sample size is < 12; the magnitude is optimistic but the sign is robust.")
