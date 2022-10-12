import statistics
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import expon

np.random.seed(1)

# stema=1, ban=0

ss = [0 for i in range(100)]
sb = [0 for i in range(100)]
bs = [0 for i in range(100)]
bb = [0 for i in range(100)]

for i in range(100):
    moneda1 = stats.binom.rvs(n=1, p=0.5, size=10)
    moneda2 = stats.binom.rvs(n=1, p=0.3, size=10)
    ssEx, sbEx, bsEx, bbEx = 0, 0, 0, 0
    for j in range(10):
        if (moneda1[j] == 1) and (moneda2[j] == 1):
            ssEx = ssEx + 1
        if (moneda1[j] == 1) and (moneda2[j] == 0):
            sbEx = sbEx + 1
        if (moneda1[j] == 0) and (moneda2[j] == 1):
            bsEx = bsEx + 1
        if (moneda1[j] == 0) and (moneda2[j] == 0):
            bbEx = bbEx + 1
    ss[i] = ssEx
    sb[i] = sbEx
    bs[i] = bsEx
    bb[i] = bbEx

az.plot_posterior({'ss': ss, 'sb': sb, 'bs': bs, 'bb': bb})
plt.show()
