import statistics

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import expon

np.random.seed(1)

a = expon.rvs(scale=1.0 / 4, size=10000)
b = expon.rvs(scale=1.0 / 6, size=10000)

x = [0 for i in range(10000)]
for i in range(10000):
    if i % 10 < 4:
        x[i] = a[i]
    else:
        x[i] = b[i]

print("media lui X: ", statistics.mean(x))
print("deviatia standard a lui X: ", statistics.stdev(x))

az.plot_posterior({'a': a, 'b': b, 'x': x})
plt.show()
