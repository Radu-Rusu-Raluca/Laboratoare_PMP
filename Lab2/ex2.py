import statistics
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import expon

np.random.seed(1)

a = stats.gamma.rvs(4, scale=1.0 / 3, size=10000)
b = stats.gamma.rvs(4, scale=1.0 / 2, size=10000)
c = stats.gamma.rvs(5, scale=1.0 / 2, size=10000)
d = stats.gamma.rvs(5, scale=1.0 / 3, size=10000)

x = [0 for i in range(10000)]
for i in range(10000):
    if i % 100 < 25:
        x[i] = a[i] + stats.expon.rvs(scale=1.0/4)
    elif (i % 100 >= 25) and (i % 100 < 50):
        x[i] = b[i] + stats.expon.rvs(scale=1.0/4)
    elif (i % 100 >= 50) and (i % 100 < 80):
        x[i] = c[i] + stats.expon.rvs(scale=1.0/4)
    else:
        x[i] = d[i] + stats.expon.rvs(scale=1.0/4)

print("media lui X: ", statistics.mean(x))
print("deviatia standard a lui X: ", statistics.stdev(x))

nr = 0
for i in range(10000):
    if x[i] > 3:
        nr = nr + 1
print("probabilitatea ca timpul de raspuns sa fie mai mare de 3 ms: ", nr / 10000)

az.plot_posterior({'a': a, 'b': b, 'c': c, 'd': d,
                   'x': x})
plt.show()
