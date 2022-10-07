import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az
from scipy.stats import expon

np.random.seed(1)


x = expon.rvs(scale=1.0/4, size=10000)
y = expon.rvs(scale=1.0/6, size=10000)
z = [0 for i in range(10000)]
print(x);
print(y);
for i in range(10000):
    if i%10<4:
        z[i]=x[i]
    else:
        z[i]=y[i]
az.plot_posterior({'x':x,'y':y,'z':z}) # Afisarea aproximarii densitatii probabilitatilor, mediei, intervalului etc. variabilelor x,y,z
plt.show()