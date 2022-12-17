import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import arviz as az
import pymc3 as pm

# 1
clusters = 3
n_cluster = [90, 170, 240]
n_total = sum(n_cluster)
means = [5, 0, 3]
std_devs = [2, 1, 2]
mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))
az.plot_kde(np.array(mix))
plt.show()
pd.DataFrame({'exp': np.array(mix)}).to_csv('sample.csv')

# 2
cs = pd.read_csv('sample.csv')
cs_exp = cs['exp']
az.plot_kde(cs_exp)
plt.hist(cs_exp, density=True, bins=30, alpha=0.3)
plt.yticks([])

clusters = [2, 3, 4]
models = []
idatas = []
for cluster in clusters:
    with pm.Model() as model:
        p = pm.Dirichlet('p', a=np.ones(cluster))
        means = pm.Normal('means', mu=np.linspace(cs_exp.min(), cs_exp.max(), cluster), sd=10, shape=cluster,
                          transform=pm.distributions.transforms.ordered)
        sd = pm.HalfNormal('sd', sd=10)
        y = pm.NormalMixture('y', w=p, mu=means, sd=sd, observed=cs_exp)
        idata = pm.sample(1000, tune=2000, target_accept=0.9, random_seed=123, return_inferencedata=True)
        idatas.append(idata)
        models.append(model)

# 3
comp1 = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="waic", scale="deviance")
az.plot_compare(comp1)

comp2 = az.compare(dict(zip([str(c) for c in clusters], idatas)), method='BB-pseudo-BMA', ic="loo", scale="deviance")
az.plot_compare(comp2)
