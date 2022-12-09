import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
import random

az.style.use('arviz-darkgrid')
dummy_data = np.loadtxt("C:\\Users\\Raluca\\Desktop\\FII an 3 sem. 1\\PMP\\Lab9\\date.csv")
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]
order = 5
x_1p = np.vstack([x_1 ** i for i
                  in range(1, order + 1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')
# plt.show()

# 1a
with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10, shape=5)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)
plt.plot(x_1s[0][idx], y_p_post[idx], 'C1', label=f'model order {order} a')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.savefig("lab9_1_a.png")

# 1b
with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=100, shape=5)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)
plt.plot(x_1s[0][idx], y_p_post[idx], 'C1', label=f'model order {order} sd=100')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.savefig("lab9_1_a_b_100.png")

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=5)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_1s)
    idata_p = pm.sample(2000, return_inferencedata=True)

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_1s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_1s)
plt.plot(x_1s[0][idx], y_p_post[idx], 'C1', label=f'model order {order} sd=array')
plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.savefig("lab9_1_a_b_100_b_array.png")

# 2a
x_2 = x_1.tolist().copy()
y_2 = y_1.tolist().copy()
minx = min(x_1)
miny = min(y_1)
maxx = max(x_1)
maxy = max(y_1)
for i in range(dummy_data.shape[0], 500):
    random_x = random.uniform(minx, maxx)
    random_y = random.uniform(miny, maxy)
    x_2.append(random_x)
    y_2.append(random_y)

x_2 = np.array(x_2)
y_2 = np.array(y_2)
order = 5
x_2p = np.vstack([x_2 ** i for i
                  in range(1, order + 1)])
x_2s = (x_2p - x_2p.mean(axis=1, keepdims=True)) / x_2p.std(axis=1, keepdims=True)
y_2s = (y_2 - y_2.mean()) / y_2.std()

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=10, shape=5)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_2s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_2s)
    idata_p = pm.sample(2000, return_inferencedata=True)

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_2s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_2s)
plt.plot(x_2s[0][idx], y_p_post[idx], 'C1', label=f'model order {order} a')
plt.scatter(x_2s[0], y_2s, c='C0', marker='.')
plt.legend()
plt.savefig("lab9_2_a.png")

# 2b
with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=100, shape=5)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_2s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_2s)
    idata_p = pm.sample(2000, return_inferencedata=True)

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_2s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_2s)
plt.plot(x_2s[0][idx], y_p_post[idx], 'C1', label=f'model order {order} sd=100')
plt.scatter(x_2s[0], y_2s, c='C0', marker='.')
plt.legend()
plt.savefig("lab9_2_a_b_100.png")

with pm.Model() as model_p:
    α = pm.Normal('α', mu=0, sd=1)
    β = pm.Normal('β', mu=0, sd=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=5)
    ε = pm.HalfNormal('ε', 5)
    μ = α + pm.math.dot(β, x_2s)
    y_pred = pm.Normal('y_pred', mu=μ, sd=ε, observed=y_2s)
    idata_p = pm.sample(2000, return_inferencedata=True)

α_p_post = idata_p.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p.posterior['β'].mean(("chain", "draw")).values
idx = np.argsort(x_2s[0])
y_p_post = α_p_post + np.dot(β_p_post, x_2s)
plt.plot(x_2s[0][idx], y_p_post[idx], 'C1', label=f'model order {order} sd=array')
plt.scatter(x_2s[0], y_2s, c='C0', marker='.')
plt.legend()
plt.savefig("lab9_2_a_b_100_b_array.png")
