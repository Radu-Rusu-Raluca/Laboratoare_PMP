import arviz as az
import matplotlib.pyplot as plt

import numpy as np
import pymc3 as pm
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv('Prices.csv')

    price = data['Price'].values
    speed = data['Speed'].values
    hardDrive = data['HardDrive'].values
    ram = data['Ram'].values
    premium = data['Premium'].values

    print(data.shape[0])
    PC_prices_model = pm.Model()
    with PC_prices_model:
        a = pm.Normal('a', mu=0, sd=10)
        bSpeed = pm.Normal('bSpeed', mu=0, sd=10)
        bHard = pm.Normal('bHard', mu=0, sd=10)
        sigma = pm.HalfNormal('sigma', sd=1)

        mu = pm.Deterministic('mu', a + bSpeed * speed + bHard * np.log(hardDrive))
        price_like = pm.Normal('price_like', mu=mu, sd=sigma, observed=price)

        trace = pm.sample(20000, tune=20000, cores=4)
        trace = pm.sample(return_inferencedata=True)

    ppc = pm.sample_posterior_predictive(trace, samples=500, model=PC_prices_model)

    az.plot_posterior(ppc, hdi_prob=0.95)

