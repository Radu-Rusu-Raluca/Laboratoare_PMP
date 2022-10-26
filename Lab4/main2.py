import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
from sympy import beta


l=20
μ=1
σ=0.5
for a in np.arange(5, 0, -0.01):
    model = pm.Model()
    with model:
        clients=pm.Poisson('C', mu=l)
        place_order = pm.Normal('P', mu=μ, sd=σ)
        make_order= pm.Exponential('M', lam=1/a)
        total=pm.ExGaussian('T', mu=μ, sigma=σ, nu=a)
        trace = pm.sample(100)

    dictionary = {
                    'clients': trace['C'].tolist(),
                    'make_order': trace['M'].tolist(),
                    'place_order': trace['P'].tolist(),
                    'total_time': np.add(trace['M'], trace['P']).tolist(),
                    'total': trace['T'].tolist(),
                    }

    df = pd.DataFrame(dictionary)
    df
    if((sum(df['total_time']<15)/400)==0.95):
        alpha=a
        mean=sum(df['total_time'])/400
        break
    print(sum(df['total_time']<15)/400)
print(alpha)
print(mean)

