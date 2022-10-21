import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az
from sympy import beta


l=20
μ=1/60
σ=1/120
a=1

model = pm.Model()
with model:
    clients=pm.Poisson('C', mu=l)
    place_order = pm.Normal('P', mu=μ, sd=σ)
    make_order= pm.Exponential('M', pow(a, -1))
    trace = pm.sample(20000)

dictionary = {
                  'clients': trace['C'].tolist(),
                  'make_order': trace['M'].tolist(),
                  'place_order': trace['P'].tolist()
                  }

df = pd.DataFrame(dictionary)

