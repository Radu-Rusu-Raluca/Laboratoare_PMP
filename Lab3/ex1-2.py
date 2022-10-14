import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az


def main():
    model = pm.Model()
    with model:
        cutremur = pm.Bernoulli('C', 0.0005)
        incendiu = pm.Deterministic('I', pm.math.switch(cutremur, 0.03, 0.01))
        alarma_d = pm.Deterministic('alarma_d', pm.math.switch(incendiu, pm.math.switch(cutremur, 0.98, 0.95), pm.math.switch(cutremur, 0.02, 0.0001)))
        alarma = pm.Bernoulli('A', p=alarma_d, observed=1)
        trace = pm.sample(20000, chains=1)

    dictionary = {
                  'cutremur': trace['C'].tolist(),
                  'incendiu': trace['I'].tolist()
                  }
    df = pd.DataFrame(dictionary)
    p_cutremur = df[(df['cutremur'] == 1)].shape[0] / df.shape[0]
    print(p_cutremur)

    az.plot_posterior(trace)
    plt.show()

if __name__ == "__main__":
    main()