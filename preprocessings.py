# %%
#Escalonamento de Poisson
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def poisson(input_data, mc=True):
  mean = np.mean(input_data, axis=0)
  # Calcular a escala de Pareto
  escala_poisson = 1 / np.sqrt(mean)
  X_poisson = input_data*escala_poisson
  if mc:
        mean_poisson = np.mean(X_poisson, axis=0)
        X_poisson_mc = X_poisson - mean_poisson
        return pd.DataFrame(X_poisson_mc)
  else:
        return pd.DataFrame(X_poisson)

def pareto(input_data, mc=True):
  sd = np.std(input_data, axis=0)
  # Calcular a escala de Pareto
  escala_pareto = 1 / np.sqrt(sd)
  X_pareto = input_data*escala_pareto
  if mc:
        mean = np.mean(X_pareto, axis=0)
        X_pareto_mc = X_pareto - mean
        return pd.DataFrame(X_pareto_mc)
  else:
        return pd.DataFrame(X_pareto)

def mc(input_data):
      mc = np.mean(input_data, axis=0) #poderia omitir o with_mean=True pq ele é padrão
      X_mc = input_data - mc
      return pd.DataFrame(X_mc)

def auto_scaling(input_data):
      auto_sc = StandardScaler(with_mean=True, with_std=True) #poderia omitir o with_mean=True pq ele é padrão
      X_sc = auto_sc.fit_transform(input_data)
      return pd.DataFrame(X_sc)


