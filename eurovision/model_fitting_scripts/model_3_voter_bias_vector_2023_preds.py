# This script fits the 'model_3_voter_bias_vector' model on all of the training data
# which is 1998-2022. 

# The fitted scipy data scaler is exported for re-use with new covariates during out
# of sample prediction.

# The set of mcmc samples are exported to be used for inference later on.


import stan
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import json

df = pd.read_csv('eurovision/df_main.csv')
def format_votes(x):
  if x == 12.:
    return 10
  elif x == 10.:
    return 9
  return int(x)
df['indexed_votes'] = df['points'].apply(format_votes) + 1

# Given gender is a categoric variable with 3 classes, encode as binary w.r.t default gender='group'
df['male'] = [1 if gender=='male' else 0 for gender in df['gender']]
df['female'] = [1 if gender=='female' else 0 for gender in df['gender']]

# Evaluate binary variables for boolean covariates to be used
df['Contains_English_bin'] = df['Contains_English'].apply(lambda x: 1 if x else 0)
df['Contains_Own_Language_bin'] = df['Contains_Own_Language'].apply(lambda x: 1 if x else 0)

# build vector of voter/performer pair indicies and corresponding lookup tables
performers = sorted(df['to_code2'].unique())
voters = sorted(df['from_code2'].unique())

# create 0-indexed lookup tables for voter-performer pairings
vptoi = {}
itovp = {}
counter = 0
for p in performers:
    for v in voters:
        # only include v-p pairs that have occured in the data, different countries were voting and performing in different years
        if ( p != v ) and ( not df.loc[ (df['from_code2'] == v) & (df['to_code2'] == p) ].empty ):
            vptoi[f'{v}-{p}'] = counter
            itovp[f'{counter}'] = f'{v}-{p}'
            counter += 1

# save vptoi lookup table to be used in competition simulation
with open('eurovision/generated_objects/vptoi.json', 'w') as f:
  json.dump(vptoi,f)
with open('eurovision/generated_objects/itovp.json', 'w') as f:
  json.dump(itovp,f)

df['vp'] = df.apply(lambda x: vptoi[f'{x["from_code2"]}-{x["to_code2"]}'], axis=1)

# train on all the data
df_train = df.loc[ df['year'] <= 2022 ]

model = """
// overload add function for adding an int to an array of ints
functions {
  array[] int add(array[] int x, int y) {
      int x_size = size(x);
      array[x_size] int z;
      for (i in 1:x_size){
        z[i] = x[i] + y;
      }
      return z;
  }
}
data {
  int<lower=2> S;
  int<lower=0> N;   // total number of performances
  int<lower=1> B;   // number of performance dependent covariates
  int<lower=1> PHI; // number of voter/performer pair dependent covariates
  int<lower=1> VP;  // number of voter/performer combinations

  array[N] int<lower=1, upper=S> y;
  matrix[N,B] xbeta;       // performance dependent covariates
  matrix[VP,PHI] xphi;     // voter/performer pair dependent covariates
  array[N] int<lower=0,upper=VP-1> vp;  // voter/performer pair index

}
parameters {
  vector[B] beta;
  vector[PHI] phi;
  vector[VP] alpha;
  ordered[S-1] lambda;
  real gamma;
  real<lower=0> sigmaAlpha;
}
model {
  gamma ~ normal(0, 10000);
  beta ~ normal(0, 10000);
  lambda ~ normal(0, 3.2);
  sigmaAlpha ~ cauchy(0,1);

  alpha ~ normal( xphi * phi, sigmaAlpha );
  
  // remembering that vp is 0-indexed and alpha is 1-indexed
  y ~ ordered_logistic( gamma + alpha[ add(vp,1) ] + (xbeta * beta), lambda );

}
generated quantities {
  vector[N] y_hat;
  for (n in 1:N) {
    y_hat[n] = ordered_logistic_rng( gamma + alpha[ add(vp[n],1) ] + (xbeta[n] * beta), lambda);
  }
}
"""

# build xbeta matrix
xbeta_train = df_train.loc[:,['Contains_English_bin','Contains_Own_Language_bin','male','female','comps_without_win']].values
# minmax scaling of 'comps_since_last_win'
scaler = MinMaxScaler() 
xbeta_train_norm = scaler.fit_transform(xbeta_train)

# save scaler to scale the unseen 2023 data with!
with open('eurovision/model_output/scaler_model_3_voter_bias_vector_no_preds_3000_samples_4_chains_1998-2022.pkl', 'wb') as f:
  pickle.dump(scaler, f)

df_border = pd.read_csv('eurovision/final_border_data_long.csv')

# build xphi matrix
xphi = np.zeros((len(vptoi), 2))
for pair,idx in vptoi.items():
    v = pair[:2]
    p = pair[-2:]
    has_border = df_border.loc[ (df_border['country_code_1'] == v) & (df_border['country_code_2'] == p) ]['has_border'].item()
    xphi[idx][0] = 0.0 if math.isnan(has_border) else has_border

    migration_series = df.loc[ (df['from_code2'] == v) & (df['to_code2'] == p) ]['prop_emigrants_v2p']
    if migration_series.isnull().any():
        # if no migration data is available, use the plot below to infer the most appropriate substitute value
        # this should be a better alternative to assuming 0 migration
        xphi[idx][1] = math.exp(-9) - 2.6e-08/2
    else:
        mean_migration = migration_series.mean()
        xphi[idx][1] = mean_migration

# standardise the migration intensity feature in xphi with a log transform and standardisation
mig_ints = xphi[:,1]
mig_ints_log = np.log10(mig_ints + 2.6e-08/2)   # rule of thumb, half of smallest non-zero value
std_scaler = StandardScaler()
mig_ints_log_std = std_scaler.fit_transform(mig_ints_log.reshape(-1, 1))

# write scaler version of xphi
xphi_norm = xphi.copy()
xphi_norm[:,1] = mig_ints_log_std.reshape((xphi_norm.shape[0],))

data = {
    'S': 11,
    'N': df_train.shape[0],
    'B': xbeta_train_norm.shape[1],
    'PHI' : xphi_norm.shape[1],
    'VP' : xphi_norm.shape[0],
    'y': df_train['indexed_votes'].values,
    'xbeta': xbeta_train_norm,
    'xphi' : xphi_norm,
    'vp' : df_train['vp'].values
}

posterior = stan.build(model, data=data)
fit = posterior.sample(num_chains=4, num_warmup=1000, num_samples=3000)

az_fit = az.from_pystan(
    posterior=fit, 
    observed_data="y", 
    posterior_predictive="y_hat",
    posterior_model=posterior)

az_fit.to_json("eurovision/model_output/model_3_voter_bias_vector_no_preds_3000_samples_4_chains_1998-2022.json")

az.plot_trace(az_fit, ["beta","lambda"], figsize=(20,8), legend=True, show=True)
# az.plot_trace(az_fit, ["beta"], figsize=(25,35), legend=True, compact=False, show=True)
