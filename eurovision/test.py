import stan
import arviz as az
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import math
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df_past = pd.read_csv('eurovision/df_main.csv')
def format_votes(x):
  if x == 12.:
    return 10
  elif x == 10.:
    return 9
  return int(x)
df_past['indexed_votes'] = df_past['points'].apply(format_votes) + 1

df_future = pd.read_csv('eurovision/df_2023.csv')
df = df_past.append(df_future,ignore_index=True, verify_integrity=True)

# Given gender is a categoric variable with 3 classes, encode as binary w.r.t default gender='group'
df['male'] = [1 if gender=='male' else 0 for gender in df['gender']]
df['female'] = [1 if gender=='female' else 0 for gender in df['gender']]

# Evaluate binary variables for boolean covariates to be used
df['Contains_English_bin'] = df['Contains_English'].apply(lambda x: 1 if x else 0)
df['Contains_Own_Language_bin'] = df['Contains_Own_Language'].apply(lambda x: 1 if x else 0)

# build vector of voter/performer pair indicies and corresponding lookup tables
performers = sorted(df['to_code2'].unique())
voters = sorted(df['from_code2'].unique())

# # create 0-indexed lookup tables for voter-performer pairings
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

df['vp'] = df.apply(lambda x: vptoi[f'{x["from_code2"]}-{x["to_code2"]}'], axis=1)

# test/train split
df_train = df.loc[ df['year'] <= 2022 ]
df_test = df.loc[ df['year'] > 2022 ]

# recast 'indexed_votes' to int (they were cast to float during df.append)
df_train['indexed_votes'] = df_train['indexed_votes'].apply(lambda x:int(x))

# build xbeta matrix
xbeta_train = df_train.loc[:,['Contains_English_bin','Contains_Own_Language_bin','male','female','comps_without_win']].values
# minmax scaling of 'comps_since_last_win'
scaler = MinMaxScaler() 
xbeta_train_norm = scaler.fit_transform(xbeta_train)


with open('eurovision/generated_objects/vptoi.json') as f:
    oldvptoi = f.read()
oldvptoi = json.loads(oldvptoi)

print(set(vptoi.keys()) - set(oldvptoi.keys()))

