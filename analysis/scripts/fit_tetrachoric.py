import os, sys
import numpy as np
from os.path import dirname
from pandas import DataFrame, read_csv
from cmdstanpy import CmdStanModel
from tqdm import tqdm
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
stan_model = 'tetrachoric'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
mace = read_csv(os.path.join(ROOT_DIR, 'data', 'mace.csv'))

## Apply rejections.
covariates = read_csv(os.path.join(ROOT_DIR, 'data', 'covariates.csv')).query('infreq <= 0.5')
mace = mace.loc[mace.subject.isin(covariates.subject)]

## Pivot data.
mace = mace.pivot_table('response', ['study','subject'], 'item')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Compute tetrachoric correlations.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load StanModel
StanModel = CmdStanModel(stan_file=os.path.join('stan_models', f'{stan_model}.stan'))

## Define indices.
indices = np.tril_indices(mace.columns.size, k=-1)

## Main loop.
tetrachoric = []
for i, j in tqdm(np.column_stack(indices)):
    
    ## Preallocate space.
    row = dict(k1=i, k2=j)
    
    ## Compute tetrachoric correlation (teicher2015).
    df = mace.loc['teicher2015', [i+1,j+1]].dropna()
    dd = dict(J=df.shape[0], Y=df.values.astype(int))
    StanFit = StanModel.optimize(data=dd, seed=0)
    row['teicher2015'] = float(StanFit.optimized_params_pd['rho'])
    
    ## Compute tetrachoric correlation (tuominen2022).
    df = mace.loc['tuominen2022', [i+1,j+1]].dropna()
    dd = dict(J=df.shape[0], Y=df.values.astype(int))
    StanFit = StanModel.optimize(data=dd, seed=0)
    row['tuominen2022'] = float(StanFit.optimized_params_pd['rho'])
    
    ## Compute tetrachoric correlation (joint).
    df = mace.loc[:,[i+1,j+1]].dropna()
    dd = dict(J=df.shape[0], Y=df.values.astype(int))
    StanFit = StanModel.optimize(data=dd, seed=0)
    row['joint'] = float(StanFit.optimized_params_pd['rho'])
    
    ## Append.
    tetrachoric.append(row)
    
## Convert to DataFrame.
tetrachoric = DataFrame(tetrachoric)

## Save.
tetrachoric.to_csv(os.path.join(ROOT_DIR, 'stan_results', 'tetrachoric.csv'), index=False)