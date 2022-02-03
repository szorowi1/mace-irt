import os, sys
import numpy as np
from os.path import dirname
from pandas import read_csv
from cmdstanpy import CmdStanModel
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
stan_model = '2plq'
study = sys.argv[1]
q_matrix = int(sys.argv[2])

## Sampling parameters.
iter_warmup   = 3000
iter_sampling = 1000
chains = 4
thin = 1
parallel_chains = 4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
mace = read_csv(os.path.join(ROOT_DIR, 'data', 'mace.csv'))

## Apply rejections.
covariates = read_csv(os.path.join(ROOT_DIR, 'data', 'covariates.csv')).query('infreq <= 0.5')
mace = mace.loc[mace.subject.isin(covariates.subject)]

## Restrict to specified dataset.
if study == "teicher2015": mace = mace.query('study == "teicher2015"')
if study == "tuominen2022": mace = mace.query('study == "tuominen2022"')

## Prepare MACE data.
mace = mace[mace.response.notnull()]

## Load design data.
design = read_csv(os.path.join(ROOT_DIR, 'data', 'design.csv'), index_col=0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Assemble data for Stan.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define metadata.
J = np.unique(mace['subject'], return_inverse=True)[-1] + 1
K = np.unique(mace['item'], return_inverse=True)[-1] + 1
N = len(J)

## Define response data.
Y = mace.response.values.astype(int)

## Define Q-matrix.
if q_matrix == 1: 
    cols = ['general']
elif q_matrix == 2: 
    cols = ['general', 'EN', 'NVEA', 'PN', 'PPhysA', 'PVA', 'PeerPhysA', 'PeerVA', 'SexA', 'WIPV', 'WSV']
elif q_matrix == 3:
    cols = ['general', 'peer', 'reverse']
else:
    raise ValueError(f'Invalid input: q_matrix == {q_matrix}')
    
Q = design[cols].values.astype(int)
M = Q.shape[-1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Fit Stan Model.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Assemble data.
dd = dict(N=N, M=M, J=J, K=K, Y=Y, Q=Q)

## Load StanModel
StanModel = CmdStanModel(stan_file=os.path.join('stan_models', f'{stan_model}.stan'))

## Fit Stan model.
StanFit = StanModel.sample(data=dd, chains=chains, iter_warmup=iter_warmup, iter_sampling=iter_sampling, thin=thin, parallel_chains=parallel_chains, seed=0, show_progress=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save samples.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print('Saving data.')
    
## Define fout.
fout = os.path.join(ROOT_DIR, 'stan_results', joint, f'{stan_model}_m{q_matrix}')
    
## Extract summary and samples.
samples = StanFit.draws_pd()
summary = StanFit.summary()
    
## Define columns to save.
cols = np.concatenate([
    
    ## Diagnostic variables.
    samples.filter(regex='__').columns,
    
    ## Subject abilities (orthogonalized).
    samples.filter(regex='theta\[').columns,
    
    ## Item discriminations (post-transform).
    [f'alpha[{i+1},{j+1}]' for i,j in np.column_stack([np.where(Q)]).T],
    
    ## Item difficulties.
    samples.filter(regex='beta\[').columns,
    
    ## Item loadings (standardized).
    [f'lambda[{i+1},{j+1}]' for i,j in np.column_stack([np.where(Q)]).T],
    
    ## Item thresholds (standardized).
    samples.filter(regex='tau').columns
    
])
    
## Save.
samples[cols].to_csv(f'{fout}.tsv.gz', sep='\t', index=False, compression='gzip')
summary.loc[samples[cols].filter(regex='[^_]$').columns].to_csv(f'{fout}_summary.tsv', sep='\t')