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
stan_model = 'grmq'
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

## Load design data.
design = read_csv(os.path.join(ROOT_DIR, 'data', 'design.csv'), index_col=0)

## Define Q-matrix.
if q_matrix == 1: 
    cols = ['general']
elif q_matrix == 2: 
    cols = ['general', 'EN', 'NVEA', 'PN', 'PA', 'VA', 'PeerPA', 'PeerVA', 'SA', 'WIPV', 'WSV']
elif q_matrix == 3:
    cols = ['general', 'neglect']
elif q_matrix == 4:
    cols = ['general', 'peer', 'reverse']
else:
    raise ValueError(f'Invalid input: q_matrix == {q_matrix}')
    
design = design[cols]
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Merge dependent items.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define locally dependent items.
ld = [[6,7,8], [9,10,11], [13,14], [15,16], [19,20], [21,22,23], [24,25], [33,34,35], [36,37]]

## Convert items to pivot table.
mace = mace.pivot_table('response', 'subject', 'item')

## Iterate over tuples.
for ix in ld:
    
    ## Sum across items.
    mace[ix[0]] = mace[ix].sum(axis=1, skipna=False)
    
    ## Drop vestigial columns / rows.
    mace = mace.drop(columns=ix[1:])
    design = design.drop(index=ix[1:])
    
## Unstack data.
mace = mace.unstack().reset_index(name='response')
mace = mace.sort_values(['item','subject'])

## Drop missing data.
mace = mace.dropna()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Assemble data for Stan.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define metadata.
N = len(mace)
M = len(design.T)
K = int(mace.item.nunique())
J = np.unique(mace.subject, return_inverse=True)[-1] + 1

## Define response data.
Y = mace.response.values.astype(int) + 1
    
## Define Q-matrix.
Q = design.values.astype(int)

## Define prior counts.
C = np.row_stack(mace.groupby('item').response.apply(np.bincount, minlength=Y.max()))
s = np.unique(np.where(C)[0], return_counts=True)[-1]
r = mace.item.value_counts(sort=False).values.astype(int)
C = C[np.where(C)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Fit Stan Model.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Assemble data.
dd = dict(N=N, J=J, K=K, M=M, Y=Y, Q=Q, C=C, r=r, s=s)

## Load StanModel
StanModel = CmdStanModel(stan_file=os.path.join('stan_models', f'{stan_model}.stan'))

## Fit Stan model.
StanFit = StanModel.sample(data=dd, chains=chains, iter_warmup=iter_warmup, iter_sampling=iter_sampling, thin=thin, parallel_chains=parallel_chains, seed=0, show_progress=True)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save samples.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print('Saving data.')
    
## Define fout.
fout = os.path.join(ROOT_DIR, 'stan_results', study, f'{stan_model}_m{q_matrix}')
    
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
    samples.filter(regex='tau_pr\[').columns,
    
    ## Item loadings (standardized).
    [f'lambda[{i+1},{j+1}]' for i,j in np.column_stack([np.where(Q)]).T],
    
])
    
## Save.
samples[cols].to_csv(f'{fout}.tsv.gz', sep='\t', index=False, compression='gzip')
summary.loc[samples[cols].filter(regex='[^_]$').columns].to_csv(f'{fout}_summary.tsv', sep='\t')