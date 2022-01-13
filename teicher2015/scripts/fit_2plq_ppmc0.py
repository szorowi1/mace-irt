import os, sys
import numpy as np
from os.path import dirname
from pandas import DataFrame, read_csv
from numba import njit
from psis import psisloo
from tqdm import tqdm
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
stan_model = '2plq'
q_matrix = int(sys.argv[1])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
mace = read_csv(os.path.join(ROOT_DIR, 'data', 'mace.csv'))

## Prepare MACE data.
mace = mace.dropna()

## Load design data.
design = read_csv(os.path.join(ROOT_DIR, 'data', 'design.csv'), index_col=0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Assemble data for Stan.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define metadata.
J = np.unique(mace['subject'], return_inverse=True)[-1]
K = np.unique(mace['item'], return_inverse=True)[-1]
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
### Load Stan fit.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load samples.
f = os.path.join(ROOT_DIR, 'stan_results', f'{stan_model}_m{q_matrix}.tsv.gz')
samples = read_csv(f, sep='\t', compression='gzip')
n_samp = len(samples)

## Extract parameters.
theta = samples.filter(regex='theta').values.reshape(-1,J.max()+1,M)
beta = samples.filter(regex='beta').values
alpha = np.zeros((n_samp, K.max()+1, M))

for i, j in np.column_stack([np.where(Q)]).T:
    alpha[:,i,j] = samples[f'alpha[{i+1},{j+1}]'].values
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Posterior predictive check.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
np.random.seed(47404)

@njit
def inv_logit(x):
    return 1. / (1 + np.exp(-x))

## Preallocate space.
Y_hat  = np.zeros((n_samp, N))
Y_pred = np.zeros((n_samp, N))

## Iterate over samples.
for i in tqdm(range(n_samp)):
    
    ## Compute linear predictor
    mu = np.einsum('nd,nd->n',theta[i,J],alpha[i,K]) - beta[i,K]
    
    ## Compute p(endorse).
    p = inv_logit(mu)
    
    ## Simulate data.
    Y_hat[i] = np.random.binomial(1, p)
    
    ## Compute likelihood.
    Y_pred[i] = np.where(mace.response, p, 1-p)
    
## Append average accuracy / likelihood.
mace['y_hat']  = Y_hat.mean(axis=0).round(6)
mace['y_pred'] = Y_pred.mean(axis=0).round(6)

## Compute PSIS-LOO.
louo, louos, ku = psisloo(np.log(Y_pred))
mace['loo'] = louos.round(6)
mace['ku'] = ku.round(6)

## Save.
mace.to_csv(os.path.join(ROOT_DIR, 'stan_results', f'{stan_model}_m{q_matrix}_ppmc0.csv'), index=False)