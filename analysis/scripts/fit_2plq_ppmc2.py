import os, sys
import numpy as np
from os.path import dirname
from pandas import DataFrame, read_csv
from arviz import hdi
from numba import njit
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

## Apply rejections.
covariates = read_csv(os.path.join(ROOT_DIR, 'data', 'covariates.csv')).query('infreq <= 0.5')
mace = mace.loc[mace.subject.isin(covariates.subject)]

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

@njit
def sokalmichener(u,v):
    return np.mean(u==v)

@njit
def srmr(u,v):
    return np.sqrt( (2 * np.square(u - v).sum()) / u.size )

## Define indices.
indices = np.tril_indices(mace.item.nunique(), k=-1)
L = len(indices[0])

## Compute correlations (observed).
SK = mace.pivot_table('response', 'subject', 'item').corr(sokalmichener).values[indices]

## Preallocate space.
SKr = np.zeros((n_samp, L))

## Iterate over samples.
for i in tqdm(range(n_samp)):
    
    ## Compute linear predictor
    mu = np.einsum('nd,nd->n',theta[i,J],alpha[i,K]) - beta[i,K]
    
    ## Compute p(endorse).
    p = inv_logit(mu)
    
    ## Simulate data.
    mace['y_hat'] = np.random.binomial(1, p)
    
    ## Compute correlations.
    SKr[i] = mace.pivot_table('y_hat', 'subject', 'item').corr(sokalmichener).values[indices]

## Compute lower/upper bounds of null distribution.
lb, ub = np.apply_along_axis(hdi, 0, SKr, hdi_prob=0.95)
    
## Compute pvals.
pvals = (SK >= SKr).mean(axis=0)
    
## Compute average correlations.
ESK = np.mean(SKr, axis=0)
    
## Convert to DataFrame.
df = DataFrame(dict(
    k1 = indices[0].astype(int),
    k2 = indices[1].astype(int),
    sk = SK,
    lb = lb, 
    ub = ub,
    pval = pvals
))

## Compute standardized root mean square residual.
SRMR  = srmr(SK, ESK)
SRMRr = np.array([srmr(u, ESK) for u in SKr])

## Compute lower/upper bounds of null distribution.
lb, ub = hdi(SRMRr, hdi_prob=0.95)

## Compute ppp-val.
pppval = (SRMR >= SRMRr).mean()

## Insert summary row.
df = df.append({'k1': 0, 'k2': 0, 'sk': np.round(SRMR, 6), 'lb':lb, 'ub':ub, 'pval': pppval}, ignore_index=True)
df = df.sort_values(['k1','k2'])

## Format data.
df[['k1','k2']] = df[['k1','k2']].astype(int)
df[['sk','lb','ub','pval']] = df[['sk','lb','ub','pval']].round(6)

## Save.
df.to_csv(os.path.join(ROOT_DIR, 'stan_results', f'{stan_model}_m{q_matrix}_ppmc2.csv'), index=False)