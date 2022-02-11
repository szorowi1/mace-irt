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
study = sys.argv[1]
q_matrix = int(sys.argv[2])

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
    cols = ['general', 'peer']
elif q_matrix == 4:
    cols = ['general', 'peer', 'sexual', 'reverse']
else:
    raise ValueError(f'Invalid input: q_matrix == {q_matrix}')
    
Q = design[cols].values.astype(int)
M = Q.shape[-1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load Stan fit.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load samples.
f = os.path.join(ROOT_DIR, 'stan_results', study, f'{stan_model}_m{q_matrix}.tsv.gz')
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
def smbc(u, v):
    return (u @ v) / (np.sqrt(u @ u) * np.sqrt(v @ v))

## Define indices.
indices = np.tril_indices(mace.item.nunique(), k=-1)
L = indices[0].size

## Preallocate space.
stats = np.zeros((n_samp, L + 1))
pvals = np.zeros(L + 1)

for i in tqdm(range(n_samp)):
    
    ## Compute linear predictor
    mu = np.einsum('nd,nd->n', theta[i,J], alpha[i,K]) - beta[i,K]
    
    ## Compute p(endorse).
    p = inv_logit(mu)
    
    ## Compute residuals (observed).
    mace['r'] = mace.response - p
    
    ## Compute SMBC (observed).
    stats[i,1:] = mace.pivot_table('r', 'subject', 'item').corr(smbc).values[indices]
    
    ## Compute SGGDM (observed).
    stats[i,0] = np.abs(stats[i,1:]).mean()
    
    ## Compute residuals (replicated).
    mace['r'] = np.random.binomial(1,p) - p
    
    ## Compute SMBC (replicated).
    SMBCr = mace.pivot_table('r', 'subject', 'item').corr(smbc).values[indices]
    
    ## Compute SGGDM (replicated).
    SGGDMr = np.abs(SMBCr).mean()
    
    ## Increment p-values.
    pvals[1:] += (stats[i,1:] >= SMBCr).astype(float) / n_samp
    pvals[0] += (stats[i,0] >= SGGDMr).astype(float) / n_samp
    
## Compute summary stats.
mu = stats.mean(axis=0)
lb, ub = np.apply_along_axis(hdi, 0, stats, hdi_prob=0.95)

## Convert to DataFrame.
df = DataFrame(dict(
    k1 = np.append(0, indices[0]),
    k2 = np.append(0, indices[1]),
    mu = mu,
    lb = lb, 
    ub = ub,
    pval = pvals
))

## Format data.
df[['k1','k2']] = df[['k1','k2']].astype(int)
df[['mu','lb','ub','pval']] = df[['mu','lb','ub','pval']].round(6)

## Save.
df.to_csv(os.path.join(ROOT_DIR, 'stan_results', study, f'{stan_model}_m{q_matrix}_ppmc2.csv'), index=False)