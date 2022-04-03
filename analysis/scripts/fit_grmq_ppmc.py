import os, sys
import numpy as np
from os.path import dirname
from pandas import DataFrame, read_csv
from psis import psisloo
from numba import njit
from tqdm import tqdm
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
stan_model = 'grmq'
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

## Load design data.
design = read_csv(os.path.join(ROOT_DIR, 'data', 'design.csv'), index_col=0)

## Define Q-matrix.
if q_matrix == 1: 
    cols = ['general']
elif q_matrix == 2: 
    cols = ['general', 'EN', 'NVEA', 'PN', 'PPhysA', 'PVA', 'PeerPhysA', 'PeerVA', 'SexA', 'WIPV', 'WSV']
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

## Re-number items. 
mace['item'] = np.unique(mace.item, return_inverse=True)[-1]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Assemble data for Stan.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define metadata.
N = len(mace)
M = len(design.T)
K = np.unique(mace.item, return_inverse=True)[-1]
J = np.unique(mace.subject, return_inverse=True)[-1]

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
### Load Stan fit.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

@njit 
def logit(p):
    return np.log(p / (1-p))
    
def make_simplex(gamma):
    gamma = np.column_stack([gamma, np.ones(len(gamma), dtype=np.float64)])
    return (gamma.T / np.sum(gamma, axis=1)).T

def make_cutpoints(pi):
    return logit(np.cumsum(pi[:,:-1], axis=1))    

## Load samples.
f = os.path.join(ROOT_DIR, 'stan_results', study, f'{stan_model}_m{q_matrix}.tsv.gz')
samples = read_csv(f, sep='\t', compression='gzip')
n_samp = len(samples)

## Extract parameters.
theta = samples.filter(regex='theta').values.reshape(-1, M, J.max()+1).swapaxes(1,2)
tau = samples.filter(regex='tau').values
alpha = np.zeros((n_samp, K.max()+1, M))

## Organize item discrimination parameters.
for i, j in np.column_stack([np.where(Q)]).T:
    alpha[:,i,j] = samples[f'alpha[{i+1},{j+1}]'].values
    
## Organize item cutpoints.
pos = 0
cutpoints = []
for k in range(K.max()+1):    
    pi = make_simplex(tau[:,pos:pos+s[k]-1])
    cutpoints.append(make_cutpoints(pi))
    pos += s[k]-1
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Simulate data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
np.random.seed(47404)

@njit
def inv_logit(x):
    return 1. / (1 + np.exp(-x))

def ordinal_pmf(eta, cutpoints):
    p = inv_logit(cutpoints - eta[:,np.newaxis])
    p = np.column_stack([np.zeros_like(eta), p, np.ones_like(mu)])
    return np.diff(p, axis=1)

def ordinal_rng(p):
    random = np.random.random((len(p), 1))
    thresh = np.cumsum(p[:,:-1], axis=1)
    return np.sum(random > thresh, axis=1)

## Extract observed responses.
Y = mace.response.values.astype(int)

## Preallocate space.
Y_hat = np.zeros((n_samp, N))
Y_pred = np.zeros((n_samp, N))
R_obs = np.zeros((n_samp, N))
R_hat = np.zeros((n_samp, N))

## Iterate over samples.
for n in tqdm(range(N)):
    
    ## Compute linear predictor.
    mu = np.einsum('nd,nd->n', theta[:,J[n]], alpha[:,K[n]])
    
    ## Compute p(response).
    p = ordinal_pmf(mu, cutpoints[K[n]])
    
    ## Compute expectation.
    expected = p @ np.arange(p.shape[1])
    
    ## Simulate data.
    Y_hat[:,n] = ordinal_rng(p)
    
    ## Compute likelihood.
    Y_pred[:,n] = p[:,Y[n]]
    
    ## Compute residuals (observed).
    R_obs[:,n] = Y[n] - expected
    
    ## Compute residuals (simulated).
    R_hat[:,n] = Y_hat[:,n] - expected
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Model comparison.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print('Computing LOO-CV.')

## Append average accuracy / likelihood.
mace['y_hat'] = Y_hat.mean(axis=0).round(6)
mace['y_pred'] = Y_pred.mean(axis=0).round(6)

## Compute PSIS-LOO.
louo, louos, ku = psisloo(np.log(Y_pred))
mace['loo'] = louos.round(6)
mace['ku'] = ku.round(6)

## Save.
mace.to_csv(os.path.join(ROOT_DIR, 'stan_results', study, f'{stan_model}_m{q_matrix}_ppmc0.csv'), index=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Discrepancy measure: observed scores.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print('Computing discrepancy (x2).')

## Define score range.
minlength = np.sum(s-1) + 1

## Compute simulated scores.
S_hat = np.zeros((n_samp, J.max() + 1), dtype=int)
for j in np.unique(J): S_hat[:,j] = Y_hat[:,J==j].sum(axis=1)

## Compute simulated counts.
NC = np.apply_along_axis(np.bincount, 1, S_hat, minlength=minlength)

## Compute observed counts.
scores = mace.groupby('subject').response.sum()
counts = np.bincount(scores, minlength=minlength)

## Convert to DataFrame.
NC = DataFrame(np.row_stack([counts, NC]))

## Save.
NC.index = NC.index.rename('sample')
NC.to_csv(os.path.join(ROOT_DIR, 'stan_results', study, f'{stan_model}_m{q_matrix}_ppmc1.csv'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Discrepancy measure: SGDDM.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print('Computing discrepancy (sgddm).')

@njit
def smbc(u, v):
    return (u @ v) / (np.sqrt(u @ u) * np.sqrt(v @ v))

## Define indices.
indices = np.tril_indices(mace.item.nunique(), k=-1)
n_pairs = len(indices[0])

## Preallocate space.
summary = np.zeros((n_pairs, 3))
sgddm = np.zeros(3)

## Iteratively compute SGGDM.
for i in tqdm(range(n_samp)):
    
    ## Compute SMBC (observed).
    mace['r'] = R_obs[i]
    obs = mace.pivot_table('r','subject','item').corr(smbc).values[indices]
    summary[:,0] += obs

    ## Compute SMBC (simulated).
    mace['r'] = R_hat[i]
    hat = mace.pivot_table('r','subject','item').corr(smbc).values[indices]
    summary[:,1] += hat
        
    ## Compute ppp-values.
    summary[:,2] += (obs > hat).astype(int)
        
    ## Compute SGDDM (observed). 
    obs = np.abs(obs).mean()
    sgddm[0] += obs
    
    ## Compute SGDDM (simulated). 
    hat = np.abs(hat).mean()
    sgddm[1] += hat
    
    ## Compute ppp-values.
    sgddm[2] += (obs > hat).astype(int)
    
## Normalize values.
summary /= n_samp
sgddm /= n_samp

## Convert to DataFrame.
df = DataFrame(np.row_stack([sgddm, summary]), columns=['obs', 'pred', 'pval'])
df.insert(0, 'k2', np.append(0, indices[1]))
df.insert(0, 'k1', np.append(0, indices[0]))

## Save.
df.to_csv(os.path.join(ROOT_DIR, 'stan_results', study, f'{stan_model}_m{q_matrix}_ppmc2.csv'), index=False)