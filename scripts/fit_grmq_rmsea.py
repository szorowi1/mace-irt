import os, sys
import numpy as np
from numba import njit
from os.path import dirname
from itertools import combinations
from pandas import DataFrame, read_csv
from tqdm import tqdm
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
stan_model = 'grmq'
studies = ['teicher2015','tuominen2022']
q_matrices = [1,2,3,4]

## RMSEA paraeters.
thin = 10

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main loop.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Preallocate space.
RMSEA = dict()

for study, q_matrix in [(study, q_matrix) for study in studies for q_matrix in q_matrices]:

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
    n_subj = J.max() + 1

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

    ## Iterate over samples.
    for n in tqdm(range(N)):

        ## Compute linear predictor.
        mu = np.einsum('nd,nd->n', theta[:,J[n]], alpha[:,K[n]])

        ## Compute p(response).
        p = ordinal_pmf(mu, cutpoints[K[n]])

        ## Simulate data.
        Y_hat[:,n] = ordinal_rng(p)
        
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Discrepancy measure: BMRSEA.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    ## Compute number of response levels per item.
    lvls = mace.groupby('item').response.nunique().values

    ## Compute start point per item.
    idxs = np.append(0, np.cumsum(lvls[:-1]))
    lmax = lvls[-1] + idxs[-1]

    ## Compute cell means (observed).
    C_obs = np.zeros((lmax, lmax))
    for j in np.unique(J):
        for a, b in combinations(idxs[K[J==j]] + Y[J==j], 2):
            C_obs[a,b] += 1
    C_obs = C_obs[np.triu_indices_from(C_obs, k=1)] / n_subj

    ## Compute degrees of freedom.
    df = C_obs.size - Q.sum()
    
    ## Compute RMSEA.
    rmsea = []
    for i in tqdm(np.arange(0, n_samp, thin)):

        ## Compute cell means (one sample).
        C_hat = np.zeros((lmax, lmax))
        for j in np.unique(J):
            for a, b in combinations(idxs[K[J==j]] + Y_hat[i,J==j].astype(int), 2):
                C_hat[a,b] += 1
        C_hat = C_hat[np.triu_indices_from(C_hat, k=1)] / n_subj

        ## Compute Pearson's X2.
        x2 = n_subj * np.sum(np.divide(np.square(C_obs - C_hat), C_hat, where=C_hat > 0))
        
        ## Compute RMSEA.
        rmsea.append( np.sqrt((x2 - df) / (df * (n_subj - 1))) )
        
    ## Store.
    RMSEA[f'{study}_{q_matrix}'] = rmsea
    
## Convert to DataFrame.
RMSEA = DataFrame(RMSEA)

## Save.
RMSEA.to_csv(os.path.join(ROOT_DIR, 'stan_results', 'RMSEA.csv'), index=False)