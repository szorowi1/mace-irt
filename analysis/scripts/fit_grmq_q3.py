import os, sys
import numpy as np
from numba import njit
from os.path import dirname
from pandas import DataFrame, read_csv, concat
from tqdm import tqdm
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
stan_model = 'grmq'
studies = ['teicher2015','tuominen2022']
q_matrices = [1,2,3,4]

## Fit paraeters.
thin = 4

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main loop.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

data = []
for study in studies:

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
    K = np.unique(mace.item, return_inverse=True)[-1]
    J = np.unique(mace.subject, return_inverse=True)[-1]

    ## Define response data.
    Y = mace.response.values.astype(int)

    ## Define prior counts.
    C = np.row_stack(mace.groupby('item').response.apply(np.bincount, minlength=Y.max() + 1))
    s = np.unique(np.where(C)[0], return_counts=True)[-1]
    r = mace.item.value_counts(sort=False).values.astype(int)
    C = C[np.where(C)]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Inner loop.
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    for q_matrix in q_matrices: 

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        ### Prepare design matrix.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

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

        ## Define Q-matrix.
        Q = design[cols].values.astype(int)
        M = len(Q.T)

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
        print(f'Simulating data (m{q_matrix}).')
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
        ### Compute Q3 indices.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#   
        print('Computing Q3 indices.')
        
        ## Define indices.
        ix = np.tril_indices(K.max() + 1, k=-1)
        
        ## Binarize data.
        Y = np.where(Y > 1, 1, Y)
        Y_hat = np.where(Y_hat > 1, 1, Y_hat)
        
        ## Compute expected response.
        Y_exp = np.mean(Y_hat, axis=0)
        
        ## Compute Q3 indices (observed).
        mace['R'] = Y - Y_exp
        Q3_obs = mace.pivot_table('R', 'subject', 'item').corr().values[ix]
        
        ## Compute Q3_star indices (observed).
        Q3_obs_star =  Q3_obs - np.mean(Q3_obs)
        
        ## Compute Q3 indices (simulated).
        Q3_max_null, Q3_star_max_null = [], []
        for i in tqdm(range(0, n_samp, thin)):
            
            ## Compute residuals.
            mace['R'] = Y_hat[i] - Y_exp
            
            ## Comute Q3 indices.
            Q3_null = mace.pivot_table('R', 'subject', 'item').corr().values[ix]
            
            ## Normalize.
            Q3_star_null = Q3_null - np.mean(Q3_null)
            
            ## Compute max value.
            Q3_max_null.append(Q3_null.max()) 
            Q3_star_max_null.append(Q3_star_null.max())
            
        ## Compute critical threshold.
        Q3_95 = np.percentile(Q3_max_null, 95)
        Q3_99 = np.percentile(Q3_max_null, 99)
        Q3_star_95 = np.percentile(Q3_star_max_null, 95)
        Q3_star_99 = np.percentile(Q3_star_max_null, 99)
        
        ## Convert to DataFrame.
        df = DataFrame(dict(
            study = np.repeat(study, Q3_obs.size),
            model = np.repeat(q_matrix, Q3_obs.size),
            k1 = ix[0],
            k2 = ix[1],
            Q3 = Q3_obs.round(6),
            Q3_95 = np.repeat(Q3_95, Q3_obs.size).round(6),
            Q3_99 = np.repeat(Q3_99, Q3_obs.size).round(6),
            Q3_star = Q3_obs_star.round(6),
            Q3_star_95 = np.repeat(Q3_star_95, Q3_obs.size).round(6),
            Q3_star_99 = np.repeat(Q3_star_99, Q3_obs.size).round(6),
        ))
        data.append(df)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 

## Convert to DataFrame.
data = concat(data)

## Save.
data.to_csv(os.path.join(ROOT_DIR, 'stan_results', 'Q3.csv'), index=False)