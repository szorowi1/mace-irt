import os, sys
import numpy as np
from numba import njit
from os.path import dirname
from pandas import DataFrame, read_csv, concat
from tqdm import tqdm
# ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))
ROOT_DIR = '.'

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
    n_subj = J.max() + 1
    n_item = K.max() + 1
    
    ## Define response data.
    Y = mace.response.values.astype(int)

    ## Define prior counts.
    C = np.row_stack(mace.groupby('item').response.apply(np.bincount, minlength=Y.max() + 1))
    s = np.unique(np.where(C)[0], return_counts=True)[-1]
    r = mace.item.value_counts(sort=False).values.astype(int)
    C = C[np.where(C)]
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Load Stan fit (baseline model).
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
    f = os.path.join(ROOT_DIR, 'stan_results', study, f'baseline.tsv.gz')
    samples = read_csv(f, sep='\t', compression='gzip')
    n_samp = len(samples)

    ## Extract parameters.
    tau = samples.filter(regex='tau').values

    ## Organize item cutpoints.
    pos = 0
    cutpoints = []
    for k in range(K.max()+1):    
        pi = make_simplex(tau[:,pos:pos+s[k]-1])
        cutpoints.append(make_cutpoints(pi))
        pos += s[k]-1

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### Simulate data (baseline model).
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    print(f'Simulating data (baseline).')
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
        mu = np.zeros(n_samp)

        ## Compute p(response).
        p = ordinal_pmf(mu, cutpoints[K[n]])

        ## Simulate data.
        Y_hat[:,n] = ordinal_rng(p)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    ### SEM indices (baseline).
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
    @njit
    def binprop(arr):
        c = np.bincount(arr, minlength=4)
        return c / np.sum(c)

    def numpy_combinations(x):
        a, b = np.triu_indices(len(x), k=1)
        return (x[a], x[b])

    ## Binarize data.
    Y = np.where(Y > 1, 1, Y).astype(int)
    Y_hat = np.where(Y_hat > 1, 1, Y_hat).astype(int)

    ## Compute expected response.
    Y_exp = np.mean(Y_hat, axis=0)

    ## First-order marginal residuals.
    C1_obs, C1_null_hat = [], []
    for k in range(n_item):

        ## Define item indices.
        ix, = np.where(K == k)
        
        ## Compute cell proportions (observed).
        prob_obs = Y[ix].mean()
        C1_obs.append(prob_obs)

        ## Compute cell proportions (predicted).
        prop_hat = Y_hat[::thin,ix].mean(axis=1)
        C1_null_hat.append(prop_hat)

    ## Second-order marginal residuals.
    C2_obs, C2_null_hat = [], []
    for k1, k2 in tqdm(np.column_stack(numpy_combinations(np.arange(n_item)))):

        ## Define item indices. 
        ix1, = np.where(K == k1)
        ix2, = np.where(K == k2)

        ## Error catching: unequal number of responses.
        while ix1.size != ix2.size:
            if ix1.size > ix2.size: ix1 = ix1[np.in1d(J[ix1], J[ix2])]
            elif ix2.size > ix1.size: ix2 = ix2[np.in1d(J[ix2], J[ix1])]
            
        ## Compute cell proportions (observed).
        prop_obs = binprop(Y[ix1] + 2 * Y[ix2])
        C2_obs.append(prop_obs[-1])
            
        ## Compute cell proportions (predicted).
        prop_hat = np.apply_along_axis(binprop, 1, Y_hat[::thin,ix1] + 2 * Y_hat[::thin,ix2])
        C2_null_hat.append(prop_hat[:,-1])

    ## Concatenate arrays.
    C_obs = np.concatenate([C1_obs, C2_obs])
    C_null_hat = np.column_stack([np.stack(C1_null_hat, axis=-1), np.column_stack(C2_null_hat)])
    C_null_exp = C_null_hat.mean(axis=0)

    ## Compute chi-square.
    X2_null = n_subj * np.sum(np.square(C_obs - C_null_exp) / (C_null_exp * (1-C_null_exp)))
    df_null = C_null_exp.size - n_item
    
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
        ### SEM indices.
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
                
        ## Binarize data.
        Y_hat = np.where(Y_hat > 1, 1, Y_hat).astype(int)
        
        ## Compute expected response.
        Y_exp = np.mean(Y_hat, axis=0)

        ## First-order marginal residuals.
        C1_hat = []
        for k in range(n_item):

            ## Define item indices.
            ix, = np.where(K == k)

            ## Compute cell proportions (observed).
            prob_obs = Y[ix].mean()
            C1_obs.append(prob_obs)

            ## Compute cell proportions (predicted).
            prop_hat = Y_hat[::thin,ix].mean(axis=1)
            C1_hat.append(prop_hat)

        ## Second-order marginal residuals.
        C2_hat = []
        for k1, k2 in tqdm(np.column_stack(numpy_combinations(np.arange(n_item)))):

            ## Define item indices. 
            ix1, = np.where(K == k1)
            ix2, = np.where(K == k2)

            ## Error catching: unequal number of responses.
            while ix1.size != ix2.size:
                if ix1.size > ix2.size: ix1 = ix1[np.in1d(J[ix1], J[ix2])]
                elif ix2.size > ix1.size: ix2 = ix2[np.in1d(J[ix2], J[ix1])]

            ## Compute cell proportions (observed).
            prop_obs = binprop(Y[ix1] + 2 * Y[ix2])
            C2_obs.append(prop_obs[-1])

            ## Compute cell proportions (predicted).
            prop_hat = np.apply_along_axis(binprop, 1, Y_hat[::thin,ix1] + 2 * Y_hat[::thin,ix2])
            C2_hat.append(prop_hat[:,-1])

        ## Concatenate arrays.
        C_hat = np.column_stack([np.stack(C1_hat, axis=-1), np.column_stack(C2_hat)])
        C_exp = C_hat.mean(axis=0)

        ## Compute chi-square.
        X2_obs = n_subj * np.sum(np.square(C_obs - C_exp) / (C_exp * (1-C_exp)))
        df_obs = C_obs.size - Q.sum() - n_item
        
        ## Compute root mean square error of approximation (RMSEA).
        rmsea = np.sqrt((X2_obs - df_obs) / (df_obs * (n_subj - 1)))
        
        ## Compute comparative fit index (CFI).
        cfi = 1 - (X2_obs - df_obs) / (X2_null - df_null)
        
        ## Compute Tucker-Lewis index (TLI).
        tli = ((X2_null / df_null) - (X2_obs / df_obs)) / ((X2_null / df_null) - 1)
        
        ## Store indices.
        data.append(dict(
            study = study,
            model = q_matrix,
            x2 = X2_obs,
            df = df_obs,
            rmsea = rmsea,
            cfi = cfi,
            tli = tli
        ))
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Save data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~# 

## Convert to DataFrame.
data = DataFrame(data)

## Format data.
data[['x2','rmsea','cfi','tli']] = data[['x2','rmsea','cfi','tli']].round(6)

## Save.
data.to_csv(os.path.join(ROOT_DIR, 'stan_results', 'X2.csv'), index=False)