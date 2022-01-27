import os, sys
import numpy as np
from os.path import dirname
from pandas import DataFrame, concat, read_csv
from statsmodels.api import Logit
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
mace = read_csv(os.path.join(ROOT_DIR, 'data', 'mace.csv'))
covariates = read_csv(os.path.join(ROOT_DIR, 'data', 'covariates.csv'))

## Apply screening.
covariates = covariates.query('infreq <= 0.5')
mace = mace[mace.subject.isin(covariates.query('infreq <= 0.5').subject)]

## Compute observed & rest scores.
zscore = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
mace['score'] = mace.groupby('subject').response.transform(np.nansum)
mace['rest'] = mace['score'] - mace['response'] 
mace['rest'] = zscore(mace.rest)

## Dummy-code group variables.
mace = mace.merge(covariates[['subject','gender','age']], on='subject')
mace['gender'] = mace.gender.replace({'Male': -0.5, 'Female': 0.5, 'Other': 0, 'Rather not say': 0})
mace['study'] = mace.study.replace({'teicher2015':-0.5, 'tuominen2022': 0.5})
mace['age'] = zscore(mace.age)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Fit logistic models.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define formula.
formula = 'response ~ rest + study + gender + age'

## Preallocate space.
dif = []

## Iterate over items.
for item in mace.item.unique():
        
    ## Restrict DataFrame.
    df = mace.query(f'item == {item}')
    df = df[df.response.notnull()]
    
    ## Fit logistic regression model.
    fit = Logit.from_formula(formula, data=df).fit(disp=0)
    
    ## Check convergence.
    if not fit.mle_retvals['converged']: 
        print(item)
        continue
        
    ## Convert to DataFrame.
    df = DataFrame([fit.params, fit.tvalues, fit.pvalues], index=['coef','tval','pval']).T
    df = df.reset_index().rename(columns={'index':'variable'}).round(6)
    df.insert(0, 'item', item)
    
    ## Append.
    dif.append(df)
    
## Concatenate DataFrames.
dif = concat(dif).query('variable != "Intercept"').reset_index(drop=True)

## Save.
dif.to_csv(os.path.join(ROOT_DIR, 'stan_results', 'logdif.csv'), index=False)