import os
import numpy as np
from os.path import dirname
from pandas import read_csv
ROOT_DIR = '.'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
raw = read_csv(os.path.join(ROOT_DIR, 'raw', 'pone.0117423.s018.csv'))

## Set index.
raw = raw.rename(columns={'Username':'subject'}).set_index('subject')
raw.index = [s.lower() for s in raw.index]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Prepare MACE data.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load mapping file.
mapping = read_csv(os.path.join(ROOT_DIR, 'raw', 'mapping.csv'))

## Limit to MACE items & melt.
mace = raw[mapping.item_id].copy()
mace = mace.melt(var_name='item_id', value_name='response', ignore_index=False)

## Merge with mapping.
mace = mace.assign(subject = mace.index).merge(mapping, on='item_id')

## Re-organize columns.
cols = ['subject','item','item_id','subscale','response']
mace = mace[cols].sort_values(['subject','item'])

## Reverse score items.
reverse = [42, 43, 44, 45, 51, 52]
mace.response = np.where(np.in1d(mace.item, reverse), 1-mace.response, mace.response)

## Save data.
mace.to_csv(os.path.join(ROOT_DIR, 'data', 'mace.csv'), index=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Prepare covariates.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define columns of interest.
cols = ['Gender','Age','race','ethnicity','sibs','SQ_Anx','SQ_Dep','SQ_Som','SQ_Hos',
        'DES_SCORE','LSCL33','ASIQ_tot']

## Restrict to columns of interest.
covariates = raw[cols].copy().reset_index()

## Format columns.
covariates = covariates.rename(columns={
    'index':'subject',
    'Gender':'gender',
    'Age':'age',
    'sibs':'siblings',
    'SQ_Anx':'anxiety',
    'SQ_Dep':'depression',
    'SQ_Som':'somatic',
    'SQ_Hos':'hostility',
    'DES_SCORE':'dissociation',
    'LSCL33':'irritability',
    'ASIQ_tot':'suicidality' 
})
covariates.gender = covariates.gender.replace({'M':0, 'F':1})

## Save data.
covariates.to_csv(os.path.join(ROOT_DIR, 'data', 'covariates.csv'), index=False)