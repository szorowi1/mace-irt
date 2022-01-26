import os, json
import numpy as np
from os.path import dirname
from pandas import DataFrame, concat, read_csv
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DIR = os.path.join(ROOT_DIR, 'raw')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Teicher & Parigger (2015)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
teicher2015 = read_csv(os.path.join(ROOT_DIR, 'raw', 'teicher2015', 'pone.0117423.s018.csv'))

## Define columns of interest.
cols = ['Username','Gender','Age','race','ethnicity','sibs','SQ_Anx','SQ_Dep','SQ_Som','SQ_Hos',
        'DES_SCORE','LSCL33','ASIQ_tot']

## Restrict to columns of interest.
teicher2015 = teicher2015[cols].copy()

## Format columns.
teicher2015 = teicher2015.rename(columns={
    'Username':'subject',
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

## Format data.
teicher2015['ethnicity'] = teicher2015.ethnicity.replace({
    'Not Hisp': 'Not Hispanic or Latino',
    'Hisp': 'Hispanic or Latino'
})
teicher2015['race'] = teicher2015.race.replace({
    'Black': 'Black or African American'
})
teicher2015['gender'] = teicher2015.gender.replace({
    'M': 'Male',
    'F': 'Female'
})
teicher2015['age'] = teicher2015.age.round(2)

## Insert study.
teicher2015.insert(0, 'study', 'teicher2015')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Tuominen & Zorowitz (2022)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Locate files.
fdir = os.path.join(ROOT_DIR, 'raw', 'tuominen2022')
files = sorted([f for f in os.listdir(fdir) if f.endswith('.json') and not 'siblings' in f])

## Preallocate space.
tuominen2022 = []

for f in files:
    
    ## Load file.
    subject = f.replace('.json','')

    with open(os.path.join(fdir, f), 'r') as f:
        JSON = json.load(f)
    
    ## Initialize dictionary.
    dd = dict(subject=subject)
    
    ## Gather demographics info.
    demo, = [dd for dd in JSON if dd['trial_type'] == 'survey-demo']
    dd.update(demo['responses'])
    
    ## Format data.
    dd['siblings'] = int(dd['siblings']) if 'siblings' in dd else np.nan
    
    ## HACK: integrate siblings survey.
    sf = os.path.join(fdir, f'{subject}_siblings.json')
    
    if os.path.isfile(sf):
        with open(sf, 'r') as f:
            siblings, = json.load(f)
            
            ## Try to convert response to integer.
            try: 
                dd['siblings'] = int(siblings['response'].get('siblings', np.nan))   
                
            ## Custom data handling.
            except:
                lut = {'one': 1}
                dd['siblings'] = lut[siblings['response'].get('siblings', np.nan)]
    
    ## Gather infrequency items.
    infreq = 0
    infreq += sum([dd.get('infrequency', 0) for dd in JSON if dd.get('survey', '') == 'bisbas'])
    infreq += sum([dd.get('infrequency', 0) for dd in JSON if dd.get('survey', '') == 'sns'])
    infreq += sum([dd.get('infrequency', 0) for dd in JSON if dd.get('survey', '') == 'map'])
    infreq += sum([dd.get('instructed', 0) for dd in JSON if dd.get('survey', '') == 'mace'])
    dd['infreq'] = infreq
    
    ## Gather honeypot.
    bot = 0
    bot += sum([dd.get('honeypot', 0) for dd in JSON if dd.get('survey', '') == 'bisbas'])
    bot += sum([dd.get('honeypot', 0) for dd in JSON if dd.get('survey', '') == 'sns'])
    bot += sum([dd.get('honeypot', 0) for dd in JSON if dd.get('survey', '') == 'map'])
    bot += sum([dd.get('honeypot', 0) for dd in JSON if dd.get('survey', '') == 'mace'])
    dd['bot'] = bot
    
    ## Append info.
    tuominen2022.append(dd)
    
## Concatenate data.
tuominen2022 = DataFrame(tuominen2022)

## Restrict to columns of interest.
drop_cols = ['gender-free-response', 'education']
tuominen2022 = tuominen2022.drop(columns=drop_cols).rename(columns={'gender-categorical':'gender'})

## Format data.
tuominen2022['race'] = tuominen2022.race.apply(lambda x: 'Multiracial' if len(x) > 1 else x[0])
tuominen2022['age'] = tuominen2022.age.astype(float).round(2)

## Insert study.
tuominen2022.insert(0, 'study', 'tuominen2022')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Merge & save DataFrames.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Concatenate DataFrames.
data = concat([
    teicher2015,
    tuominen2022
])

## Re-index subjects.
f = lambda x: x.study + '_' + x.subject
data['subject'] = np.unique(data.apply(f, 1), return_inverse=True)[-1] + 1

## Format data.
data[['infreq','bot']] = data[['infreq','bot']].fillna(0)

## Save data.
data.to_csv(os.path.join(DATA_DIR, 'covariates.csv'), index=False)