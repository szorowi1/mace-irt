import os, json
import numpy as np
from os.path import dirname
from pandas import DataFrame, concat
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DIR = os.path.join(ROOT_DIR, 'raw')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main loop.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Locate files.
files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith('.json')])

## Preallocate space.
DATA = []

for f in files:
    
    ## Load file.
    subject = f.replace('.json','')

    with open(os.path.join(RAW_DIR, f), 'r') as f:
        JSON = json.load(f)
    
    ## Initialize dictionary.
    dd = dict(subject=subject)
    
    ## Gather demographics info.
    demo, = [dd for dd in JSON if dd['trial_type'] == 'survey-demo']
    dd.update(demo['responses'])
    
    ## Format data.
    dd['siblings'] = int(dd['siblings']) if 'siblings' in dd else np.nan
    
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
    DATA.append(dd)
    
## Convert to DataFrames.
DATA = DataFrame(DATA).rename(columns={'gender-categorical':'gender'})

## Save data.
DATA.to_csv(os.path.join(DATA_DIR, 'covariates.csv'), index=False)