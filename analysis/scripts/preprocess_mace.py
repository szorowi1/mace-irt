import os, json
import numpy as np
from os.path import dirname
from pandas import DataFrame, concat, read_csv
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DIR = os.path.join(ROOT_DIR, 'raw')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load mapping file.
mapping = read_csv(os.path.join(ROOT_DIR, 'raw', 'mapping.csv'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Teicher & Parigger (2015)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
teicher2015 = read_csv(os.path.join(ROOT_DIR, 'raw', 'teicher2015', 'pone.0117423.s018.csv'))

## Set index.
teicher2015 = teicher2015.rename(columns={'Username':'subject'}).set_index('subject')
teicher2015.index = [s.lower() for s in teicher2015.index]

## Limit to MACE items & melt.
teicher2015 = teicher2015[mapping.item_id].copy()
teicher2015 = teicher2015.melt(var_name='item_id', value_name='response', ignore_index=False)
teicher2015 = teicher2015.reset_index().rename(columns={'index':'subject'})

## Merge with mapping.
teicher2015 = teicher2015.merge(mapping, on='item_id')

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
    
    ## Gather siblings data.
    demo, = [dd for dd in JSON if dd['trial_type'] == 'survey-demo']
    siblings = int(demo['responses'].get('siblings', 99))
    
    ## HACK: integrate siblings survey.
    sf = os.path.join(fdir, f'{subject}_siblings.json')
    
    if os.path.isfile(sf):
        with open(sf, 'r') as f:
            siblings, = json.load(f)
            
            ## Try to convert response to integer.
            try: 
                siblings = int(siblings['response'].get('siblings', 99))   
                
            ## Custom data handling.
            except:
                lut = {'one': 1}
                siblings = lut[siblings['response'].get('siblings', 99)]
    
    ## Gather MACE data.
    mace = [dd for dd in JSON if 'survey' in dd and dd['survey']=='mace']
    
    ## Preallocate space.
    responses = np.zeros((53, 2))
    
    ## Iterate over responses.
    current_item = -1
    for k, v in [(k,v) for dd in mace for k,v in dd['responses'].items()]:
        
        ## Update current item.
        if 'ages' not in k: current_item += 1
            
        ## Indicate response.
        if 'ages' not in k:
            responses[current_item,0] = 1 if v == 'Yes' else 0
        
        ## Increment age counts.
        else:
            responses[current_item,1] += 1
            
    ## Convert to DataFrame.
    df = DataFrame(responses, columns=['response','n_years']).drop(50)
    
    ## Insert metadata. 
    df.insert(0, 'subject', subject)
    df.insert(1, 'item', np.arange(len(df))+1)
    
    ## Score WSV subscale.
    df.loc[np.in1d(df.item, [15, 16, 17, 18]), ['response','n_years']] *= 1 if siblings else np.nan
    
    ## Append.
    tuominen2022.append(df)
    
## Concatenate data.
tuominen2022 = concat(tuominen2022)

## Merge with mapping.
tuominen2022 = tuominen2022.merge(mapping, on='item')

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

## Re-organize columns.
cols = ['study','subject','item','item_id','subscale','response','n_years']
data = data[cols].sort_values(['subject','item'])

## Reverse score items.
indices = np.in1d(data.item, [42, 43, 44, 45, 51, 52])
data.loc[indices, 'response'] = 1 - data.loc[indices, 'response']
data.loc[indices, 'n_years'] = 18 - data.loc[indices, 'n_years']

## Save data.
data.to_csv(os.path.join(DATA_DIR, 'mace.csv'), index=False)