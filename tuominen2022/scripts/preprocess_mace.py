import os, json
import numpy as np
from os.path import dirname
from pandas import DataFrame, concat, read_csv
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
    
    ## Gather siblings data.
    demo, = [dd for dd in JSON if dd['trial_type'] == 'survey-demo']
    siblings = int(demo['responses'].get('siblings', 0))
    
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
    DATA.append(df)
    
## Concatenate data.
DATA = concat(DATA)

## Reverse score items.
indices = np.in1d(DATA.item, [42, 43, 44, 45, 51, 52])
DATA.loc[indices, 'response'] = 1 - DATA.loc[indices, 'response']
DATA.loc[indices, 'n_years'] = 18 - DATA.loc[indices, 'n_years']

## Merge with metadata.
mapping = read_csv(os.path.join(RAW_DIR, 'mapping.csv'))
DATA = DATA.merge(mapping, on='item')

## Re-organize data.
DATA = DATA[['subject','item','item_id','subscale','response','n_years']]
DATA = DATA.sort_values(['subject','item'])

## Save data.
DATA.to_csv(os.path.join(DATA_DIR, 'mace.csv'), index=False)