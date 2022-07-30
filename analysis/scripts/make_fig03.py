import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import dirname
from pandas import Categorical, DataFrame, concat, read_csv
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))
sns.set_theme(context='notebook', style='white', font='sans-serif', font_scale=1.33)
sns.set_style("ticks")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define data parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## I/O parameters.
study = 'tuominen2022'

## Define subscales by model.
factors = {
    1: ['general'],
    2: ['general', 'EN', 'NVEA', 'PN', 'PA', 'VA', 'PeerPA', 'PeerVA', 'SA', 'WIPV', 'WSV'],
    3: ['general', 'neglect'],
    4: ['general','peer','reverse']
}

## Define item ordering.
order = ['VA','PA','NVEA','SA','WSV','WIPV','EN','PN','PeerVA','PeerPA']

## Define LD item pairs.
ld = [[6,7,8], [9,10,11], [13,14], [15,16], [19,20], [21,22,23], [24,25], [33,34,35], [36,37]]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Load and prepare design matrix.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load design data.
design = read_csv(os.path.join(ROOT_DIR, 'data', 'design.csv'), index_col=0)

## Collapse across LD items.
for ix in ld: 
    design = design.drop(index=ix[1:])
    design = design.rename(index={ix[0]:'/'.join(['%s' %i for i in ix])})
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Contruct loadings table. 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
## Prepare factor loadings.
loadings = np.zeros((len(design), len(factors)*2))

## Main loop.
k = 0
for model, cols in factors.items():
    
    ## Load Stan summary.
    df = os.path.join(ROOT_DIR, 'stan_results', study, f'grmq_m{model}_summary.tsv')
    df = read_csv(df, sep='\t', index_col=0)
    
    ## Extract factor loadings.
    for i, j in np.column_stack([np.where(design[cols])]).T:
        loadings[i,k+int(j > 0)] = df.loc[f'lambda[{i+1},{j+1}]','Mean']
    
    ## Update counter.
    k += 2
    
## Restrict to non-empty columns.
loadings = loadings[:,np.any(loadings, axis=0)]
    
## Convert to DataFrame.
loadings = DataFrame(loadings)
loadings.insert(0, 'item', design.index)
loadings.insert(1, 'subscale', np.sort(order)[np.where(design[np.sort(order)])[-1]])

## Sort DataFrame.
loadings['subscale'] = Categorical(loadings.subscale, categories=order, ordered=True)
loadings = loadings.sort_values(['subscale','item']).set_index(['subscale','item'])

## Extract loadings.
X1 = loadings.values
X2 = loadings.applymap(np.digitize, bins=[0.35, 0.6]).values

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define plot parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Palette generator: https://learnui.design/tools/data-color-picker.html#single

## Define palette.
palette = ['#e3e6e6','#a1bfc8','#6598a8']

## Define axis limits.
xlim = (0,1)
ylim = (-0.5,38.5)

## Define plot labels.
xlabels = ['General','General','Specific','General','Specific','General','Specific']

## Define subscale labels.
subscales = dict(
    VA = 0, PA = 4, NVEA = 6, SA = 12, WSV = 16,
    WIPV = 19, EN = 21, PN = 26, PeerVA = 31, PeerPA = 36
)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define layout.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Initialize canvas.
fig = plt.figure(figsize=(10,10))

## Initialize gridspec layout.
gs = fig.add_gridspec(1, 4, left=0.06, right=0.97, top=0.97, bottom=0.06, hspace=0, wspace=0.55)

## Define axes.
axes = [ax for i in range(4) for ax in gs[i].subgridspec(1, 2, wspace=0)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Plot loadings.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Define convenience functions.
xticks = lambda i: [0.5, 1.0] if i % 2 else [0.0, 0.5, 1.0]
annot = lambda x: ('%0.2f' %x)[1:]
label = lambda k: ['0.85', '0.35', '0.0'][k]
ha = lambda i: 'right' if i % 2 else 'left'

for i, (ax, X, K) in enumerate(zip(axes[1:], X1.T, X2.T)):
    
    ## Initialize axis.
    ax = plt.subplot(ax)
    
    ## Iteratively plot loadings.
    for y, (x, k) in enumerate(zip(X, K)): 
        
        ## Plot loading.
        ax.barh(y, x, height=1, edgecolor='none', color=palette[k])
        
        ## Add annotation.
        if x: ax.annotate(annot(x), (0,0), (1.01, y), xycoords='data', ha=ha(i), va='center', 
                          fontsize=12, color=label(k))

    ## Add detail.
    ax.set(xlim=xlim, xticks=xticks(i), yticks=[], yticklabels=[], ylim=ylim)
    ax.hlines(np.arange(39)-0.5, 0, 1, color='w', lw=0.5)
    if i % 2: ax.axvline(0.0, color='w')
    ax.set_xticklabels(xticks(i), fontsize=13)
    ax.set_xlabel(xlabels[i], fontsize=14, labelpad=2)
    ax.set_facecolor('#f1f1f1')
        
    ## Add title.
    if not i: ax.annotate('Model 1', (0,0), (0, -0.6), ha='left', va='bottom', fontsize=18)
    elif not i % 2: ax.annotate(f'Model {i//2+1}', (0,0), (0, -0.6), ha='center', va='bottom', fontsize=18)
        
    ## Rotate axes.
    if i % 2: ax.invert_xaxis()
    ax.invert_yaxis()
    
    ## Manipulate spines.
    sns.despine(left=True, right=True, top=True, bottom=False, ax=ax)
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Plot labels.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    
## Initialize axis.
ax = plt.subplot(axes[0])
    
## Define coordinate space.
ax.barh(np.arange(39), np.zeros(39))

## Plot item numbers.
for i, (ss, item) in enumerate(loadings.index):
    label = f'{ss} - {item}' 
    if item in [42,43,44,45,51,52]: label = '** ' + label
    fontsize = 12 if len(label) < 15 else 11
    ax.annotate(label, (0,0), (0.9, i), ha='right', va='center', fontsize=fontsize)
    
## Add detail.
ax.set(xlim=xlim, xticks=[], ylim=ylim, yticks=[])

## Rotate axes.
ax.invert_yaxis()

## Manipulate spines.
sns.despine(left=True, right=True, top=True, bottom=True, ax=ax)
    
## Save figure.
plt.savefig(os.path.join(ROOT_DIR, '..', 'figures', 'fig03.png'), dpi=100)