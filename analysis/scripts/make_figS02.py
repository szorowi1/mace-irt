import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import dirname
from pandas import read_csv
import matplotlib.gridspec as gridspec
sns.set_theme(style='white', context='notebook', font_scale=1.33)
ROOT_DIR = dirname(dirname(os.path.realpath(__file__)))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define data table.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
mace = read_csv(os.path.join(ROOT_DIR, 'data', 'mace.csv'))
covariates = read_csv(os.path.join(ROOT_DIR, 'data', 'covariates.csv'))
dif = read_csv(os.path.join(ROOT_DIR, 'stan_results', 'logdif.csv'))

## Apply screening.
covariates = covariates.query('infreq <= 0.5')
mace = mace[mace.subject.isin(covariates.query('infreq <= 0.5').subject)]

## Restrict to available data.
table = mace[mace.response.notnull()]

## Merge with covariates.
table = table.merge(covariates[['subject','gender']])
table = table.query('gender == "Male" or gender == "Female"')

## Construct tables.
table = table.pivot_table('response',['subscale','item'],'gender').reset_index()
table = table.merge(dif.loc[dif.variable == 'gender', ['item','coef']])

## Add fake rows (for padding).
for i in range(9): table = table.append({'item':-(i+1)},ignore_index=True)

## Sort items.
order = [1, 2, 3, 4, -1, 6, 7, 8, 9, 10, 11, -2, 5, 40, 41, 48, 49, 50, -3, 12, 13, 14, 19, 20, 36, 37,
         -4, 38, 39, 42, 43, 52, -5, 44, 45, 46, 47, 51, -6, 15, 16, 17, 18, -7, 21, 22, 23, 24, 25, -8, 
         26, 27, 28, 29, 30, -9, 31, 32, 33, 34, 35]
table = table.set_index('item').loc[order].reset_index()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Initialize canvas.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Initialize canvas.
fig = plt.figure(figsize=(7,11), constrained_layout=True)

## Initialize panels.
panel = fig.add_gridspec(25, 2, left=0.12, right=0.98, top=0.98, bottom=0.12, hspace=0.0, wspace=0.1)

## Define aesthetics.
yticks = [i for i, coef in enumerate(table.coef) if not np.isnan(coef)]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Panel A: Absolute rates.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Initialize axis.
ax = plt.subplot(panel[:-1,0])

## Plot connecting lines.
for i, (x, y) in enumerate(zip(table.Male, table.Female)):
    ax.plot([x,y], [i,i], color='0.5')

## Plot points.
sns.stripplot(x='Male', y='item', data=table, order=order, color='0.2', orient='h', 
              jitter=False, size=8, linewidth=1, edgecolor='w', ax=ax)
sns.stripplot(x='Female', y='item', data=table, order=order, color='#52abdd', orient='h', 
              jitter=False, size=8, linewidth=1, edgecolor='w', ax=ax)

## Add details.
f = lambda x: '%s%s - %s' %('** ' if x['item'] in [42,43,44,45,51,52] else '', x['subscale'], x['item']) 
ax.set(ylim=(60.5, -0.5), yticks=yticks, ylabel='', xlim=(-0.02, 0.82), xticks=[0.0, 0.2, 0.4, 0.6, 0.8], 
       xticklabels=['0%', '20%', '40%', '60%', '80%'], xlabel='Percentage endorsed')
ax.set_yticklabels(table.dropna().apply(f, 1).values, fontsize=11)
ax.grid(which='major', axis='y', lw=0.5)
ax.annotate('A', (0,0), (0,1), 'axes fraction', va='bottom', ha='center', fontsize=24, fontweight='bold')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Panel B: Relative rates.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Initialize axis.
ax = plt.subplot(panel[:-1,1])

## Define palette.
palette = np.where(table.coef < -0.638, '0.2', np.where(table.coef > 0.638, '#52abdd', '0.8'))
bounds = [-0.638, 0.00, 0.638]

## Plot points.
sns.stripplot(x='coef', y='item', data=table, order=order, palette=palette, orient='h', 
              jitter=False, size=8, linewidth=1, edgecolor='w', ax=ax)

## Add lines.
for subscale in table.subscale.unique():
    ix = table.query(f'subscale == "{subscale}"').index
    ax.vlines(bounds, ix.min(), ix.max(), colors=['0.8','0.7','0.8'], linestyles='--', lw=0.9)

## Add details.
ax.set(ylim=(60.5, -0.5), yticks=yticks, ylabel='', yticklabels=[], xlim=(-1.65,1.65), 
       xticks=[-1, 0, 1], xlabel='Log odds')
ax.grid(which='major', axis='y', lw=0.5)
ax.annotate('B', (0,0), (0,1), 'axes fraction', va='bottom', ha='center', fontsize=24, fontweight='bold')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Panel C: Legend.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Initialize axis.
ax = plt.subplot(panel[-1,:])

## Add legend.
ax.scatter([], [], s=80, color='0.2', label='Men')
ax.scatter([], [], s=80, color='#52abdd', label='Women')
ax.set(xticks=[], yticks=[], xlabel='', ylabel='')
ax.legend(loc=10, ncol=2, frameon=False, columnspacing=1.5, handletextpad=0, fontsize=14)

sns.despine(left=True, right=True, top=True, bottom=True)
plt.savefig(os.path.join(ROOT_DIR, '..', 'figures', 'figS02.png'), dpi=100)