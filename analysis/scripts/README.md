# scripts

Scripts used for analysis. Described in the order in which they ought to be run.

## Descriptions

#### (1) Data preprocessing
- Preprocess MACE data  (`preprocess_mace.py`): collates the raw MACE data into one CSV
- Preprocess covariates (`preprocess_covariates.py`): collates the raw covariate data into one CSV

#### (2) Descriptive analysis
- Differential item functioning (`fit_logdif.py`): test for DIf using logistic regression.
- Tetrachoric correlations (`fit_tetrachoric.py`): compute tetrachoric item correlation matrix.
- Polychoric correlations (`polychoric.R`): compute polychoric item correlation matrix.

#### (3) Confirmatory item response modeling
- Fit graded response model (`fit_grmq.py`): fits the confirmatory item response model using Stan.
- Fit baseline model (`fit_grmb.py`): fits the baseline model using Stan.

#### (4) Goodness-of-fit & model comparison
- Pareto smoothed importance sampling (`psis.py`)
- Model comparison (`fit_grmq_ppc.py`)
- Chi-square discrepancy checks (`fit_grmq_x2.py`)
- Local dependence checks (`fit_grmq_q3.py`)

#### (5) Exploratory item response modeling
- Conduct exploratory factor analysis (`fit_efa.R`): perform EFA using lavaan.

#### (6) Figures / plotting
- Figure 02 (`make_fig02.py`): make factor loadings plot (original sample).
- Figure 03 (`make_fig03.py`): make factor loadings plot (replication sample).
- Figure 04 (`make_fig04.py`): make factor loadings plot (exploratory factor analysis, cf-quartimax).
- Figure S01 (`make_figS01.py`): make differential item functioning plot (by sample).
- Figure S02 (`make_figS02.py`): make differential item functioning plot (by gender).
- Figure S04 (`make_fig04.py`): make factor loadings plot (exploratory factor analysis, geomin).