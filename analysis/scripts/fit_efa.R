library(tidyverse)
library(lavaan)
# https://solomonkurz.netlify.app/post/2021-05-11-yes-you-can-fit-an-exploratory-factor-analysis-with-lavaan/
# https://vankesteren.github.io/efast/efa_lavaan

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
data = read.csv('../data/mace.csv')

## Define studies.
studies = list('teicher2015', 'tuominen2022')

## Define locally dependent variables.
ld = list(c('x6', 'x7', 'x8'), c('x9', 'x10', 'x11'), c('x13', 'x14'), 
          c('x15', 'x16'), c('x19', 'x20'), c('x21', 'x22', 'x23'), 
          c('x24', 'x25'), c('x33', 'x34', 'x35'), c('x36', 'x37'))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main loop.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

for (study in studies) {
  
  ## Construct pivot table.
  df <- data %>% 
    filter(study == c(study)) %>% 
    pivot_wider(id_cols = subject, names_from = item, names_prefix= "x", values_from = response) 
  
  ## Iteratively collapse across LD pairs.
  for (items in ld) {
    df[items[1]] = rowSums(df[items])
    df = df[,!(names(df) %in% items[2:length(items)])]
  }
  
  ## Define formula (2-factor).
  f2 <- '
  efa("efa")*f1 +
  efa("efa")*f2 =~ x1 + x2 + x3 + x4 + x5 + x6 + x9 + x12 + x13 + x15 + x17 + x18 + 
                   x19 + x21 + x24 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + 
                   x36 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47 + 
                   x48 + x49 + x50 + x51 + x52'
  
  ## Fit EFA (2-factor).
  efa <- cfa(model = f2, data = df, rotation = "geomin", estimator = "WLSMV",
             orthogonal = TRUE, ordered = TRUE, std.lv = TRUE)
  
  ## Extract & save factor loadings.
  loadings = inspect(efa, what="std")$lambda
  write.csv(loadings, paste('..', 'stan_results', study, 'efa_f2_geomin.csv', sep='/'))
  
  ## Define formula (3-factor).
  f3 <- '
  efa("efa")*f1 +
  efa("efa")*f2 +
  efa("efa")*f3 =~ x1 + x2 + x3 + x4 + x5 + x6 + x9 + x12 + x13 + x15 + x17 + x18 + 
                   x19 + x21 + x24 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + 
                   x36 + x38 + x39 + x40 + x41 + x42 + x43 + x44 + x45 + x46 + x47 + 
                   x48 + x49 + x50 + x51 + x52'
  
  ## Fit EFA (3-factor).
  efa <- cfa(model = f3, data = df, rotation = "geomin", estimator = "WLSMV",
             orthogonal = TRUE, ordered = TRUE, std.lv = TRUE)
  
  ## Extract & save factor loadings.
  loadings = inspect(efa, what="std")$lambda
  write.csv(loadings, paste('..', 'stan_results', study, 'efa_f3_geomin.csv', sep='/'))
  
}


