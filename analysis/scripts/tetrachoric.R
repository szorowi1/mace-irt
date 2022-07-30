library(tidyverse)
library(psych)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Define parameters.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Load data.
data = read.csv('../data/mace.csv')

## Define studies.
studies = list('teicher2015', 'tuominen2022')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
### Main loop.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

## Preallocate space.
corr = c()

for (study in studies) {
  
  ## Restrict to study.
  df = data[data$study == study,]
  
  ## Construct pivot table.
  df <- df %>% 
    pivot_wider(id_cols = subject, names_from = item, names_prefix= "x", values_from = response) 
  
  ## Drop subject column.
  df <- subset(df, select=-c(subject))
  
  ## Compute tetrachoric correlation.
  df = tetrachoric(df)$rho
  df = round(df, 6)
  
  ## Convert to long-list.
  df = as.data.frame(as.table(df))
  df = rename(df, k1 = Var1, k2 = Var2, !!study := Freq)
  
  ## Append to list.
  corr = append(corr, list(df), 0)
  
}

## Merge DataFrames.
corr = corr %>%
  Reduce(function(dtf1,dtf2) left_join(dtf1,dtf2,by=c("k1","k2")), .)

## Save data.
write.csv(corr, paste('..', 'stan_results', 'tetrachoric.csv', sep='/'), row.names=FALSE)
