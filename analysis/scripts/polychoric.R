library(tidyverse)
library(psych)

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

## Preallocate space.
corr = c()

for (study in studies) {
  
  ## Restrict to study.
  df = data[data$study == study,]
  
  ## Construct pivot table.
  df <- df %>% 
    pivot_wider(id_cols = subject, names_from = item, names_prefix= "x", values_from = response) 
  
  ## Iteratively collapse across LD pairs.
  for (items in ld) {
    df[items[1]] = rowSums(df[items])
    df = df[,!(names(df) %in% items[2:length(items)])]
  }
  
  ## Drop subject column.
  df <- subset(df, select=-c(subject))
  
  ## Compute tetrachoric correlation.
  df = polychoric(df)$rho
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
write.csv(corr, paste('..', 'stan_results', 'polychoric.csv', sep='/'), row.names=FALSE)
