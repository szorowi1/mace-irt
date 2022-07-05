// Graded response model (confirmatory item factor analysis w/ Q-matrix)
// https://discourse.mc-stan.org/t/ragged-array-of-simplexes/1382/15

functions {

    // Return simplex
    vector make_simplex(vector gamma) {
        real sum_gamma = sum(gamma) + 1;
        return append_row(gamma, 1) / sum_gamma;
    }

    // Return ordinal cutpoints from a vector of response probabilities.
    vector make_cutpoints(vector pi) {
        int C = rows(pi) - 1; 
        return logit( cumulative_sum( pi[:C] ) );
    }

}
data {

    // Metadata
    int<lower=1>  N;                        // Number of total observations
    int<lower=1>  K;                        // Number of total items

    // Item partitions
    array[K] int<lower=1>  r;               // Number of observations per item
    array[K] int<lower=2>  s;               // Number of response levels per item

    // Response data
    array[N] int<lower=1, upper=max(s)> Y;  // Item response
    
    // Prior counts
    vector<lower=1>[sum(s)]  C;             // Number of observed responses

}
transformed data {

    int  NS = sum(s);                       // Number of total intercepts

}
parameters {

    // Item difficulties
    vector<lower=0>[NS]  tau_pr;

}
model {

    // Generated quantities
    int pos[3] = rep_array(1, 3);

    // Main loop
    for (k in 1:K) {
        
        // Make endorsement rates
        vector[s[k]] pi = make_simplex(segment(tau_pr, pos[1], s[k] - 1));
        
        // Priors
        target += dirichlet_lpdf(pi | segment(C, pos[2], s[k]));
                
        // Make cutpoints
        vector[s[k] - 1] cutpoints = make_cutpoints(pi);
        
        // Response likelihood
        target += ordered_logistic_lupmf(segment(Y, pos[3], r[k]) | zeros_vector(r[k]), cutpoints);
        
        // Increment counters
        pos[1] += s[k] - 1;
        pos[2] += s[k];
        pos[3] += r[k];
        
    }
    
    // Priors
    target += -NS * log(sum(tau_pr) + K);
    
}
