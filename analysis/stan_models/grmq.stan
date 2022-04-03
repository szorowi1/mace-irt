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
    int<lower=1>  M;                        // Number of latent dimensions
    int<lower=1>  K;                        // Number of total items
    int<lower=1>  J[N];                     // Person-indicator per observation

    // Item partitions
    array[K] int<lower=1>  r;               // Number of observations per item
    array[K] int<lower=2>  s;               // Number of response levels per item

    // Response data
    array[N] int<lower=1, upper=max(s)> Y;  // Item response
    
    // Design matrix
    int<lower=0, upper=1>  Q[K, M];         // Q-matrix
    
    // Prior counts
    vector<lower=1>[sum(s)]  C;             // Number of observed responses

}
transformed data {

    int  NJ = max(J);                       // Number of total persons
    int  NQ = sum(to_array_1d(Q));          // Number of total loadings
    int  NS = sum(s);                       // Number of total intercepts
    real c = -0.841;                        // Center loadings

}
parameters {

    // Subject abilities
    matrix[NJ,M]  theta_pr;
        
    // Item discriminations
    vector[NQ]  alpha_pr;
     
    // Item difficulties
    vector<lower=0>[NS]  tau_pr;

}
transformed parameters {

    // Construct subject abilities
    matrix[NJ,M] theta = qr_thin_Q(theta_pr) * sqrt(NJ - 1);
    
    // Construct item discriminations
    vector[M] alpha[K];
    {
    int q = 1;
    for (k in 1:K) {
        for (m in 1:M) {
            if (Q[k,m] == 1) {
                alpha[k,m] = Phi_approx(c + alpha_pr[q]) * 5;
                q += 1;
            } else {
                alpha[k,m] = 0;
            }
        }
    }
    }

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
        target += ordered_logistic_glm_lpmf(segment(Y, pos[3], r[k]) | theta[segment(J, pos[3], r[k])], alpha[k], cutpoints);
        
        // Increment counters
        pos[1] += s[k] - 1;
        pos[2] += s[k];
        pos[3] += r[k];
        
    }
    
    // Priors
    target += std_normal_lpdf(to_vector(theta_pr));
    target += std_normal_lpdf(alpha_pr);
    target += -NS * log(sum(tau_pr) + K);
    
}
generated quantities {

    // Standardized factor loadings
    vector[M] lambda[K];
    
    // Construction block
    {
    real D = 1.702;
    for (k in 1:K) {
        real omega = sqrt(1 + sum(square(alpha[k] / D)));
        lambda[k] = (alpha[k] / D) / omega;
    }
    }
    
}