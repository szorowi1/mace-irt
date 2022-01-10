data {

    // Metadata
    int<lower=1>  N;                        // Number of total observations
    int<lower=1>  M;                        // Number of latent dimensions
    int<lower=1>  J[N];                     // Person-indicator per observation
    int<lower=1>  K[N];                     // Item-indicator per observation
    
    // Response data
    int<lower=0, upper=1>  Y[N];            // Item response
    
    // Design matrix
    int<lower=0, upper=1>  Q[max(K), M];    // Q-matrix

}
transformed data {

    int  NJ = max(J);                       // Number of total persons
    int  NK = max(K);                       // Number of total items
    int  NQ = sum(to_array_1d(Q));          // Number of total loadings
    real c = -0.841;                        // Center loadings

}
parameters {

    matrix[NJ,M]  theta_pr;                 // Subject abilities (standardized)
    vector[NK]  beta;                       // Item difficulties
    vector[NQ]  alpha_pr;                   // Item discriminations (standardized)

}
transformed parameters {

    // Construct subject abilities (orthogonalized)
    matrix[M,NJ] theta = transpose(qr_thin_Q(theta_pr) * sqrt(NJ - 1));
    
    // Constuct item discriminations
    row_vector[M] alpha[NK];
    
    // Construction block 
    {
    int q = 1;
    for (k in 1:NK) {
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

    // Compute linear predictor
    vector[N] mu;
    for (n in 1:N) {
        mu[n] = inv_logit(alpha[K[n]] * theta[,J[n]] - beta[K[n]]);
    }
    
    // Accuracy likelihood
    target += bernoulli_lpmf(Y | mu);
    
    // Priors
    target += std_normal_lpdf(to_vector(theta_pr));
    target += std_normal_lpdf(alpha_pr);
    target += normal_lpdf(beta | 0, 2);

}
generated quantities {

    row_vector[M]   lambda[NK];             // Standardized item loadings
    row_vector[NK]  tau;                    // Standardized item thresholds
    
    // Construction block
    {
    real D = 1.702;
    for (k in 1:NK) {
        real omega = sqrt(1 + sum(square(alpha[k] / D)));
        lambda[k] = (alpha[k] / D) / omega;
        tau[k] = -beta[k] / omega;
    }
    }

}