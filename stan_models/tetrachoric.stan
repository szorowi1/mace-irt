functions {

    // Adapted from: https://github.com/stan-dev/stan/issues/2356
    real binormal_cdf(real z1, real z2, real rho) {
        if (z1 != 0 || z2 != 0) {
            real denom = fabs(rho) < 1.0 ? sqrt((1 + rho) * (1 - rho)) : not_a_number();
            real a1 = (z2 / z1 - rho) / denom;
            real a2 = (z1 / z2 - rho) / denom;
            real product = z1 * z2;
            real delta = product < 0 || (product == 0 && (z1 + z2) < 0);
            return 0.5 * (Phi(z1) + Phi(z2) - delta) - owens_t(z1, a1) - owens_t(z2, a2);
        }
        return 0.25 + asin(rho) / (2 * pi());
    }
  
    // Adapted from: https://discourse.mc-stan.org/t/bivariate-probit-in-stan/2025/7
    real biprobit_lpdf(vector y1, vector y2, real mu1, real mu2, real rho) {
    
        // Compute generated quantities
        int n = size(y1);
        vector[n] q1 = 2 * y1 - 1;
        vector[n] q2 = 2 * y2 - 1;
        vector[n] w1 = q1 * mu1;
        vector[n] w2 = q2 * mu2;
        vector[n] r12 = rho * q1 .* q2;

        // Compute log-likelihood
        real LL = 0;
        for (i in 1:n) {
            LL += log(binormal_cdf(w1[i], w2[i], r12[i]));
        }
        
        return LL;
    }
  
}
data {

    // Metadata
    int<lower=1>  J;                            // Number of subjects
    
    // Data
    matrix<lower=0, upper=1>[J,2] Y;            // Response accuracy
    
}
parameters {

    // Item parameters
    vector[2]  mu;                              // Mean accuracy
    real<lower=-1, upper=1> rho;
    
}
model {

    // Likelihood
    target += biprobit_lpdf(Y[,1] | Y[,2], mu[1], mu[2], rho);

    // Priors
    target += std_normal_lpdf(mu);
    target += uniform_lpdf(rho | -1, 1);
    
}