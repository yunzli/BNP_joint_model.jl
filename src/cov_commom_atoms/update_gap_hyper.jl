function update_mu_lambda(cur, hyper)

    s_lambda = hyper["s_lambda"]
    S_lambda = hyper["S_lambda"]

    g = cur["g"]
    lambda = cur["lambda"]
    sigma2_lambda = hyper["sigma2_lambda"]

    S_lambda_new = 1 / (1/S_lambda + g/sigma2_lambda)
    s_lambda_new = S_lambda_new * (s_lambda/S_lambda + sum(log.(lambda))/sigma2_lambda)

	mu_lambda = rand(Normal(s_lambda_new, sqrt(S_lambda_new)),1)[1]

	return mu_lambda 
end


function update_b_eta(cur, hyper)

    a_eta = hyper["a_eta"]
    r_eta = hyper["r_eta"]
    R_eta = hyper["R_eta"]

    g = cur["g"]
    eta = cur["eta"]

	r_eta_new = r_eta + a_eta * g
    R_eta_new = R_eta + sum(eta .^ 2)

	b_eta = rand(InverseGamma(r_eta_new, R_eta_new), 1)[1]

	return b_eta
end 


function update_mu_gamma(cur, hyper)

    s_gamma = hyper["s_gamma"]
    S_gamma_inv = hyper["S_gamma_inv"]
    Sigma_gamma_inv = hyper["Sigma_gamma_inv"] 

    gamma = cur["gamma"]
    g = cur["g"]

    S_gamma_inv_new = S_gamma_inv + g*Sigma_gamma_inv
    S_gamma_new = svd2inv(S_gamma_inv_new) 

    s_gamma_new = S_gamma_new * (S_gamma_inv * s_gamma + Sigma_gamma_inv * sum(gamma, dims=1)[1,:])

    mu_gamma_new = rand(MvNormal(s_gamma_new, S_gamma_new), 1)

    return vec(mu_gamma_new)
end