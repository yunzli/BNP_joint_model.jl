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
    S_gamma = hyper["S_gamma"]

    g = cur["g"]
    gamma = cur["gamma"]
    sigma2_gamma = hyper["sigma2_gamma"] 

    S_gamma_new = 1 / (1/S_gamma + g/sigma2_gamma)
    s_gamma_new = S_gamma_new * (s_gamma/S_gamma + sum(gamma)/sigma2_gamma)

    mu_gamma = rand(Normal(s_gamma_new, sqrt(S_gamma_new)), 1)[1]

    return mu_gamma
end