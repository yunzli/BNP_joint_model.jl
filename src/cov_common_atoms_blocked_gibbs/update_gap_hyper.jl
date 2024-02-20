function update_mu_lambda(cur, hyper)

    s_lambda = hyper["s_lambda"]
    S_lambda = hyper["S_lambda"]

    BH = hyper["BH"]
    lambda = cur["lambda"]
    sigma2_lambda = hyper["sigma2_lambda"]

    S_lambda_new = 1 / (1/S_lambda + BH/sigma2_lambda)
    s_lambda_new = S_lambda_new * (s_lambda/S_lambda + sum(log.(lambda))/sigma2_lambda)

	mu_lambda = rand(Normal(s_lambda_new, sqrt(S_lambda_new)),1)[1]

	return mu_lambda 
end


function update_b_eta(cur, hyper)

    a_eta = hyper["a_eta"]
    r_eta = hyper["r_eta"]
    R_eta = hyper["R_eta"]

    BH = hyper["BH"]
    eta = cur["eta"]

	r_eta_new = r_eta + a_eta * BH
    R_eta_new = R_eta + sum(eta .^ 2)

	b_eta = rand(InverseGamma(r_eta_new, R_eta_new), 1)[1]

	return b_eta
end 