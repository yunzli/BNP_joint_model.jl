function update_mu_theta(cur, hyper)

    s_theta = hyper["s_theta"]
    S_theta = hyper["S_theta"]

    k = cur["k"]
    theta = cur["theta"]
    sigma2_theta = hyper["sigma2_theta"]

    S_theta_new = 1 / (1/S_theta + k/sigma2_theta)
    s_theta_new = S_theta_new * (s_theta/S_theta + sum(log.(theta))/sigma2_theta)

	mu_theta = rand(Normal(s_theta_new, sqrt(S_theta_new)),1)[1]

	return mu_theta 
end


function update_b_phi(cur, hyper)

    a_phi = hyper["a_phi"]
    r_phi = hyper["r_phi"]
    R_phi = hyper["R_phi"]

    k = cur["k"]
    phi = cur["phi"]

	r_phi_new = r_phi + a_phi * k
    R_phi_new = R_phi + sum(phi .^ 2)

	b_phi = rand(InverseGamma(r_phi_new, R_phi_new), 1)[1]

	return b_phi
end 


function update_mu_beta(cur, hyper)

    s_beta = hyper["s_beta"]
    S_beta = hyper["S_beta"]

    k = cur["k"]
    beta = cur["beta"]
    sigma2_beta = hyper["sigma2_beta"]

    S_beta_new = 1 / (1/S_beta + k/sigma2_beta)
    s_beta_new = S_beta_new * (s_beta/S_beta + sum(beta)/sigma2_beta)

    mu_beta = rand(Normal(s_beta_new, sqrt(S_beta_new)), 1)[1]

    return mu_beta
end