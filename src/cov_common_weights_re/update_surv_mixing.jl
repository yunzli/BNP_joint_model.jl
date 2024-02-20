using PolyaGammaDistribution

function update_surv_pg_lantent(dat, cur) 

    survival = dat["survival"]
    epsilon = cur["epsilon"]
    n = length(survival)
    nu = dat["nu"]
    x = dat["x"]
    L = cur["L"]
    
    phi, theta, beta = cur["phi"], cur["theta"], cur["beta"]

    u = zeros(n)

    for i in 1:n
        Li = L[i]
        a = 1 + nu[i]
        b = phi[Li] * (log(survival[i]) - beta[Li,:]' * x[i,:] + log(epsilon[i]) - log(theta[Li]))
        u[i] = rand(PolyaGamma(a, b), 1)[1]
    end

    return u 
end


function update_theta(dat, cur, hyper)

    L = cur["L"]
    nl = cur["nl"]
    k = cur["k"]

    u = cur["u"]
    phi = cur["phi"]
    beta = cur["beta"]
    epsilon = cur["epsilon"]
    
    survival = dat["survival"]
    x = dat["x"]
    nu = dat["nu"]

    sigma2_theta = hyper["sigma2_theta"]
    mu_theta = cur["mu_theta"]

	theta_new = zeros(k)
    for l in 1:k 
        ind = findall(L .== l)

        sigma2_theta_new_inv = 1/sigma2_theta + phi[l]^2 * sum(u[ind])
        sigma2_theta_new = 1 / sigma2_theta_new_inv

        tmp1 = mu_theta / sigma2_theta
        tmp2 = phi[l]^2 * sum(u[ind] .* (log.(survival[ind]) .- x[ind,:] * beta[l,:] .+ log.(epsilon[ind])))
        tmp3 = 0.5 * phi[l] * (nl[l] - sum(nu[ind]))
        mu_theta_new = sigma2_theta_new * (tmp1 + tmp2 + tmp3)
        theta_new[l] = exp(rand(Normal(mu_theta_new, sqrt(sigma2_theta_new)), 1)[1])

    end

    return theta_new
end

function MH_phi_sampler(phi2, nu, c, a_phi, b_phi)

	phi2prop = exp(log(phi2) + rand(Normal(0, 1),1)[1])

	logcur = logpdf(Gamma(a_phi,b_phi), phi2) + 0.5 * sum((nu .- 1) .* c) * sqrt(phi2) + log(phi2)
	logpro = logpdf(Gamma(a_phi,b_phi), phi2prop) + 0.5 * sum((nu .- 1) .* c) * sqrt(phi2prop) + log(phi2prop)

	if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
		return phi2prop
	else
		return phi2
	end 
end 

function update_beta(dat, cur, hyper)

    nu = dat["nu"]
    survival = dat["survival"]
    x = dat["x"]
    p = dat["p"]

    Sigma_beta_inv = hyper["Sigma_beta_inv"]
    mu_beta = cur["mu_beta"]
    muSigma = Sigma_beta_inv * mu_beta

    u = cur["u"]
    epsilon = cur["epsilon"]
    L = cur["L"]

    k = cur["k"]
    theta = cur["theta"]
    phi = cur["phi"]

    beta_new = zeros(k, p)

    for l in 1:k 
        ind = findall(L .== l)

        Sigma_beta_inv_new = Sigma_beta_inv 
        for i in ind 
            Sigma_beta_inv_new += phi[l]^2 * u[i] * x[i,:] * x[i,:]' 
        end

        tmp1 = zeros(p)
        tmp2 = zeros(p)
        for i in ind 
            tmp1 += u[i] * (log(survival[i]) + log(epsilon[i]) - log(theta[l])) * x[i,:]
            tmp2 += (1 - nu[i]) * x[i,:]
        end
        Sigma_beta_new = svd2inv(Sigma_beta_inv_new)
        mu_beta_new = Sigma_beta_new * (muSigma + phi[l]^2 * tmp1 + 0.5*phi[l]*tmp2)

        beta_new[l,:] = rand(MvNormal(mu_beta_new, Sigma_beta_new), 1)
    end

    return beta_new
end

function update_phi(dat, cur, hyper)

	nu = dat["nu"] 
    survival = dat["survival"]
    x = dat["x"]

	a_phi = hyper["a_phi"]
	b_phi = cur["b_phi"]
	u = cur["u"]
    epsilon = cur["epsilon"]

    k = cur["k"]
	theta = cur["theta"]
    beta = cur["beta"]
	phi = cur["phi"]
	L = cur["L"]

	phi_new = zeros(k)
	
	for l in 1:k

        ind = findall(L .== l)

        a_phi_new = sum(nu[ind])/2 + a_phi 

        c = log.(survival[ind]) .+ log.(epsilon[ind]) .- x[ind,:] * beta[l,:] .- log(theta[l]) 

        b_phi_new = 1 / (1 / b_phi + 0.5 * sum(u[ind] .* (c .^ 2)))


        if length(ind) == sum(nu[ind]) 
            new = rand(Gamma(a_phi_new, b_phi_new), 1)[1]
        else
            new = MH_phi_sampler(phi[l]^2, nu[ind], c, a_phi_new, b_phi_new)
        end

		phi_new[l] = sqrt(new)
	end

	return phi_new 
end 