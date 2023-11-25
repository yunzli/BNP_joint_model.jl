using PolyaGammaDistribution

function update_surv_pg_lantent(dat, cur) 

    survival = dat["survival"]
    epsilon = cur["epsilon"]
    n = length(survival)
    nu = dat["nu"]
    L = cur["L"]
    
    phi, theta = cur["phi"], cur["theta"]

    u = zeros(n)

    for i in 1:n
        Li = L[i]
        a = 1 + nu[i]
        b = phi[Li] * (log(survival[i]) + log(epsilon[i]) - log(theta[Li]))
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
    epsilon = cur["epsilon"]
    
    survival = dat["survival"]
    nu = dat["nu"]

    sigma2_theta = hyper["sigma2_theta"]
    mu_theta = cur["mu_theta"]

	theta_new = zeros(k)
    for l in 1:k 
        ind = findall(L .== l)

        sigma2_theta_new_inv = 1/sigma2_theta + phi[l]^2 * sum(u[ind])
        sigma2_theta_new = 1 / sigma2_theta_new_inv 

        tmp1 = mu_theta / sigma2_theta
        tmp2 = phi[l]^2 * sum(u[ind] .* (log.(survival[ind]) .+ log.(epsilon[ind])))
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

function update_phi(dat, cur, hyper)

	nu = dat["nu"] 
    survival = dat["survival"]

	a_phi = hyper["a_phi"]
	b_phi = cur["b_phi"]
	u = cur["u"]
    epsilon = cur["epsilon"]

    k = cur["k"]
	theta = cur["theta"]
	phi = cur["phi"]
	L = cur["L"]

	phi_new = zeros(k)
	
	for l in 1:k

        ind = findall(L .== l)

        a_phi_new = sum(nu[ind])/2 + a_phi 

        c = log.(survival[ind]) .+ log.(epsilon[ind]) .- log(theta[l]) 

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