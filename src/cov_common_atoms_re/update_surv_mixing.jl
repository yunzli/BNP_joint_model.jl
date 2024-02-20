using PolyaGammaDistribution

function update_surv_pg_lantent(datC, datT, cur) 

    survivalC = datC["survival"]
    survivalT = datT["survival"]
    nC = datC["n"]
    nT = datT["n"]
    epsilon = cur["epsilon"]
    epsilonC = epsilon[1]
    epsilonT = epsilon[2]
    nuC = datC["nu"]
    nuT = datT["nu"]
    L = cur["L"]
    LC = L[1]
    LT = L[2]
    
    phi, theta = cur["phi"], cur["theta"]

    uC = zeros(nC)
    uT = zeros(nT)

    for i in 1:nC
        Li = LC[i]
        a = 1 + nuC[i]
        b = phi[Li] * (log(survivalC[i]) + log(epsilonC[i]) - log(theta[Li]))
        uC[i] = rand(PolyaGamma(a, b), 1)[1]
    end

    for i in 1:nT
        Li = LT[i]
        a = 1 + nuT[i]
        b = phi[Li] * (log(survivalT[i]) + log(epsilonT[i]) - log(theta[Li]))
        uT[i] = rand(PolyaGamma(a, b), 1)[1]
    end

    u = [uC, uT]
    return u 
end

function update_theta(datC, datT, cur, hyper)

    L = cur["L"]
    LC = L[1]
    LT = L[2]

    nl = cur["nl"]
    nlC = nl[1]
    nlT = nl[2]

    k = cur["k"]

    u = cur["u"]
    uC = u[1]
    uT = u[2]

    phi = cur["phi"]
    epsilon = cur["epsilon"]
    epsilonC = epsilon[1]
    epsilonT = epsilon[2]
    
    survivalC = datC["survival"]
    nuC = datC["nu"]
    
    survivalT = datT["survival"]
    nuT = datT["nu"]

    sigma2_theta = hyper["sigma2_theta"]
    mu_theta = cur["mu_theta"]

	theta_new = zeros(k)
    for l in 1:k 
        indC = findall(LC .== l)
        indT = findall(LT .== l)

        sigma2_theta_new_inv = 1/sigma2_theta + phi[l]^2 * (sum(uC[indC]) + sum(uT[indT]))
        sigma2_theta_new = 1 / sigma2_theta_new_inv 

        tmp1 = mu_theta / sigma2_theta
        tmp2 = phi[l]^2 * (sum(uC[indC] .* (log.(survivalC[indC]) .+ log.(epsilonC[indC]))) + sum(uT[indT] .* (log.(survivalT[indT]) .+ log.(epsilonT[indT]))))
        tmp3 = 0.5 * phi[l] * (nlC[l] - sum(nuC[indC]) + nlT[l] - sum(nuT[indT]))
        mu_theta_new = sigma2_theta_new * (tmp1 + tmp2 + tmp3) 
        theta_new[l] = exp(rand(Normal(mu_theta_new, sqrt(sigma2_theta_new)), 1)[1])

    end

    return theta_new 
end

function MH_phi_sampler(phi2, nuC, nuT, cC, cT, a_phi, b_phi)

	phi2prop = exp(log(phi2) + rand(Normal(0, 1),1)[1])

	logcur = logpdf(Gamma(a_phi,b_phi), phi2) + 0.5 * (sum((nuC .- 1) .* cC) + sum((nuT .- 1) .* cT)) * sqrt(phi2) + log(phi2) 
	logpro = logpdf(Gamma(a_phi,b_phi), phi2prop) + 0.5 * (sum((nuC .- 1) .* cC) + sum((nuT .- 1) .* cT)) * sqrt(phi2prop) + log(phi2prop) 

	if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
		return phi2prop
	else
		return phi2
	end 
end 

function update_phi(datC, datT, cur, hyper)

	nuC = datC["nu"] 
    survivalC = datC["survival"]

	nuT = datT["nu"] 
    survivalT = datT["survival"]

	a_phi = hyper["a_phi"]
	b_phi = cur["b_phi"]
	u = cur["u"]
    uC = u[1]
    uT = u[2]
    epsilon = cur["epsilon"]
    epsilonC = epsilon[1]
    epsilonT = epsilon[2]

    k = cur["k"]
	theta = cur["theta"]
	phi = cur["phi"]
	L = cur["L"]
    LC = L[1]
    LT = L[2]

	phi_new = zeros(k)
	
	for l in 1:k

        indC = findall(LC .== l)
        indT = findall(LT .== l)

        a_phi_new = (sum(nuC[indC]) + sum(nuT[indT]))/2 + a_phi 

        cC = log.(survivalC[indC]) .+ log.(epsilonC[indC]) .- log(theta[l]) 
        cT = log.(survivalT[indT]) .+ log.(epsilonT[indT]) .- log(theta[l]) 

        b_phi_new = 1 / (1 / b_phi + 0.5 * (sum(uC[indC] .* (cC .^ 2)) + sum(uT[indT] .* (cT .^ 2))))


        if length(indC) == sum(nuC[indC]) & length(indT) == sum(nuT[indT])
            new = rand(Gamma(a_phi_new, b_phi_new), 1)[1]
        else
            new = MH_phi_sampler(phi[l]^2, nuC[indC], nuT[indT], cC, cT, a_phi_new, b_phi_new)
        end

		phi_new[l] = sqrt(new)
	end

	return phi_new 
end 