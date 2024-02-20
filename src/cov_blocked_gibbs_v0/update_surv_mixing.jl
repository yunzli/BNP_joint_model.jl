using PolyaGammaDistribution

function update_surv_pg_lantent(datC, datT, cur) 

    survivalC = datC["survival"]
    survivalT = datT["survival"]

    nC = datC["n"]
    nT = datT["n"]

    xC = datC["x"]
    xT = datT["x"]

    epsilonC = cur["epsilonC"]
    epsilonT = cur["epsilonT"]
    
    nuC = datC["nu"]
    nuT = datT["nu"]

    LC = cur["LC"]
    LT = cur["LT"]
    
    phi, theta, beta = cur["phi"], cur["theta"], cur["beta"]

    uC = zeros(nC)
    uT = zeros(nT)

    for i in 1:nC
        Li = LC[i]
        a = 1 + nuC[i]
        b = phi[Li] * (log(survivalC[i]) - xC[i,:]'*beta[Li,:] + log(epsilonC[i]) - log(theta[Li]))
        uC[i] = rand(PolyaGamma(a, b), 1)[1]
    end

    for i in 1:nT
        Li = LT[i]
        a = 1 + nuT[i]
        b = phi[Li] * (log(survivalT[i]) - xT[i,:]'*beta[Li,:] + log(epsilonT[i]) - log(theta[Li]))
        uT[i] = rand(PolyaGamma(a, b), 1)[1]
    end

    return Dict("uC" => uC, "uT" => uT)
end

function update_theta(datC, datT, cur, hyper)

    LC = cur["LC"]
    LT = cur["LT"]

    BG = hyper["BG"]

    uC = cur["uC"]
    uT = cur["uT"]

    phi = cur["phi"]
    beta = cur["beta"]

    epsilonC = cur["epsilonC"]
    epsilonT = cur["epsilonT"]

    survivalC = datC["survival"]
    nuC = datC["nu"]
    xC = datC["x"]
    
    survivalT = datT["survival"]
    nuT = datT["nu"]
    xT = datT["x"]

    sigma2_theta = hyper["sigma2_theta"]
    mu_theta = cur["mu_theta"]

	theta_new = zeros(BG)
    for l in 1:BG
        indC = findall(LC .== l)
        indT = findall(LT .== l)

        if (length(indC) == 0) & (length(indT) == 0)
            new = exp(rand(Normal(mu_theta, sqrt(sigma2_theta)), 1)[1])
        else
            sigma2_theta_new_inv = 1/sigma2_theta + phi[l]^2 * (sum(uC[indC]) + sum(uT[indT]))
            sigma2_theta_new = 1/sigma2_theta_new_inv

            tmp1 = mu_theta / sigma2_theta
            tmp2 = phi[l]^2 * (sum(uC[indC] .* (log.(survivalC[indC]) .- xC[indC,:]*beta[l,:] .+ log.(epsilonC[indC]))) + sum(uT[indT] .* (log.(survivalT[indT]) .- xT[indT,:]*beta[l,:] .+ log.(epsilonT[indT]))))
            tmp3 = 0.5 * phi[l] * (length(indC) - sum(nuC[indC]) + length(indT) - sum(nuT[indT]))
            mu_theta_new = sigma2_theta_new * (tmp1 + tmp2 + tmp3)
            new = exp(rand(Normal(mu_theta_new, sqrt(sigma2_theta_new)), 1)[1])
        end
        theta_new[l] = new
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
    xC = datC["x"]

	nuT = datT["nu"] 
    survivalT = datT["survival"]
    xT = datT["x"]

	a_phi = hyper["a_phi"]
	b_phi = cur["b_phi"]
	uC = cur["uC"]
	uT = cur["uT"]

    epsilonC = cur["epsilonC"]
    epsilonT = cur["epsilonT"]

	LC = cur["LC"]
	LT = cur["LT"]

    BG = hyper["BG"]

	theta = cur["theta"]
    beta = cur["beta"]
	phi = cur["phi"]

	phi_new = zeros(BG)
	
	for l in 1:BG

        indC = findall(LC .== l)
        indT = findall(LT .== l)

        if (length(indC) == 0) & (length(indT) == 0)
            new = rand(Gamma(a_phi, b_phi), 1)[1]
        else
            a_phi_new = (sum(nuC[indC]) + sum(nuT[indT]))/2 + a_phi

            cC = log.(survivalC[indC]) .+ log.(epsilonC[indC]) .- xC[indC,:]*beta[l,:] .- log(theta[l])
            cT = log.(survivalT[indT]) .+ log.(epsilonT[indT]) .- xT[indT,:]*beta[l,:] .- log(theta[l])

            b_phi_new = 1 / (1 / b_phi + 0.5 * (sum(uC[indC] .* (cC .^ 2)) + sum(uT[indT] .* (cT .^ 2))))

            if (length(indC) == sum(nuC[indC])) & (length(indT) == sum(nuT[indT]))
                new = rand(Gamma(a_phi_new, b_phi_new), 1)[1]
            else
                new = MH_phi_sampler(phi[l]^2, nuC[indC], nuT[indT], cC, cT, a_phi_new, b_phi_new)
            end
        end

		phi_new[l] = sqrt(new)
	end

	return phi_new 
end 


function update_beta(datC, datT, cur, hyper)

    LC = cur["LC"]
    LT = cur["LT"]

    nuC = datC["nu"]
    survivalC = datC["survival"]
    xC = datC["x"]

    nuT = datT["nu"]
    survivalT = datT["survival"]
    xT = datT["x"]

    p = datC["p"]

    Sigma_beta = hyper["Sigma_beta"]
    Sigma_beta_inv = hyper["Sigma_beta_inv"]
    mu_beta = cur["mu_beta"]

    uC = cur["uC"]
    uT = cur["uT"]

    epsilonC = cur["epsilonC"]
    epsilonT = cur["epsilonT"] 

    BG = hyper["BG"]
    theta = cur["theta"]
    phi = cur["phi"]

    beta_new = zeros(BG, p)

    for l in 1:BG

        indC = findall(LC .== l)
        indT = findall(LT .== l)

        if (length(indC) == 0) & (length(indT) == 0)
            new = rand(MvNormal(vec(mu_beta), Sigma_beta), 1)
        else
            ezz = zeros(p, p)
            Asum = zeros(p)
            Bsum = zeros(p)
            
            for i in indC
                ezz += uC[i] * (xC[i,:] * xC[i,:]')
                Asum += uC[i] * (log(survivalC[i]) + log(epsilonC[i]) - log(theta[l])) * xC[i,:]
                Bsum += (1 - nuC[i]) * xC[i,:]
            end
            for i in indT
                ezz += uT[i] * (xT[i,:] * xT[i,:]')
                Asum += uT[i] * (log(survivalT[i]) + log(epsilonT[i]) - log(theta[l])) * xT[i,:]
                Bsum += (1 - nuT[i]) * xT[i,:]
            end

            Sigma_beta_new_inv = phi[l]^2 * ezz + Sigma_beta_inv 
            Sigma_beta_new = svd2inv(Sigma_beta_new_inv)

            mu_beta_new = Sigma_beta_new * (Sigma_beta_inv * mu_beta + phi[l]^2 * Asum + 0.5 * phi[l] * Bsum)

            new = rand(MvNormal(vec(mu_beta_new), Sigma_beta_new), 1)
        end

        beta_new[l,:] = new
    end

    return beta_new
end