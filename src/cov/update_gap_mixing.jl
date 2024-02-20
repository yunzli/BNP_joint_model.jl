using PolyaGammaDistribution

function update_gap_pg_lantent(datC, datT, cur) 

    survivalC = datC["survival"]
    gapC = datC["gap"]
    NvecC = datC["Nvec"]
    NC = datC["N"]
    nC = datC["n"]
    iotaC = datC["iota"]

    survivalT = datT["survival"]
    gapT = datT["gap"]
    NvecT = datT["Nvec"]
    NT = datT["N"]
    nT = datT["n"]
    iotaT = datT["iota"]

    xi = cur["xi"]
    xiC = xi[1]
    xiT = xi[2]

    tU = cur["tU"]
    tUC = tU[1]
    tUT = tU[2]
    
    eta, lambda, gamma = cur["eta"], cur["lambda"], cur["gamma"]

    varsigmaC = zeros(NC+nC)
    varsigmaT = zeros(NT+nT)

    for h in 1:(NC+nC)
        tUh = tUC[h]
        a = 1 + iotaC[h]
        if h <= NC
            i, j = h2ij(h, NvecC)
            b = eta[tUh] * (log(gapC[i][j]) + log(xiC[i]) - log(lambda[tUh]))
            varsigmaC[h] = rand(PolyaGamma(a, b), 1)[1]
        else
            i = h - NC
            if NvecC[i] == 0
                b = eta[tUh] * (log(survivalC[i]) + log(xiC[i]) - log(lambda[tUh]))
            else
                b = eta[tUh] * (log(survivalC[i] - sum(gapC[i])) + log(xiC[i]) - log(lambda[tUh]))
            end
        
            varsigmaC[h] = rand(PolyaGamma(a, b), 1)[1]
        end 
    end

    for h in 1:(NT+nT)
        tUh = tUT[h]
        a = 1 + iotaT[h]
        if h <= NT
            i, j = h2ij(h, NvecT)
            b = eta[tUh] * (log(gapT[i][j]) - gamma[tUh] + log(xiT[i]) - log(lambda[tUh]))
            varsigmaT[h] = rand(PolyaGamma(a, b), 1)[1]
        else
            i = h - NT
            if NvecT[i] == 0
                b = eta[tUh] * (log(survivalT[i]) - gamma[tUh] + log(xiT[i]) - log(lambda[tUh]))
            else
                b = eta[tUh] * (log(survivalT[i]-sum(gapT[i])) - gamma[tUh] + log(xiT[i]) - log(lambda[tUh]))
            end
        
            varsigmaT[h] = rand(PolyaGamma(a, b), 1)[1]
        end 
    end

    varsigma = [varsigmaC, varsigmaT]
    return varsigma
end

function update_lambda(datC, datT, cur, hyper)

    tU = cur["tU"]
    tUC = tU[1]
    tUT = tU[2]

    g = cur["g"]

    varsigma = cur["varsigma"]
    varsigmaC = varsigma[1]
    varsigmaT = varsigma[2]

    eta = cur["eta"]
    gamma = cur["gamma"]

    xi = cur["xi"]
    xiC = xi[1]
    xiT = xi[2]
    
    survivalC = datC["survival"]
    NC = datC["N"]
    gapC = datC["gap"]
    iotaC = datC["iota"]
    NvecC = datC["Nvec"]
    
    survivalT = datT["survival"]
    NT = datT["N"]
    gapT = datT["gap"]
    iotaT = datT["iota"]
    NvecT = datT["Nvec"]

    sigma2_lambda = hyper["sigma2_lambda"]
    mu_lambda = cur["mu_lambda"]

	lambda_new = zeros(g)
    for l in 1:g 
        indC = findall(tUC .== l)
        indT = findall(tUT .== l)

        sigma2_lambda_new_inv = 1/sigma2_lambda + eta[l]^2 * (sum(varsigmaC[indC]) + sum(varsigmaT[indT]))
        sigma2_lambda_new = 1 / sigma2_lambda_new_inv 

        tmp1 = mu_lambda / sigma2_lambda
        tmptmp2 = 0
        for h in indC
            if h <= NC 
                i,j = h2ij(h, NvecC)
                tmptmp2 += varsigmaC[h] * (log(gapC[i][j]) + log(xiC[i]))
            else
                i = h - NC
                if NvecC[i] == 0
                    tmptmp2 += varsigmaC[h] * (log(survivalC[i]) + log(xiC[i]))
                else
                    tmptmp2 += varsigmaC[h] * (log(survivalC[i]-sum(gapC[i])) + log(xiC[i]))
                end
            end
        end
        
        for h in indT
            if h <= NT 
                i,j = h2ij(h, NvecT)
                tmptmp2 += varsigmaT[h] * (log(gapT[i][j]) + log(xiT[i]) - gamma[l])
            else
                i = h - NT
                if NvecT[i] == 0
                    tmptmp2 += varsigmaT[h] * (log(survivalT[i]) + log(xiT[i]) - gamma[l])
                else
                    tmptmp2 += varsigmaT[h] * (log(survivalT[i]-sum(gapT[i])) + log(xiT[i]) - gamma[l])
                end
            end
        end

        tmp2 = eta[l]^2 * tmptmp2
        tmp3 = 0.5 * eta[l] * (sum(1 .- iotaC[indC]) + sum(1 .- iotaT[indT]))
        mu_lambda_new = sigma2_lambda_new * (tmp1 + tmp2 + tmp3) 
        lambda_new[l] = exp(rand(Normal(mu_lambda_new, sqrt(sigma2_lambda_new)), 1)[1])

    end

    return lambda_new 
end

function MH_eta_sampler(eta2, iotaC, iotaT, cC, cT, a_eta, b_eta)

	eta2prop = exp(log(eta2) + rand(Normal(0, 1),1)[1])

    tmp = 0.0
    if length(iotaC) > 0 
        tmp += sum((iotaC .- 1) .* cC)
    end
    if length(iotaT) > 0
        tmp += sum((iotaT .- 1) .* cT)
    end
	logcur = logpdf(Gamma(a_eta,b_eta), eta2) + 0.5 * tmp * sqrt(eta2) + log(eta2) 
	logpro = logpdf(Gamma(a_eta,b_eta), eta2prop) + 0.5 * tmp * sqrt(eta2prop) + log(eta2prop) 

	if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
		return eta2prop
	else
		return eta2
	end 
end 

function update_eta(datC, datT, cur, hyper)

	iotaC = datC["iota"] 
    gapC = datC["gap"]
    NvecC = datC["Nvec"]
    survivalC = datC["survival"]
    NC = datC["N"]

	iotaT = datT["iota"] 
    gapT = datT["gap"]
    NvecT = datT["Nvec"]
    survivalT = datT["survival"]
    NT = datT["N"]

	a_eta = hyper["a_eta"]
	b_eta = cur["b_eta"]
	varsigma = cur["varsigma"]
    varsigmaC = varsigma[1]
    varsigmaT = varsigma[2]

    xi = cur["xi"]
    xiC = xi[1]
    xiT = xi[2]

    g = cur["g"]
	lambda = cur["lambda"]
    gamma = cur["gamma"]
	eta = cur["eta"]
	tU = cur["tU"]
    tUC = tU[1]
    tUT = tU[2]

	eta_new = zeros(g)
	
	for l in 1:g

        indC = findall(tUC .== l)
        indT = findall(tUT .== l)

        a_eta_new = (sum(iotaC[indC]) + sum(iotaT[indT]))/2 + a_eta 

        cC = [] 
        cT = [] 
        for h in indC
            tmp = 0.0
            if h <= NC
                i,j = h2ij(h, NvecC) 
                tmp = log(gapC[i][j]) + log(xiC[i]) - log(lambda[l])
            else
                i = h - NC
                if NvecC[i] == 0
                    tmp = log(survivalC[i]) + log(xiC[i]) - log(lambda[l])
                else
                    tmp = log(survivalC[i]-sum(gapC[i])) + log(xiC[i]) - log(lambda[l])
                end
            end
            push!(cC, tmp)
        end 

        for h in indT
            tmp = 0.0
            if h <= NT
                i,j = h2ij(h, NvecT) 
                tmp = log(gapT[i][j]) - gamma[l] + log(xiT[i]) - log(lambda[l])
            else
                i = h - NT
                if NvecT[i] == 0
                    tmp = log(survivalT[i]) - gamma[l] + log(xiT[i]) - log(lambda[l])
                else
                    tmp = log(survivalT[i]-sum(gapT[i])) - gamma[l] + log(xiT[i]) - log(lambda[l])
                end
            end
            push!(cT, tmp)
        end 
        # c = log.(gap[ind]) .+ log(epsilon[ind]) .- log(theta[l]) 

        # println(length(indC), " ", length(indT))
        tmp = 0.0 
        if length(indC) > 0 
            tmp += sum(varsigmaC[indC] .* (cC .^ 2))
        end 
        if length(indT) > 0
            tmp += sum(varsigmaT[indT] .* (cT .^ 2))
        end
        
        # b_eta_new = 1 / (1 / b_eta + 0.5 * (sum(varsigmaC[indC] .* (cC .^ 2)) + sum(varsigmaT[indT] .* (cT .^ 2))))
        b_eta_new = 1 / (1 / b_eta + 0.5 * tmp)

        if (sum(iotaC[indC]) == length(indC)) & (sum(iotaT[indT]) == length(indT))
            new = rand(Gamma(a_eta_new, b_eta_new), 1)[1]
        else
            new = MH_eta_sampler(eta[l]^2, iotaC[indC], iotaT[indT], cC, cT, a_eta_new, b_eta_new)
        end

		eta_new[l] = sqrt(new)
	end

	return eta_new 
end 


function update_gamma(datC, datT, cur, hyper)
    
    iotaC = datC["iota"]
    survivalC = datC["survival"]
    gapC = datC["gap"]
    NC = datC["N"]
    nC = datC["n"]
    NvecC = datC["Nvec"]
    
    iotaT = datT["iota"]
    survivalT = datT["survival"]
    gapT = datT["gap"]
    NT = datT["N"]
    nT = datT["n"]
    NvecT = datT["Nvec"]

    sigma2_gamma = hyper["sigma2_gamma"]
    mu_gamma = cur["mu_gamma"]

    tU = cur["tU"]
    tUC = tU[1] 
    tUT = tU[2] 

    varsigma = cur["varsigma"]
    varsigmaC = varsigma[1]
    varsigmaT = varsigma[2]

    xi = cur["xi"]
    xiC = xi[1]
    xiT = xi[2]

    g = cur["g"]
    lambda = cur["lambda"]
    eta = cur["eta"]

    gamma_new = zeros(g)

    for l in 1:g 
        indT = findall(tUT .== l)

        sigma2_gamma_new_inv = 1 / sigma2_gamma + eta[l]^2 * sum(varsigmaT[indT])
        sigma2_gamma_new = 1/ sigma2_gamma_new_inv

        tmp1 = 0.0
        tmp2 = 0.0
        for h in indT 
            if h <= NT 
                i,j = h2ij(h, NvecT)
                tau_i = gapT[i][j]
            else
                i = h - NT 
                if NvecT[i] == 0
                    tau_i = survivalT[i]
                else
                    tau_i = survivalT[i] - sum(gapT[i])
                end
            end 
            tmp1 += varsigmaT[h] * ( log(tau_i) + log(xiT[i]) - log(lambda[l]) ) 
            tmp2 += (1 - iotaT[h]) 
        end

        mu_gamma_new = sigma2_gamma_new * (mu_gamma/sigma2_gamma + eta[l]^2*tmp1 + 0.5*eta[l]*tmp2)

        gamma_new[l] = rand(Normal(mu_gamma_new, sqrt(sigma2_gamma_new)), 1)[1]
    end 

    return gamma_new
end