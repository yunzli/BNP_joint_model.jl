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

    xiC = cur["xiC"]
    xiT = cur["xiT"]

    tUC = cur["tUC"]
    tUT = cur["tUT"]
    
    eta, lambda= cur["eta"], cur["lambda"]

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
                b = eta[tUh] * (log(survivalC[i]-sum(gapC[i])) + log(xiC[i]) - log(lambda[tUh]))
            end
        
            varsigmaC[h] = rand(PolyaGamma(a, b), 1)[1]
        end 
    end

    for h in 1:(NT+nT)
        tUh = tUT[h]
        a = 1 + iotaT[h]
        if h <= NT
            i, j = h2ij(h, NvecT)
            b = eta[tUh] * (log(gapT[i][j]) + log(xiT[i]) - log(lambda[tUh]))
            varsigmaT[h] = rand(PolyaGamma(a, b), 1)[1]
        else
            i = h - NT 
            if NvecT[i] == 0
                b = eta[tUh] * (log(survivalT[i]) + log(xiT[i]) - log(lambda[tUh]))
            else
                b = eta[tUh] * (log(survivalT[i]-sum(gapT[i])) + log(xiT[i]) - log(lambda[tUh]))
            end
        
            varsigmaT[h] = rand(PolyaGamma(a, b), 1)[1]
        end 
    end

    return Dict("varsigmaC" => varsigmaC, "varsigmaT" => varsigmaT)
end

function update_lambda(datC, datT, cur, hyper)

    tUC = cur["tUC"]
    tUT = cur["tUT"]

    xiC = cur["xiC"]
    xiT = cur["xiT"]

    BH = hyper["BH"]

    varsigmaC = cur["varsigmaC"]
    varsigmaT = cur["varsigmaT"]

    eta = cur["eta"]
    
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

	lambda_new = zeros(BH)
    for l in 1:BH 
        indC = findall(tUC .== l)
        indT = findall(tUT .== l)

        if (length(indC) == 0) & (length(indT) == 0)
            new = exp(rand(Normal(mu_lambda, sqrt(sigma2_lambda)), 1)[1])
        else
            sigma2_lambda_new_inv = 1/sigma2_lambda + eta[l]^2 * ( sum(varsigmaC[indC]) + sum(varsigmaT[indT]) )
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
                    tmptmp2 += varsigmaT[h] * (log(gapT[i][j]) + log(xiT[i]))
                else
                    i = h - NT
                    if NvecT[i] == 0
                        tmptmp2 += varsigmaT[h] * (log(survivalT[i]) + log(xiT[i]))
                    else
                        tmptmp2 += varsigmaT[h] * (log(survivalT[i]-sum(gapT[i])) + log(xiT[i]))
                    end
                end
            end
            tmp2 = eta[l]^2 * tmptmp2
            tmp3 = 0.5 * eta[l] * ( sum(1 .- iotaC[indC]) + sum(1 .- iotaT[indT]) )
            mu_lambda_new = sigma2_lambda_new * (tmp1 + tmp2 + tmp3) 
            new = exp(rand(Normal(mu_lambda_new, sqrt(sigma2_lambda_new)), 1)[1])
        end
        lambda_new[l] = new
    end

    return lambda_new 
end

function MH_eta_sampler(eta2, iotaC, iotaT, cC, cT, a_eta, b_eta)

	eta2pro = exp(log(eta2) + rand(Normal(0, 1),1)[1])

    logcur = logpdf(Gamma(a_eta, b_eta), eta2) + log(eta2)
    logpro = logpdf(Gamma(a_eta, b_eta), eta2pro) + log(eta2pro)

    if length(iotaC) > 0
        logcur += 0.5 * sum((iotaC .- 1) .* cC) * sqrt(eta2)
        logpro += 0.5 * sum((iotaC .- 1) .* cC) * sqrt(eta2pro)
    end
    if length(iotaT) > 0
        logcur += 0.5 * sum((iotaT .- 1) .* cT) * sqrt(eta2)
        logpro += 0.5 * sum((iotaT .- 1) .* cT) * sqrt(eta2pro)
    end

	if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
		return eta2pro
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

	varsigmaC = cur["varsigmaC"]
	varsigmaT = cur["varsigmaT"]
    xiC = cur["xiC"]
    xiT = cur["xiT"]
	tUC = cur["tUC"]
	tUT = cur["tUT"]

	a_eta = hyper["a_eta"]
	b_eta = cur["b_eta"]

	lambda = cur["lambda"]
	eta = cur["eta"]

    BH = hyper["BH"]

	eta_new = zeros(BH)
	
	for l in 1:BH

        indC = findall(tUC .== l)
        indT = findall(tUT .== l)

        if (length(indC) == 0) & (length(indT) == 0)
            new = rand(Gamma(a_eta, b_eta), 1)[1]
        else

            a_eta_new = (sum(iotaC[indC]) + sum(iotaT[indT])) /2 + a_eta 

            cC = [] 
            for h in indC
                tmp = 0.0
                if h <= NC
                    i,j = h2ij(h, NvecC)
                    twh = gapC[i][j]
                else
                    i = h - NC
                    if NvecC[i] == 0
                        twh = survivalC[i]
                    else
                        twh = survivalC[i] - sum(gapC[i])
                    end
                end
                tmp = log(twh) + log(xiC[i]) - log(lambda[l]) 
                push!(cC, tmp)
            end 

            cT = [] 
            for h in indT
                tmp = 0.0
                if h <= NT
                    i,j = h2ij(h, NvecT)
                    twh = gapT[i][j]
                else
                    i = h - NT
                    if NvecT[i] == 0
                        twh = survivalT[i]
                    else
                        twh = survivalT[i] - sum(gapT[i])
                    end
                end
                tmp = log(twh) + log(xiT[i]) - log(lambda[l]) 
                push!(cT, tmp)
            end 

            b_eta_new_inv = 1 / b_eta
            if length(indC) > 0
                b_eta_new_inv += 0.5 * sum(varsigmaC[indC] .* (cC .^ 2))  
            end
            if length(indT) > 0
                b_eta_new_inv += 0.5 * sum(varsigmaT[indT] .* (cT .^ 2))  
            end
            b_eta_new = 1 / b_eta_new_inv

            if (sum(iotaC[indC]) == length(indC)) & (sum(iotaT[indT]) == length(indT))
                new = rand(Gamma(a_eta_new, b_eta_new), 1)[1]
            else
                new = MH_eta_sampler(eta[l]^2, iotaC[indC], iotaT[indT], cC, cT, a_eta_new, b_eta_new)
            end
        end

		eta_new[l] = sqrt(new)
	end

	return eta_new 
end 