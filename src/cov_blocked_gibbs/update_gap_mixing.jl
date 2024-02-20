using PolyaGammaDistribution

function update_gap_pg_lantent(datC, datT, cur) 

    survivalC = datC["survival"]
    gapC = datC["gap"]
    NvecC = datC["Nvec"]
    NC = datC["N"]
    nC = datC["n"]
    zC = datC["z"]
    iotaC = datC["iota"]

    survivalT = datT["survival"]
    gapT = datT["gap"]
    NvecT = datT["Nvec"]
    NT = datT["N"]
    nT = datT["n"]
    zT = datT["z"]
    iotaT = datT["iota"]

    xiC = cur["xiC"]
    xiT = cur["xiT"]

    tUC = cur["tUC"]
    tUT = cur["tUT"]
    
    eta, lambda, gamma = cur["eta"], cur["lambda"], cur["gamma"]

    varsigmaC = zeros(NC+nC)
    varsigmaT = zeros(NT+nT)

    for h in 1:(NC+nC)
        tUh = tUC[h]
        a = 1 + iotaC[h]
        if h <= NC
            i, j = h2ij(h, NvecC)
            b = eta[tUh] * (log(gapC[i][j]) - zC[i,:]'*gamma[tUh,:] + log(xiC[i]) - log(lambda[tUh]))
            varsigmaC[h] = rand(PolyaGamma(a, b), 1)[1]
        else
            i = h - NC 
            if NvecC[i] == 0
                b = eta[tUh] * (log(survivalC[i]) - zC[i,:]'*gamma[tUh,:] + log(xiC[i]) - log(lambda[tUh]))
            else
                b = eta[tUh] * (log(survivalC[i]-sum(gapC[i])) - zC[i,:]'*gamma[tUh,:] + log(xiC[i]) - log(lambda[tUh]))
            end
        
            varsigmaC[h] = rand(PolyaGamma(a, b), 1)[1]
        end 
    end

    for h in 1:(NT+nT)
        tUh = tUT[h]
        a = 1 + iotaT[h]
        if h <= NT
            i, j = h2ij(h, NvecT)
            b = eta[tUh] * (log(gapT[i][j]) + log(xiT[i]) - log(lambda[tUh]) - zT[i,:]'*gamma[tUh,:])
            varsigmaT[h] = rand(PolyaGamma(a, b), 1)[1]
        else
            i = h - NT 
            if NvecT[i] == 0
                b = eta[tUh] * (log(survivalT[i]) + log(xiT[i]) - log(lambda[tUh]) - zT[i,:]'*gamma[tUh,:])
            else
                b = eta[tUh] * (log(survivalT[i]-sum(gapT[i])) + log(xiT[i]) - log(lambda[tUh]) - zT[i,:]'
                *gamma[tUh,:])
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
    gamma = cur["gamma"]
    
    survivalC = datC["survival"]
    NC = datC["N"]
    zC = datC["z"]
    gapC = datC["gap"]
    iotaC = datC["iota"]
    NvecC = datC["Nvec"]
    
    survivalT = datT["survival"]
    NT = datT["N"]
    zT = datT["z"]
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
                    tmptmp2 += varsigmaC[h] * (log(gapC[i][j]) + log(xiC[i]) - gamma[l,:]'*zC[i,:])
                else
                    i = h - NC
                    if NvecC[i] == 0
                        tmptmp2 += varsigmaC[h] * (log(survivalC[i]) + log(xiC[i]) - gamma[l,:]'*zC[i,:])
                    else
                        tmptmp2 += varsigmaC[h] * (log(survivalC[i]-sum(gapC[i])) + log(xiC[i]) - gamma[l,:]'*zC[i,:])
                    end
                end
            end
            for h in indT 
                if h <= NT 
                    i,j = h2ij(h, NvecT)
                    tmptmp2 += varsigmaT[h] * (log(gapT[i][j]) + log(xiT[i]) - gamma[l,:]'*zT[i,:])
                else
                    i = h - NT
                    if NvecT[i] == 0
                        tmptmp2 += varsigmaT[h] * (log(survivalT[i]) + log(xiT[i]) - gamma[l,:]'*zT[i,:])
                    else
                        tmptmp2 += varsigmaT[h] * (log(survivalT[i]-sum(gapT[i])) + log(xiT[i]) - gamma[l,:]'*zT[i,:])
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
    zC = datC["z"]

	iotaT = datT["iota"] 
    gapT = datT["gap"]
    NvecT = datT["Nvec"]
    survivalT = datT["survival"]
    NT = datT["N"]
    zT = datT["z"]

	varsigmaC = cur["varsigmaC"]
	varsigmaT = cur["varsigmaT"]
    xiC = cur["xiC"]
    xiT = cur["xiT"]
	tUC = cur["tUC"]
	tUT = cur["tUT"]

	a_eta = hyper["a_eta"]
	b_eta = cur["b_eta"]

	lambda = cur["lambda"]
    gamma = cur["gamma"]
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
                tmp = log(twh) - gamma[l,:]' * zC[i,:] + log(xiC[i]) - log(lambda[l]) 
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
                tmp = log(twh) - gamma[l,:]' * zT[i,:] + log(xiT[i]) - log(lambda[l]) 
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

function update_gamma(datC, datT, cur, hyper)

    iotaC = datC["iota"]
    survivalC = datC["survival"]
    gapC = datC["gap"]
    NC = datC["N"]
    NvecC = datC["Nvec"]
    zC = datC["z"]

    iotaT = datT["iota"]
    survivalT = datT["survival"]
    gapT = datT["gap"]
    NT = datT["N"]
    NvecT = datT["Nvec"]
    zT = datT["z"]

    q = datC["q"]

    tUC = cur["tUC"]
    varsigmaC = cur["varsigmaC"]
    xiC = cur["xiC"]

    tUT = cur["tUT"]
    varsigmaT = cur["varsigmaT"]
    xiT = cur["xiT"]

    Sigma_gamma = hyper["Sigma_gamma"]
    Sigma_gamma_inv = hyper["Sigma_gamma_inv"]
    mu_gamma = cur["mu_gamma"]

    BH = hyper["BH"]
    lambda = cur["lambda"]
    eta = cur["eta"]

    gamma_new = zeros(BH, q)

    for l in 1:BH 
        indC = findall(tUC .== l)
        indT = findall(tUT .== l)

        if (length(indC) == 0) & (length(indT) == 0)
            new = rand(MvNormal(vec(mu_gamma), Sigma_gamma), 1)
        else
            vzz = zeros(q, q)
            tmp1 = zeros(q)
            tmp2 = zeros(q)

            for h in indC 
                if h <= NC 
                    i,j = h2ij(h, NvecC)
                    tau_i = gapC[i][j]
                else
                    i = h - NC
                    if NvecC[i] == 0
                        tau_i = survivalC[i]
                    else
                        tau_i = survivalC[i] - sum(gapC[i])
                    end
                end 
                vzz += varsigmaC[h] * (zC[i,:] * zC[i,:]')
                tmp1 += varsigmaC[h] * ( log(tau_i) + log(xiC[i]) - log(lambda[l]) )  * zC[i,:]
                tmp2 += (1 - iotaC[h]) * zC[i,:]
            end

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
                vzz += varsigmaT[h] * (zT[i,:] * zT[i,:]')
                tmp1 += varsigmaT[h] * ( log(tau_i) + log(xiT[i]) - log(lambda[l]) )  * zT[i,:]
                tmp2 += (1 - iotaT[h]) * zT[i,:]
            end

            Sigma_gamma_inv_new = Sigma_gamma_inv + eta[l]^2 * vzz
            Sigma_gamma_new = svd2inv(Sigma_gamma_inv_new)

            mu_gamma_new = Sigma_gamma_new * (Sigma_gamma_inv * mu_gamma + eta[l]^2 * tmp1 + 0.5 * eta[l] * tmp2)

            if !isposdef(Sigma_gamma_new)
                println(minimum(varsigma))
                println(eigvals(Sigma_gamma_new))
                println(Sigma_gamma_new)
            end
            new = rand(MvNormal(vec(mu_gamma_new), Sigma_gamma_new), 1)
        end

        gamma_new[l,:] = new
    end 

    return gamma_new
end


# using PolyaGammaDistribution

# function update_gap_pg_lantent(datC, datT, cur) 

#     survivalC = datC["survival"]
#     gapC = datC["gap"]
#     NvecC = datC["Nvec"]

#     NC = datC["N"]
#     nC = datC["n"]
#     zC = datC["z"]

#     iotaC = datC["iota"]
#     xiC = cur["xiC"]
#     tUC = cur["tUC"]

#     survivalT = datT["survival"]
#     gapT = datT["gap"]
#     NvecT = datT["Nvec"]

#     NT = datT["N"]
#     nT = datT["n"]
#     zT = datT["z"]

#     iotaT = datT["iota"]
#     xiT = cur["xiT"]
#     tUT = cur["tUT"]
    
#     eta, lambda, gamma = cur["eta"], cur["lambda"], cur["gamma"]

#     varsigmaC = zeros(NC+nC)
#     varsigmaT = zeros(NT+nT)

#     for h in 1:(NC+nC)
#         tUh = tUC[h]
#         a = 1 + iotaC[h]
#         if h <= NC
#             i, j = h2ij(h, NvecC)
#             obs = gapC[i][j]
#         else
#             i = h - NC 
#             if NvecC[i] == 0
#                 obs = survivalC[i]
#             else
#                 obs = survivalC[i] - sum(gapC[i])
#             end
#         end 
#         b = eta[tUh] * (log(obs) - gamma[tUh,:]'*zC[i,:] + log(xiC[i]) - log(lambda[tUh]))
#         varsigmaC[h] = rand(PolyaGamma(a, b), 1)[1]
#     end

#     for h in 1:(NT+nT)
#         tUh = tUT[h]
#         a = 1 + iotaT[h]
#         if h <= NT
#             i, j = h2ij(h, NvecT)
#             obs = gapT[i][j]
#         else
#             i = h - NT 
#             if NvecT[i] == 0
#                 obs = survivalT[i]
#             else
#                 obs = survivalT[i] - sum(gapT[i])
#             end
#         end 
#         b = eta[tUh] * (log(obs) - gamma[tUh,:]'*zT[i,:] + log(xiT[i]) - log(lambda[tUh]))
#         varsigmaT[h] = rand(PolyaGamma(a, b), 1)[1]
#     end

#     return Dict("varsigmaC" => varsigmaC, "varsigmaT" => varsigmaT)
# end

# function update_lambda(datC, datT, cur, hyper)

#     tUC = cur["tUC"]
#     tUT = cur["tUT"]

#     varsigmaC = cur["varsigmaC"]
#     varsigmaT = cur["varsigmaT"]

#     gamma, eta = cur["gamma"], cur["eta"]

#     xiC = cur["xiC"]
#     xiT = cur["xiT"]

#     survivalC = datC["survival"]
#     NC = datC["N"]
#     zC = datC["z"]
#     gapC = datC["gap"]
#     iotaC = datC["iota"]
#     NvecC = datC["Nvec"]

#     survivalT = datT["survival"]
#     NT = datT["N"]
#     zT = datT["z"]
#     gapT = datT["gap"]
#     iotaT = datT["iota"]
#     NvecT = datT["Nvec"]

#     sigma2_lambda = hyper["sigma2_lambda"]
#     mu_lambda = cur["mu_lambda"]

#     BH = hyper["BH"]

# 	lambda_new = zeros(BH)
#     for l in 1:BH 
#         indC = findall(tUC .== l)
#         indT = findall(tUT .== l)

#         if (length(indC)==0) & (length(indT) == 0)
#             new = exp(rand(Normal(mu_lambda, sqrt(sigma2_lambda)), 1)[1])
#         else
#             sigma2_lambda_new_inv = 1/sigma2_lambda + eta[l]^2 * (sum(varsigmaC[indC]) + sum(varsigmaT[indT]))
#             sigma2_lambda_new = 1 / sigma2_lambda_new_inv 
    
#             tmp1 = mu_lambda / sigma2_lambda
#             tmptmp2 = 0
#             for h in indC
#                 if h <= NC 
#                     i,j = h2ij(h, NvecC)
#                     tmptmp2 += varsigmaC[h] * (log(gapC[i][j]) + log(xiC[i]) - gamma[l,:]'*zC[i,:])
#                 else
#                     i = h - NC
#                     if NvecC[i] == 0
#                         tmptmp2 += varsigmaC[h] * (log(survivalC[i]) + log(xiC[i]) - gamma[l,:]'*zC[i,:])
#                     else
#                         tmptmp2 += varsigmaC[h] * (log(survivalC[i]-sum(gapC[i])) + log(xiC[i]) - gamma[l,:]'*zC[i,:])
#                     end
#                 end
#             end

#             for h in indT
#                 if h <= NT 
#                     i,j = h2ij(h, NvecT)
#                     tmptmp2 += varsigmaT[h] * (log(gapT[i][j]) + log(xiT[i]) - gamma[l,:]'*zT[i,:])
#                 else
#                     i = h - NT
#                     if NvecT[i] == 0
#                         tmptmp2 += varsigmaT[h] * (log(survivalT[i]) + log(xiT[i]) - gamma[l,:]'*zT[i,:])
#                     else
#                         tmptmp2 += varsigmaT[h] * (log(survivalT[i]-sum(gapT[i])) + log(xiT[i]) - gamma[l,:]'*zT[i,:])
#                     end
#                 end
#             end
#             tmp2 = eta[l]^2 * tmptmp2
#             tmp3 = 0.5 * eta[l] * (sum(1 .- iotaC[indC]) + sum(1 .- iotaT[indT]))
#             mu_lambda_new = sigma2_lambda_new * (tmp1 + tmp2 + tmp3)
#             new = exp(rand(Normal(mu_lambda_new, sqrt(sigma2_lambda_new)), 1)[1])
#         end
#         lambda_new[l] = new
#     end

#     return lambda_new 
# end

# function MH_eta_sampler(eta2, iotaC, iotaT, cC, cT, a_eta, b_eta)

# 	eta2prop = exp(log(eta2) + rand(Normal(0, 1),1)[1])

#     tmp = 0.0
#     if length(iotaC) > 0 
#         tmp += sum((iotaC .- 1) .* cC)
#     end
#     if length(iotaT) > 0
#         tmp += sum((iotaT .- 1) .* cT)
#     end
# 	logcur = logpdf(Gamma(a_eta,b_eta), eta2) + 0.5 * tmp * sqrt(eta2) + log(eta2) 
# 	logpro = logpdf(Gamma(a_eta,b_eta), eta2prop) + 0.5 * tmp * sqrt(eta2prop) + log(eta2prop) 

# 	if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
# 		return eta2prop
# 	else
# 		return eta2
# 	end 
# end 

# function update_eta(datC, datT, cur, hyper)

# 	iotaC = datC["iota"] 
#     gapC = datC["gap"]
#     NvecC = datC["Nvec"]
#     survivalC = datC["survival"]
#     NC = datC["N"]
#     zC = datC["z"]

# 	iotaT = datT["iota"] 
#     gapT = datT["gap"]
#     NvecT = datT["Nvec"]
#     survivalT = datT["survival"]
#     NT = datT["N"]
#     zT = datT["z"]

# 	a_eta = hyper["a_eta"]
# 	b_eta = cur["b_eta"]
# 	varsigmaC = cur["varsigmaC"]
# 	varsigmaT = cur["varsigmaT"]

#     xiC = cur["xiC"]
#     xiT = cur["xiT"]

# 	lambda = cur["lambda"]
# 	eta = cur["eta"]
#     gamma = cur["gamma"]

# 	tUC = cur["tUC"]
# 	tUT = cur["tUT"]

#     BH = hyper["BH"]
# 	eta_new = zeros(BH)
	
# 	for l in 1:BH

#         indC = findall(tUC .== l)
#         indT = findall(tUT .== l)

#         if (length(indC) == 0) & (length(indT) == 0)
#             new = rand(Gamma(a_eta, b_eta), 1)[1]
#         else
#             a_eta_new = (sum(iotaC[indC]) + sum(iotaT[indT]))/2 + a_eta 

#             cC = [] 
#             cT = [] 
#             for h in indC
#                 tmp = 0.0
#                 if h <= NC
#                     i,j = h2ij(h, NvecC) 
#                     tmp = log(gapC[i][j]) - gamma[l,:]'*zC[i,:] + log(xiC[i]) - log(lambda[l])
#                 else
#                     i = h - NC
#                     if NvecC[i] == 0
#                         tmp = log(survivalC[i]) - gamma[l,:]'*zC[i,:] + log(xiC[i]) - log(lambda[l])
#                     else
#                         tmp = log(survivalC[i]-sum(gapC[i])) - gamma[l,:]'*zC[i,:] + log(xiC[i]) - log(lambda[l])
#                     end
#                 end
#                 push!(cC, tmp)
#             end 

#             for h in indT
#                 tmp = 0.0
#                 if h <= NT
#                     i,j = h2ij(h, NvecT) 
#                     tmp = log(gapT[i][j]) - gamma[l,:]'*zT[i,:] + log(xiT[i]) - log(lambda[l])
#                 else
#                     i = h - NT
#                     if NvecT[i] == 0
#                         tmp = log(survivalT[i]) - gamma[l,:]'*zT[i,:] + log(xiT[i]) - log(lambda[l])
#                     else
#                         tmp = log(survivalT[i]-sum(gapT[i])) - gamma[l,:]'*zT[i,:] + log(xiT[i]) - log(lambda[l])
#                     end
#                 end
#                 push!(cT, tmp)
#             end 

#             tmp = 0.0 
#             if length(indC) > 0 
#                 tmp += sum(varsigmaC[indC] .* (cC .^ 2))
#             end 
#             if length(indT) > 0
#                 tmp += sum(varsigmaT[indT] .* (cT .^ 2))
#             end
            
#             b_eta_new = 1 / (1 / b_eta + 0.5 * tmp)

#             if (sum(iotaC[indC]) == length(indC)) & (sum(iotaT[indT]) == length(indT))
#                 new = rand(Gamma(a_eta_new, b_eta_new), 1)[1]
#             else
#                 new = MH_eta_sampler(eta[l]^2, iotaC[indC], iotaT[indT], cC, cT, a_eta_new, b_eta_new)
#             end
#         end

#         eta_new[l] = sqrt(new)
# 	end

# 	return eta_new 
# end 


# function update_gamma(datC, datT, cur, hyper)

#     iotaC = datC["iota"]
#     survivalC = datC["survival"]
#     gapC = datC["gap"]
#     NC = datC["N"]
#     NvecC = datC["Nvec"]
#     zC = datC["z"]
#     q = datC["q"]

#     iotaT = datT["iota"]
#     survivalT = datT["survival"]
#     gapT = datT["gap"]
#     NT = datT["N"]
#     NvecT = datT["Nvec"]
#     zT = datT["z"]

#     Sigma_gamma = hyper["Sigma_gamma"]
#     Sigma_gamma_inv = hyper["Sigma_gamma_inv"]
#     mu_gamma = cur["mu_gamma"]
#     muSigma = Sigma_gamma_inv * mu_gamma 

#     tUC = cur["tUC"]
#     varsigmaC = cur["varsigmaC"]
#     xiC = cur["xiC"]

#     tUT = cur["tUT"]
#     varsigmaT = cur["varsigmaT"]
#     xiT = cur["xiT"]

#     lambda = cur["lambda"]
#     eta = cur["eta"]

#     BH = hyper["BH"]
#     gamma_new = zeros(BH, q)

#     for l in 1:BH
#         indC = findall(tUC .== l)
#         indT = findall(tUT .== l)

#         if (length(indC) == 0) & (length(indT) == 0)
#             new = rand(MvNormal(mu_gamma, Sigma_gamma), 1)
#         else
#             vzz = zeros(q, q)
#             tmp1 = zeros(q)
#             tmp2 = zeros(q)

#             for h in indC 
#                 if h <= NC 
#                     i,j = h2ij(h, NvecC)
#                     tau_i = gapC[i][j]
#                 else
#                     i = h - NC
#                     if NvecC[i] == 0
#                         tau_i = survivalC[i]
#                     else
#                         tau_i = survivalC[i] - sum(gapC[i])
#                     end
#                 end 
#                 vzz += varsigmaC[h] * (zC[i,:] * zC[i,:]')
#                 tmp1 += varsigmaC[h] * ( log(tau_i) + log(xiC[i]) - log(lambda[l]) )  * zC[i,:]
#                 tmp2 += (1 - iotaC[h]) * zC[i,:]
#             end
#             for h in indT 
#                 if h <= NT 
#                     i,j = h2ij(h, NvecT)
#                     tau_i = gapT[i][j]
#                 else
#                     i = h - NT
#                     if NvecT[i] == 0
#                         tau_i = survivalT[i]
#                     else
#                         tau_i = survivalT[i] - sum(gapT[i])
#                     end
#                 end 
#                 vzz += varsigmaT[h] * (zT[i,:] * zT[i,:]')
#                 tmp1 += varsigmaT[h] * ( log(tau_i) + log(xiT[i]) - log(lambda[l]) )  * zT[i,:]
#                 tmp2 += (1 - iotaT[h]) * zT[i,:]
#             end

#             Sigma_gamma_inv_new = Sigma_gamma_inv + eta[l]^2 * vzz
#             Sigma_gamma_new = svd2inv(Sigma_gamma_inv_new)

#             mu_gamma_new = Sigma_gamma_new * (muSigma + eta[l]^2 * tmp1 + 0.5 * eta[l] * tmp2)

#             new = rand(MvNormal(mu_gamma_new, Sigma_gamma_new), 1)
#         end

#         gamma_new[l,:] = new
#     end 

#     return gamma_new
# end