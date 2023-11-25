using PolyaGammaDistribution

function update_gap_pg_lantent(dat, cur) 

    survival = dat["survival"]
    gap = dat["gap"]
    xi = cur["xi"]
    Nvec = dat["Nvec"]
    N = dat["N"]
    n = dat["n"]

    iota = dat["iota"]
    tU = cur["tU"]
    
    eta, lambda = cur["eta"], cur["lambda"]

    varsigma = zeros(N+n)

    for h in 1:(N+n)
        tUh = tU[h]
        a = 1 + iota[h]
        if h <= N
            i, j = h2ij(h, Nvec)
            b = eta[tUh] * (log(gap[i][j]) + log(xi[i]) - log(lambda[tUh]))
            varsigma[h] = rand(PolyaGamma(a, b), 1)[1]
        else
            i = h - N 
            if Nvec[i] == 0
                b = eta[tUh] * (log(survival[i]) + log(xi[i]) - log(lambda[tUh]))
            else
                b = eta[tUh] * (log(survival[i]-sum(gap[i])) + log(xi[i]) - log(lambda[tUh]))
            end
        
            varsigma[h] = rand(PolyaGamma(a, b), 1)[1]
        end 
    end

    return varsigma
end

function update_lambda(dat, cur, hyper)

    tU = cur["tU"]
    g = cur["g"]

    varsigma = cur["varsigma"]
    eta = cur["eta"]
    xi = cur["xi"]
    
    survival = dat["survival"]
    N = dat["N"]
    gap = dat["gap"]
    iota = dat["iota"]
    Nvec = dat["Nvec"]

    sigma2_lambda = hyper["sigma2_lambda"]
    mu_lambda = cur["mu_lambda"]

	lambda_new = zeros(g)
    for l in 1:g 
        ind = findall(tU .== l)

        sigma2_lambda_new_inv = 1/sigma2_lambda + eta[l]^2 * sum(varsigma[ind])
        sigma2_lambda_new = 1 / sigma2_lambda_new_inv 

        tmp1 = mu_lambda / sigma2_lambda
        tmptmp2 = 0
        for h in ind 
            if h <= N 
                i,j = h2ij(h, Nvec)
                tmptmp2 += varsigma[h] * (log(gap[i][j]) + log(xi[i]))
            else
                i = h - N
                if Nvec[i] == 0
                    tmptmp2 += varsigma[h] * (log(survival[i]) + log(xi[i]))
                else
                    tmptmp2 += varsigma[h] * (log(survival[i]-sum(gap[i])) + log(xi[i]))
                end
            end
        end
        tmp2 = eta[l]^2 * tmptmp2
        tmp3 = 0.5 * eta[l] * sum(1 .- iota[ind])
        mu_lambda_new = sigma2_lambda_new * (tmp1 + tmp2 + tmp3) 
        lambda_new[l] = exp(rand(Normal(mu_lambda_new, sqrt(sigma2_lambda_new)), 1)[1])

    end

    return lambda_new 
end

function MH_eta_sampler(eta2, iota, c, a_eta, b_eta)

	eta2prop = exp(log(eta2) + rand(Normal(0, 1),1)[1])

	logcur = logpdf(Gamma(a_eta,b_eta), eta2) + 0.5 * sum((iota .- 1) .* c) * sqrt(eta2) + log(eta2) 
	logpro = logpdf(Gamma(a_eta,b_eta), eta2prop) + 0.5 * sum((iota .- 1) .* c) * sqrt(eta2prop) + log(eta2prop) 

	if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
		return eta2prop
	else
		return eta2
	end 
end 

function update_eta(dat, cur, hyper)

	iota = dat["iota"] 
    gap = dat["gap"]
    Nvec = dat["Nvec"]
    survival = dat["survival"]
    N = dat["N"]

	a_eta = hyper["a_eta"]
	b_eta = cur["b_eta"]
	varsigma = cur["varsigma"]
    xi = cur["xi"]

    g = cur["g"]
	lambda = cur["lambda"]
	eta = cur["eta"]
	tU = cur["tU"]

	eta_new = zeros(g)
	
	for l in 1:g

        ind = findall(tU .== l)

        a_eta_new = sum(iota[ind])/2 + a_eta 

        c = [] 
        for h in ind 
            tmp = 0.0
            if h <= N
                i,j = h2ij(h, Nvec) 
                tmp = log(gap[i][j]) + log(xi[i]) - log(lambda[l])
            else
                i = h - N 
                if Nvec[i] == 0
                    tmp = log(survival[i]) + log(xi[i]) - log(lambda[l])
                else
                    tmp = log(survival[i]-sum(gap[i])) + log(xi[i]) - log(lambda[l])
                end
            end
            push!(c, tmp)
        end 
        # c = log.(gap[ind]) .+ log(epsilon[ind]) .- log(theta[l]) 

        b_eta_new = 1 / (1 / b_eta + 0.5 * sum(varsigma[ind] .* (c .^ 2)))


        if sum(iota[ind]) == length(ind) 
            new = rand(Gamma(a_eta_new, b_eta_new), 1)[1]
        else
            new = MH_eta_sampler(eta[l]^2, iota[ind], c, a_eta_new, b_eta_new)
        end

		eta_new[l] = sqrt(new)
	end

	return eta_new 
end 