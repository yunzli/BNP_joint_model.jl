function gap_loglikelihood(lambda, eta, xi, survival, gap, Nvec)
    
    n = length(survival)

    l = 0.0
    for i in 1:n
        if Nvec[i] > 0
            for j in 1:Nvec[i]
                l = l + logpdf(LogLogistic(lambda/xi[i], eta), gap[i][j])
            end
            l = l + logccdf(LogLogistic(lambda/xi[i], eta), survival[i]-sum(gap[i]))
        else
            l = l + logccdf(LogLogistic(lambda/xi[i], eta), survival[i])
        end
    end

    return l
end

function update_lambda(dat, cur, hyper)
    survivals = dat["survival"]
    gaps = dat["gap"]
    Nvec = dat["Nvec"]

    eta = cur["eta"]
    xi = cur["xi"]

    s_lambda = hyper["s_lambda"]
    S_lambda = hyper["S_lambda"]

    lambda_cur = cur["lambda"]
    lambda_pro = exp(log(lambda_cur) + rand(Normal(0,0.1),1)[1])

    lcur = logpdf(LogNormal(s_lambda, sqrt(S_lambda)), lambda_cur) + log(lambda_cur)
    lpro = logpdf(LogNormal(s_lambda, sqrt(S_lambda)), lambda_pro) + log(lambda_pro)

    # gap_likelihood(lambda, eta, xi, survival, gap, Nvec)
    lcur = lcur + gap_loglikelihood(lambda_cur, eta, xi, survivals, gaps, Nvec)
    lpro = lpro + gap_loglikelihood(lambda_pro, eta, xi, survivals, gaps, Nvec)

    if log(rand(Uniform(0,1), 1)[1]) < (lpro - lcur)
        lambda = lambda_pro
    else
        lambda = lambda_cur
    end

    return lambda
end

function update_eta(dat, cur, hyper)
    survivals = dat["survival"]
    gaps = dat["gap"]
    Nvec = dat["Nvec"]

    xi = cur["xi"]
    lambda = cur["lambda"]

    a_eta = hyper["a_eta"]
    b_eta = hyper["b_eta"]

    eta_cur = cur["eta"]
    eta_pro = exp(log(eta_cur) + rand(Normal(0, 0.1),1)[1])

    lcur = logpdf(Gamma(a_eta, b_eta), eta_cur) + log(eta_cur)
    lpro = logpdf(Gamma(a_eta, b_eta), eta_pro) + log(eta_pro)

    lcur = lcur + gap_loglikelihood(lambda, eta_cur, xi, survivals, gaps, Nvec)
    lpro = lpro + gap_loglikelihood(lambda, eta_pro, xi, survivals, gaps, Nvec)

    if log(rand(Uniform(0,1), 1)[1]) < (lpro - lcur)
        eta = eta_pro
    else
        eta = eta_cur
    end

    return eta
end 