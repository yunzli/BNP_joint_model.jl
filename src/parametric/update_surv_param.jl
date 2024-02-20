function surv_loglikelihood(theta, phi, epsilon, survival, nu)

    n = length(survival)

    l = 0.0
    for i in 1:n
        if nu[i] == 1
            l = l + logpdf(LogLogistic(theta/epsilon[i], phi), survival[i])
        else
            l = l + logccdf(LogLogistic(theta/epsilon[i], phi), survival[i])
        end
    end 

    return l
end 

function update_theta(dat, cur, hyper, sig)

    theta_cur = cur["theta"]
    theta_pro = exp(log(theta_cur) + rand(Normal(0, sig), 1)[1])

    survivals = dat["survival"]
    nus = dat["nu"]
    epsilon = cur["epsilon"]
    phi = cur["phi"]

    s_theta = hyper["s_theta"]
    S_theta = hyper["S_theta"]

    lcur = logpdf(LogNormal(s_theta, sqrt(S_theta)), theta_cur) + log(theta_cur)
    lpro = logpdf(LogNormal(s_theta, sqrt(S_theta)), theta_pro) + log(theta_pro)
    # println(theta_cur, " ", theta_pro, " ", phi)
    # println(lcur, " ", lpro)

    # surv_likelihood(theta, phi, epsilon, survival, nu)
    lcur = lcur + surv_loglikelihood(theta_cur, phi, epsilon, survivals, nus)
    lpro = lpro + surv_loglikelihood(theta_pro, phi, epsilon, survivals, nus)
    # println(lcur, " ", lpro)
    # println(" ")


    if log(rand(Uniform(0,1), 1)[1]) < (lpro - lcur)
        acc = 1
        theta = theta_pro
    else
        acc = 0
        theta = theta_cur
    end

    return [theta, acc]
end 

function update_phi(dat, cur, hyper, sig)

    phi_cur = cur["phi"]
    phi_pro = exp(log(phi_cur) + rand(Normal(0, sig), 1)[1])

    survivals = dat["survival"]
    nus = dat["nu"]
    epsilon = cur["epsilon"]
    theta = cur["theta"]

    a_phi = hyper["a_phi"]
    b_phi = hyper["b_phi"]

    lcur = logpdf(Gamma(a_phi, b_phi), phi_cur) + log(phi_cur)
    lpro = logpdf(Gamma(a_phi, b_phi), phi_pro) + log(phi_pro)
    # println(phi_cur, " ", phi_pro, " ", theta)
    # println(lcur, " ", lpro)

    # surv_likelihood(theta, phi, epsilon, survival, nu)
    lcur = lcur + surv_loglikelihood(theta, phi_cur, epsilon, survivals, nus)
    lpro = lpro + surv_loglikelihood(theta, phi_pro, epsilon, survivals, nus)
    # println(lcur, " ", lpro)
    # println("")

    if log(rand(Uniform(0,1), 1)[1]) < (lpro - lcur)
        acc = 1
        phi = phi_pro
    else
        acc = 0
        phi = phi_cur
    end

    return [phi, acc]
end