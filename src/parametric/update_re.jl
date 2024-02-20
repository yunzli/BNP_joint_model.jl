function re_loglikelihood(theta, phi, lambda, eta, epsilon, xi, survival, nu, gap, Nvec)

    l = 0.0
    if nu == 1
        l = l + logpdf(LogLogistic(theta/epsilon, phi), survival)
    else 
        l = l + logccdf(LogLogistic(theta/epsilon, phi), survival)
    end

    if Nvec > 0
        l = l + logccdf(LogLogistic(lambda/xi, eta), survival-sum(gap))
        for i in 1:Nvec
            l = l + logpdf(LogLogistic(lambda/xi, eta), gap[i])
        end
    else
        l = l + logccdf(LogLogistic(lambda/xi, eta), survival)
    end

    return l
end

function update_random_effects(dat, cur, hyper, Sig)
    epsilon = cur["epsilon"]
    xi = cur["xi"]

    theta = cur["theta"]
    phi = cur["phi"]
    lambda = cur["lambda"]
    eta = cur["eta"]

    survival = dat["survival"]
    gap = dat["gap"]
    Nvec = dat["Nvec"]
    nu = dat["nu"]

    n = length(survival)
    Sigma_e = cur["Sigma_e"]
    epsilon_new = zeros(n)
    xi_new = zeros(n)

    for i in 1:n
        
        tmp = rand(Uniform(0,1), 1)[1]
        if tmp < 0.95 
            walk = rand(MvNormal(zeros(2), 2.38^2/2*Sig[i]), 1)[:,1]
        else 
            walk = rand(MvNormal(zeros(2), 0.01/2*Diagonal(ones(2))), 1)[:,1]
        end 

        epsilon_i_cur = epsilon[i]
        epsilon_i_pro = exp(log(epsilon_i_cur) + walk[1])

        xi_i_cur = xi[i]
        xi_i_pro = exp(log(xi_i_cur) + walk[2])

        l_cur = logpdf(MvLogNormal(zeros(2), Sigma_e), [epsilon_i_cur, xi_i_cur]) + log(epsilon_i_cur) + log(xi_i_cur)
        l_pro = logpdf(MvLogNormal(zeros(2), Sigma_e), [epsilon_i_pro, xi_i_pro]) + log(epsilon_i_pro) + log(xi_i_pro)
        
        l_cur = l_cur + re_loglikelihood(theta, phi, lambda, eta, epsilon_i_cur, xi_i_cur, survival[i], nu[i], gap[i], Nvec[i])
        l_pro = l_pro + re_loglikelihood(theta, phi, lambda, eta, epsilon_i_pro, xi_i_pro, survival[i], nu[i], gap[i], Nvec[i])

        if log(rand(Uniform(0,1),1)[1]) < (l_pro - l_cur)
			epsilon_new[i] = epsilon_i_pro 
            xi_new[i] = xi_i_pro
        else
			epsilon_new[i] = epsilon_i_cur 
            xi_new[i] = xi_i_cur
		end 
    end

    return [epsilon_new, xi_new]
end


function update_Sigma_e(dat, cur, hyper)
    
    epsilon = cur["epsilon"]
    xi = cur["xi"]
    c_e = hyper["c_e"]
    C_e = hyper["C_e"]

    n = length(dat["survival"])
    
    c_e_new = c_e + n 
    C_e_new = C_e 
    for i in 1:n 
        tmp = [log(epsilon[i]), log(xi[i])]
        C_e_new += tmp * tmp' 
    end 

    new = rand(InverseWishart(c_e_new, C_e_new), 1)[1]
    return new 
end 