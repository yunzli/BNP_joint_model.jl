function update_random_effects(dat, cur, hyper)
    epsilon = cur["epsilon"]
    xi = cur["xi"]

    L = cur["L"]
    tU = cur["tU"]

    theta = cur["theta"]
    beta = cur["beta"]
    phi = cur["phi"]
    lambda = cur["lambda"]
    gamma = cur["gamma"]
    eta = cur["eta"]

    survival = dat["survival"]
    gap = dat["gap"]

    Nvec = dat["Nvec"]
    N = dat["N"]
    n = dat["n"]
    nu = dat["nu"]

    x = dat["x"]
    z = dat["z"]

    Sigma_e_1 = cur["Sigma_e_1"]
    Sigma_e_2 = cur["Sigma_e_2"]
    epsilon_new = zeros(n)
    xi_new = zeros(n)

    for i in 1:n
        walk = rand(MvNormal(zeros(2), 0.1 * Matrix(Diagonal(ones(2)))), 1)[:,1]

        epsilon_i_cur = epsilon[i]
        epsilon_i_pro = exp(log(epsilon_i_cur) + walk[1])

        xi_i_cur = xi[i]
        xi_i_pro = exp(log(xi_i_cur) + walk[2])

        if x[i][1] == 0
            Sigma_e = Sigma_e_1 
        else 
            Sigma_e = Sigma_e_2
        end

        l_cur = logpdf(MvLogNormal(zeros(2), Sigma_e), [epsilon_i_cur, xi_i_cur]) + log(epsilon_i_cur) + log(xi_i_cur)
        l_pro = logpdf(MvLogNormal(zeros(2), Sigma_e), [epsilon_i_pro, xi_i_pro]) + log(epsilon_i_pro) + log(xi_i_pro)
        
        Li = L[i]
        if nu[i] == 1
            l_cur += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*x[i,:])/epsilon_i_cur, phi[Li]), survival[i])
            l_pro += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*x[i,:])/epsilon_i_pro, phi[Li]), survival[i])
        else 
            l_cur += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*x[i,:])/epsilon_i_cur, phi[Li]), survival[i])
            l_pro += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*x[i,:])/epsilon_i_pro, phi[Li]), survival[i])
        end

        if Nvec[i] > 0
            for j in 1:Nvec[i]
                h = ij2h(i,j,Nvec)
                tUh = tU[h]
                l_cur += logpdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*z[i,:])/xi_i_cur, eta[tUh]), gap[i][j])
                l_pro += logpdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*z[i,:])/xi_i_pro, eta[tUh]), gap[i][j])
            end
        end

        h = N + i 
        tUh = tU[h]
        if Nvec[i] == 0 # N
            l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*z[i,:])/xi_i_cur, eta[tUh]), survival[i])
            l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*z[i,:])/xi_i_pro, eta[tUh]), survival[i])
        else
            l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*z[i,:])/xi_i_cur, eta[tUh]), survival[i]-sum(gap[i]))
            l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*z[i,:])/xi_i_pro, eta[tUh]), survival[i]-sum(gap[i]))
        end 

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


    x = dat["x"]
    n = dat["n"]
    
    c_e_1_new = c_e 
    c_e_2_new = c_e 

    C_e_1_new = C_e 
    C_e_2_new = C_e 

    for i in 1:n 
        if x[i][1] == 0
            tmp_1 = [log(epsilon[i]), log(xi[i])]
            c_e_1_new += 1
            C_e_1_new += tmp_1 * tmp_1' 
        else
            tmp_2 = [log(epsilon[i]), log(xi[i])]
            c_e_2_new += 1
            C_e_2_new += tmp_2 * tmp_2' 
        end
    end 

    new_1 = rand(InverseWishart(c_e_1_new, C_e_1_new), 1)[1]
    new_2 = rand(InverseWishart(c_e_2_new, C_e_2_new), 1)[1]
    return [new_1, new_2] 
end 

function update_random_effects_latent(dat, cur, hyper)

    survival = dat["survival"]
    x = dat["x"]
    nu = dat["nu"]
    L = cur["L"]
    theta = cur["theta"]
    beta = cur["beta"]
    phi = cur["phi"]
    epsilon = cur["epsilon"]

    gap = dat["gap"]
    z = dat["z"]
    iota = dat["iota"]
    tU = cur["tU"]
    lambda = cur["lambda"]
    gamma = cur["gamma"]
    eta = cur["eta"]
    xi = cur["xi"]

    n = dat["n"]
    N = dat["N"]
    Nvec = dat["Nvec"]

    epsilon_new = zeros(n)
    xi_new = zeros(N+n)
    for i in 1:n 
        Li = L[i]
        u_b = phi[Li] * (log(survival[i]) - beta[Li,:]*x[i,:] + log(epsilon[i]) - log(theta[Li]))
        epsilon_new[i] = rand(PolyaGamma(1+nu[i], u_b), 1)
    end 

    for h in 1:N 
    end 
    for h in 1:n 
    end

end 