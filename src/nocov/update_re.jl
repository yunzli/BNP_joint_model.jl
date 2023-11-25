function update_random_effects(dat, cur, hyper)
    epsilon = cur["epsilon"]
    xi = cur["xi"]

    L = cur["L"]
    tU = cur["tU"]

    theta = cur["theta"]
    phi = cur["phi"]
    lambda = cur["lambda"]
    eta = cur["eta"]

    survival = dat["survival"]
    gap = dat["gap"]

    Nvec = dat["Nvec"]
    N = dat["N"]
    n = length(survival)
    nu = dat["nu"]

    Sigma_e = cur["Sigma_e"]
    epsilon_new = zeros(n)
    xi_new = zeros(n)

    for i in 1:n
        walk = rand(MvNormal(zeros(2), 0.1 * Matrix(Diagonal(ones(2)))), 1)[:,1]

        epsilon_i_cur = epsilon[i]
        epsilon_i_pro = exp(log(epsilon_i_cur) + walk[1])

        xi_i_cur = xi[i]
        xi_i_pro = exp(log(xi_i_cur) + walk[2])

        l_cur = logpdf(MvLogNormal(zeros(2), Sigma_e), [epsilon_i_cur, xi_i_cur]) + log(epsilon_i_cur) + log(xi_i_cur)
        l_pro = logpdf(MvLogNormal(zeros(2), Sigma_e), [epsilon_i_pro, xi_i_pro]) + log(epsilon_i_pro) + log(xi_i_pro)
        
        Li = L[i]
        if nu[i] == 1
            l_cur += logpdf(LogLogistic(theta[Li]/epsilon_i_cur, phi[Li]), survival[i])
            l_pro += logpdf(LogLogistic(theta[Li]/epsilon_i_pro, phi[Li]), survival[i])
        else 
            l_cur += logccdf(LogLogistic(theta[Li]/epsilon_i_cur, phi[Li]), survival[i])
            l_pro += logccdf(LogLogistic(theta[Li]/epsilon_i_pro, phi[Li]), survival[i])
        end

        if Nvec[i] > 0
            for j in 1:Nvec[i]
                h = ij2h(i,j,Nvec)
                tUh = tU[h]
                l_cur += logpdf(LogLogistic(lambda[tUh]/xi_i_cur, eta[tUh]), gap[i][j])
                l_pro += logpdf(LogLogistic(lambda[tUh]/xi_i_pro, eta[tUh]), gap[i][j])
            end
        end

        h = N + i 
        tUh = tU[h]
        if Nvec[i] == 0 # N
            l_cur += logccdf(LogLogistic(lambda[tUh]/xi_i_cur, eta[tUh]), survival[i])
            l_pro += logccdf(LogLogistic(lambda[tUh]/xi_i_pro, eta[tUh]), survival[i])
        else
            l_cur += logccdf(LogLogistic(lambda[tUh]/xi_i_cur, eta[tUh]), survival[i]-sum(gap[i]))
            l_pro += logccdf(LogLogistic(lambda[tUh]/xi_i_pro, eta[tUh]), survival[i]-sum(gap[i]))
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