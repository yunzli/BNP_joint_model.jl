function update_random_effects(dat, cur, hyper, Sig)
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


# function update_random_effects_direct(dat, cur, hyper)
#     epsilon = cur["epsilon"]
#     xi = cur["xi"]

#     L = cur["L"]
#     tU = cur["tU"]

#     theta = cur["theta"]
#     phi = cur["phi"]
#     lambda = cur["lambda"]
#     eta = cur["eta"]

#     survival = dat["survival"]
#     gap = dat["gap"]

#     Nvec = dat["Nvec"]
#     N = dat["N"]
#     nu = dat["nu"]
#     n = dat["n"]
#     iota = dat["iota"]

#     Sigma_e = cur["Sigma_e"]
#     epsilon_new = zeros(n)
#     xi_new = zeros(n)

#     for i in 1:n 
#         Li = L[i]
#         a_u_tmp = nu[i]+1
#         b_u_tmp = phi[Li] * (log(survival[i]) + log(epsilon[i]) - log(theta[Li]))
#         u_tmp = rand(PolyaGamma(a_u_tmp, b_u_tmp), 1)[1]

#         mu_epsilon_tmp = xi[i] * Sigma_e[2,1] / Sigma_e[2,2]
#         sigma2_epsilon_tmp = Sigma_e[1,1] - Sigma_e[1,2] * Sigma_e[2,1] / Sigma_e[2,2] 

#         sigma2_epsilon_tilde = 1 / (u_tmp * phi[Li]^2 + 1 / sigma2_epsilon_tmp)

#         mu_epsilon_tilde = sigma2_epsilon_tilde * (mu_epsilon_tmp / sigma2_epsilon_tmp + phi[Li]^2 * (log(theta[Li]) - log(survival[i])) + 0.5*(1-nu[i])*phi[Li] )

#         epsilon_new[i] = rand(LogNormal(mu_epsilon_tilde, sqrt(sigma2_epsilon_tilde)), 1)[1]


#         mu_xi_tmp = epsilon_new[i] * Sigma_e[1,2] / Sigma_e[1,1] 
#         sigma2_xi_tmp = Sigma_e[2,2] - Sigma_e[2,1] * Sigma_e[1,2] / Sigma_e[1,1]

#         mu_xi_tilde_right = 0.0 
#         sigma2_xi_tilde_inv = 0.0 

#         if Nvec[i] > 0
#             for j in 1:Nvec[i]
#                 h = ij2h(i,j,Nvec)
#                 tUh = tU[h]
#                 a_up_tmp = 1 + iota[h] 
#                 b_up_tmp = eta[tUh] * (log(gap[i][j]) + log(xi[i]) - log(lambda[tUh]))
#                 up_tmp = rand(PolyaGamma(a_up_tmp, b_up_tmp), 1)[1]

#                 sigma2_xi_tilde_inv += up_tmp * eta[tUh]^2
#                 mu_xi_tilde_right += eta[tUh]^2 * (log(lambda[tUh]) - log(gap[i][j])) + 0.5 * (1-iota[h]) * eta[tUh]
#             end
#         end

#         h = N + i
#         tUh = tU[h]
#         a_up_tmp = 1 + iota[h] 
#         if Nvec[i] == 0
#             b_up_tmp = eta[tUh] * (log(survival[i]) + log(xi[i]) - log(lambda[tUh]))
#             up_tmp = rand(PolyaGamma(a_up_tmp, b_up_tmp), 1)[1]

#             sigma2_xi_tilde_inv += up_tmp * eta[tUh]^2
#             mu_xi_tilde_right += eta[tUh]^2 * (log(lambda[tUh]) - log(survival[i])) + 0.5 * (1-iota[h]) * eta[tUh]
#         else 
#             b_up_tmp = eta[tUh] * (log(survival[i]-sum(gap[i])) + log(xi[i]) - log(lambda[tUh]))
#             up_tmp = rand(PolyaGamma(a_up_tmp, b_up_tmp), 1)[1]

#             sigma2_xi_tilde_inv += up_tmp * eta[tUh]^2
#             mu_xi_tilde_right += eta[tUh]^2 * (log(lambda[tUh]) - log(survival[i]-sum(gap[i]))) + 0.5 * (1-iota[h]) * eta[tUh]
#         end 

#         sigma2_xi_tilde = 1 / (sigma2_xi_tilde_inv + 1 / sigma2_xi_tmp) 
#         mu_xi_tilde = sigma2_xi_tilde * (mu_xi_tmp / sigma2_xi_tmp + mu_xi_tilde_right)

#         xi_new[i] = rand(LogNormal(mu_xi_tilde, sqrt(sigma2_xi_tilde)), 1)[1]
#         if xi_new[i] == Inf 
#             xi_new[i] = Base.maximum(xi_new)
#         end

#         # println(
#         #     epsilon_new[i], " ",
#         #     xi_new[i], " ", 
#         #     Sigma_e, " ",
#         #     mu_xi_tmp, " ", 
#         #     sigma2_xi_tmp, " ", 
#         #     mu_xi_tilde, " ", 
#         #     sigma2_xi_tilde
#         # )
#     end 

#     # println(epsilon_new)
#     # println(xi_new)
#     # println("")
#     return [epsilon_new, xi_new]
# end


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