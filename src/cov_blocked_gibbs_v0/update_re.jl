function update_random_effects(datC, datT, cur, hyper, Sig)
    SigC = Sig[1]
    SigT = Sig[2]

    epsilonC = cur["epsilonC"]
    epsilonT = cur["epsilonT"]

    xiC = cur["xiC"]
    xiT = cur["xiT"]

    LC = cur["LC"]
    LT = cur["LT"]

    tUC = cur["tUC"]
    tUT = cur["tUT"]

    theta = cur["theta"]
    beta = cur["beta"]
    phi = cur["phi"]
    lambda = cur["lambda"]
    gamma = cur["gamma"]
    eta = cur["eta"]

    survivalC = datC["survival"]
    gapC = datC["gap"]
    NvecC = datC["Nvec"]
    NC = datC["N"]
    nC = datC["n"]
    nuC = datC["nu"]
    xC = datC["x"]
    zC = datC["z"]

    survivalT = datT["survival"]
    gapT = datT["gap"]
    NvecT = datT["Nvec"]
    NT = datT["N"]
    nT = datT["n"]
    nuT = datT["nu"]
    xT = datT["x"]
    zT = datT["z"]

    Sigma_e_1 = cur["Sigma_e_1"]
    Sigma_e_2 = cur["Sigma_e_2"]

    epsilonC_new = zeros(nC)
    xiC_new = zeros(nC)
    epsilonT_new = zeros(nT)
    xiT_new = zeros(nT)

    for i in 1:nC
        
        tmp = rand(Uniform(0,1), 1)[1]
        if tmp < 0.95 
            walk = rand(MvNormal(zeros(2), 2.38^2/2*SigC[i]), 1)[:,1]
        else 
            walk = rand(MvNormal(zeros(2), 0.01/2*Diagonal(ones(2))), 1)[:,1]
        end 

        epsilon_i_cur = epsilonC[i]
        epsilon_i_pro = exp(log(epsilon_i_cur) + walk[1])

        xi_i_cur = xiC[i]
        xi_i_pro = exp(log(xi_i_cur) + walk[2])

        l_cur = logpdf(MvLogNormal(zeros(2), Sigma_e_1), [epsilon_i_cur, xi_i_cur]) + log(epsilon_i_cur) + log(xi_i_cur)
        l_pro = logpdf(MvLogNormal(zeros(2), Sigma_e_1), [epsilon_i_pro, xi_i_pro]) + log(epsilon_i_pro) + log(xi_i_pro)
        
        Li = LC[i]
        if nuC[i] == 1
            l_cur += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xC[i,:])/epsilon_i_cur, phi[Li]), survivalC[i])
            l_pro += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xC[i,:])/epsilon_i_pro, phi[Li]), survivalC[i])
        else 
            l_cur += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xC[i,:])/epsilon_i_cur, phi[Li]), survivalC[i])
            l_pro += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xC[i,:])/epsilon_i_pro, phi[Li]), survivalC[i])
        end

        if NvecC[i] > 0
            for j in 1:NvecC[i]
                h = ij2h(i,j,NvecC)
                tUh = tUC[h]
                l_cur += logpdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zC[i,:])/xi_i_cur, eta[tUh]), gapC[i][j])
                l_pro += logpdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zC[i,:])/xi_i_pro, eta[tUh]), gapC[i][j])
            end
        end

        h = NC + i 
        tUh = tUC[h]
        if NvecC[i] == 0 # N
            l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zC[i,:])/xi_i_cur, eta[tUh]), survivalC[i])
            l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zC[i,:])/xi_i_pro, eta[tUh]), survivalC[i])
        else
            l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zC[i,:])/xi_i_cur, eta[tUh]), survivalC[i]-sum(gapC[i]))
            l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zC[i,:])/xi_i_pro, eta[tUh]), survivalC[i]-sum(gapC[i]))
        end 

        if log(rand(Uniform(0,1),1)[1]) < (l_pro - l_cur)
			epsilonC_new[i] = epsilon_i_pro 
            xiC_new[i] = xi_i_pro
        else
			epsilonC_new[i] = epsilon_i_cur 
            xiC_new[i] = xi_i_cur
		end 
    end

    for i in 1:nT
        
        tmp = rand(Uniform(0,1), 1)[1]
        if tmp < 0.95 
            walk = rand(MvNormal(zeros(2), 2.38^2/2*SigT[i]), 1)[:,1]
        else 
            walk = rand(MvNormal(zeros(2), 0.01/2*Diagonal(ones(2))), 1)[:,1]
        end 

        epsilon_i_cur = epsilonT[i]
        epsilon_i_pro = exp(log(epsilon_i_cur) + walk[1])

        xi_i_cur = xiT[i]
        xi_i_pro = exp(log(xi_i_cur) + walk[2])

        l_cur = logpdf(MvLogNormal(zeros(2), Sigma_e_2), [epsilon_i_cur, xi_i_cur]) + log(epsilon_i_cur) + log(xi_i_cur)
        l_pro = logpdf(MvLogNormal(zeros(2), Sigma_e_2), [epsilon_i_pro, xi_i_pro]) + log(epsilon_i_pro) + log(xi_i_pro)
        
        Li = LT[i]
        if nuT[i] == 1
            l_cur += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xT[i,:])/epsilon_i_cur, phi[Li]), survivalT[i])
            l_pro += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xT[i,:])/epsilon_i_pro, phi[Li]), survivalT[i])
        else 
            l_cur += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xT[i,:])/epsilon_i_cur, phi[Li]), survivalT[i])
            l_pro += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xT[i,:])/epsilon_i_pro, phi[Li]), survivalT[i])
        end

        if NvecT[i] > 0
            for j in 1:NvecT[i]
                h = ij2h(i,j,NvecT)
                tUh = tUT[h]
                l_cur += logpdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zT[i,:])/xi_i_cur, eta[tUh]), gapT[i][j])
                l_pro += logpdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zT[i,:])/xi_i_pro, eta[tUh]), gapT[i][j])
            end
        end

        h = NT + i 
        tUh = tUT[h]
        if NvecT[i] == 0 # N
            l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zT[i,:])/xi_i_cur, eta[tUh]), survivalT[i])
            l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zT[i,:])/xi_i_pro, eta[tUh]), survivalT[i])
        else
            l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zT[i,:])/xi_i_cur, eta[tUh]), survivalT[i]-sum(gapT[i]))
            l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[tUh,:]'*zT[i,:])/xi_i_pro, eta[tUh]), survivalT[i]-sum(gapT[i]))
        end 

        if log(rand(Uniform(0,1),1)[1]) < (l_pro - l_cur)
			epsilonT_new[i] = epsilon_i_pro 
            xiT_new[i] = xi_i_pro
        else
			epsilonT_new[i] = epsilon_i_cur 
            xiT_new[i] = xi_i_cur
		end 
    end

    return Dict("epsilonC" => epsilonC_new, "epsilonT" => epsilonT_new, "xiC" => xiC_new, "xiT" => xiT_new)
end



function update_Sigma_e(datC, datT, cur, hyper)
    epsilonC = cur["epsilonC"]
    epsilonT = cur["epsilonT"]

    xiC = cur["xiC"]
    xiT = cur["xiT"]

    c_e = hyper["c_e"]
    C_e = hyper["C_e"]

    nC = datC["n"]
    nT = datT["n"]
    
    c_e_1_new = c_e + nC
    C_e_1_new = C_e 

    c_e_2_new = c_e + nT
    C_e_2_new = C_e 

    for i in 1:nC
        tmp = [log(epsilonC[i]), log(xiC[i])]
        C_e_1_new += tmp * tmp'
    end

    for i in 1:nT
        tmp = [log(epsilonT[i]), log(xiT[i])]
        C_e_2_new += tmp * tmp' 
    end 

    new_1 = rand(InverseWishart(c_e_1_new, C_e_1_new), 1)[1]
    new_2 = rand(InverseWishart(c_e_2_new, C_e_2_new), 1)[1]
    return [new_1, new_2]
end 

# function update_random_effects(datC, datT, cur, hyper, Sig)
#     SigC = Sig[1]
#     SigT = Sig[2]

#     epsilonC = cur["epsilonC"]
#     epsilonT = cur["epsilonT"]

#     xiC = cur["xiC"]
#     xiT = cur["xiT"]

#     LC = cur["LC"]
#     LT = cur["LT"]

#     tUC = cur["tUC"]
#     tUT = cur["tUT"]

#     theta = cur["theta"]
#     beta = cur["beta"]
#     phi = cur["phi"]
#     lambda = cur["lambda"]
#     gamma = cur["gamma"]
#     eta = cur["eta"]

#     survivalC = datC["survival"]
#     gapC = datC["gap"]
#     NvecC = datC["Nvec"]
#     NC = datC["N"]
#     nC = datC["n"]
#     nuC = datC["nu"]
#     xC = datC["x"]
#     zC = datC["z"]

#     survivalT = datT["survival"]
#     gapT = datT["gap"]
#     NvecT = datT["Nvec"]
#     NT = datT["N"]
#     nT = datT["n"]
#     nuT = datT["nu"]
#     xT = datT["x"]
#     zT = datT["z"]

#     Sigma_e_1 = cur["Sigma_e_1"]
#     Sigma_e_2 = cur["Sigma_e_2"]

#     epsilonC_new = zeros(nC)
#     xiC_new = zeros(nC)
#     epsilonT_new = zeros(nT)
#     xiT_new = zeros(nT)

#     for i in 1:nC
        
#         tmp = rand(Uniform(0,1), 1)[1]
#         if tmp < 0.95 
#             walk = rand(MvNormal(zeros(2), 2.38^2/2*SigC[i]), 1)[:,1]
#         else 
#             walk = rand(MvNormal(zeros(2), 0.01/2*Diagonal(ones(2))), 1)[:,1]
#         end 

#         epsilon_i_cur = epsilonC[i]
#         epsilon_i_pro = exp(log(epsilon_i_cur) + walk[1])

#         xi_i_cur = xiC[i]
#         xi_i_pro = exp(log(xi_i_cur) + walk[2])

#         l_cur = logpdf(MvLogNormal(zeros(2), Sigma_e_1), [epsilon_i_cur, xi_i_cur]) + log(epsilon_i_cur) + log(xi_i_cur)
#         l_pro = logpdf(MvLogNormal(zeros(2), Sigma_e_1), [epsilon_i_pro, xi_i_pro]) + log(epsilon_i_pro) + log(xi_i_pro)
        
#         Li = LC[i]
#         if nuC[i] == 1
#             l_cur += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xC[i,:])/epsilon_i_cur, phi[Li]), survivalC[i])
#             l_pro += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xC[i,:])/epsilon_i_pro, phi[Li]), survivalC[i])
#         else 
#             l_cur += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xC[i,:])/epsilon_i_cur, phi[Li]), survivalC[i])
#             l_pro += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xC[i,:])/epsilon_i_pro, phi[Li]), survivalC[i])
#         end

#         if NvecC[i] > 0
#             for j in 1:NvecC[i]
#                 h = ij2h(i,j,NvecC)
#                 tUh = tUC[h]
#                 l_cur += logpdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zC[i,:])/xi_i_cur, eta[tUh]), gapC[i][j])
#                 l_pro += logpdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zC[i,:])/xi_i_pro, eta[tUh]), gapC[i][j])
#             end
#         end

#         h = NC + i 
#         tUh = tUC[h]
#         if NvecC[i] == 0 # N
#             l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zC[i,:])/xi_i_cur, eta[tUh]), survivalC[i])
#             l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zC[i,:])/xi_i_pro, eta[tUh]), survivalC[i])
#         else
#             l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zC[i,:])/xi_i_cur, eta[tUh]), survivalC[i]-sum(gapC[i]))
#             l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zC[i,:])/xi_i_pro, eta[tUh]), survivalC[i]-sum(gapC[i]))
#         end 

#         if log(rand(Uniform(0,1),1)[1]) < (l_pro - l_cur)
# 			epsilonC_new[i] = epsilon_i_pro 
#             xiC_new[i] = xi_i_pro
#         else
# 			epsilonC_new[i] = epsilon_i_cur 
#             xiC_new[i] = xi_i_cur
# 		end 
#     end

#     for i in 1:nT
        
#         tmp = rand(Uniform(0,1), 1)[1]
#         if tmp < 0.95 
#             walk = rand(MvNormal(zeros(2), 2.38^2/2*SigT[i]), 1)[:,1]
#         else 
#             walk = rand(MvNormal(zeros(2), 0.01/2*Diagonal(ones(2))), 1)[:,1]
#         end 

#         epsilon_i_cur = epsilonT[i]
#         epsilon_i_pro = exp(log(epsilon_i_cur) + walk[1])

#         xi_i_cur = xiT[i]
#         xi_i_pro = exp(log(xi_i_cur) + walk[2])

#         l_cur = logpdf(MvLogNormal(zeros(2), Sigma_e_2), [epsilon_i_cur, xi_i_cur]) + log(epsilon_i_cur) + log(xi_i_cur)
#         l_pro = logpdf(MvLogNormal(zeros(2), Sigma_e_2), [epsilon_i_pro, xi_i_pro]) + log(epsilon_i_pro) + log(xi_i_pro)
        
#         Li = LT[i]
#         if nuT[i] == 1
#             l_cur += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xT[i,:])/epsilon_i_cur, phi[Li]), survivalT[i])
#             l_pro += logpdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xT[i,:])/epsilon_i_pro, phi[Li]), survivalT[i])
#         else 
#             l_cur += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xT[i,:])/epsilon_i_cur, phi[Li]), survivalT[i])
#             l_pro += logccdf(LogLogistic(theta[Li]*exp(beta[Li,:]'*xT[i,:])/epsilon_i_pro, phi[Li]), survivalT[i])
#         end

#         if NvecT[i] > 0
#             for j in 1:NvecT[i]
#                 h = ij2h(i,j,NvecT)
#                 tUh = tUT[h]
#                 l_cur += logpdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zT[i,:])/xi_i_cur, eta[tUh]), gapT[i][j])
#                 l_pro += logpdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zT[i,:])/xi_i_pro, eta[tUh]), gapT[i][j])
#             end
#         end

#         h = NT + i 
#         tUh = tUT[h]
#         if NvecT[i] == 0 # N
#             l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zT[i,:])/xi_i_cur, eta[tUh]), survivalT[i])
#             l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zT[i,:])/xi_i_pro, eta[tUh]), survivalT[i])
#         else
#             l_cur += logccdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zT[i,:])/xi_i_cur, eta[tUh]), survivalT[i]-sum(gapT[i]))
#             l_pro += logccdf(LogLogistic(lambda[tUh]*exp(gamma[Li,:]'*zT[i,:])/xi_i_pro, eta[tUh]), survivalT[i]-sum(gapT[i]))
#         end 

#         if log(rand(Uniform(0,1),1)[1]) < (l_pro - l_cur)
# 			epsilonT_new[i] = epsilon_i_pro 
#             xiT_new[i] = xi_i_pro
#         else
# 			epsilonT_new[i] = epsilon_i_cur 
#             xiT_new[i] = xi_i_cur
# 		end 
#     end

#     return Dict("epsilonC" => epsilonC_new, "epsilonT" => epsilonT_new, "xiC" => xiC_new, "xiT" => xiT_new)
# end



# function update_Sigma_e(datC, datT, cur, hyper)
#     epsilonC = cur["epsilonC"]
#     epsilonT = cur["epsilonT"]

#     xiC = cur["xiC"]
#     xiT = cur["xiT"]

#     c_e = hyper["c_e"]
#     C_e = hyper["C_e"]

#     nC = datC["n"]
#     nT = datT["n"]
    
#     c_e_1_new = c_e + nC
#     C_e_1_new = C_e 

#     c_e_2_new = c_e + nT
#     C_e_2_new = C_e 

#     for i in 1:nC
#         tmp = [log(epsilonC[i]), log(xiC[i])]
#         C_e_1_new += tmp * tmp'
#     end

#     for i in 1:nT
#         tmp = [log(epsilonT[i]), log(xiT[i])]
#         C_e_2_new += tmp * tmp' 
#     end 

#     new_1 = rand(InverseWishart(c_e_1_new, C_e_1_new), 1)[1]
#     new_2 = rand(InverseWishart(c_e_2_new, C_e_2_new), 1)[1]
#     return [new_1, new_2]
# end 