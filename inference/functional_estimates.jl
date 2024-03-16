function survival_functional_estimation(
    sigma2_e_save,
    phi_save, 
    theta_save, 
    alpha_save,
    nl_save, 
    n_clusters, 
    a_phi,
    b_phi_save, 
    mu_theta_save, 
    sigma2_theta, 
    grids 
    )

    nkeep = length(alpha_save)
    ngrids = length(grids) 

    dens = zeros(nkeep, ngrids)
    surv = zeros(nkeep, ngrids)

    @showprogress for i in 1:nkeep

        k = n_clusters[i]
        nl = nl_save[i]

        ind = findall(nl .> 0)
        phi_star = phi_save[i][ind]
        theta_star = theta_save[i][ind]

        alpha = alpha_save[i]

        q = rand(Dirichlet(vcat(nl[ind], alpha)), 1)[:,1]

        NN = 20
        V = rand(Beta(1, alpha), NN-1)
        logp = zeros(NN) 
        logp[1] = log(V[1])
        for r in 2:(NN-1)
            logp[r] = log(V[r]) + sum(log.(1 .- V[1:(r-1)]))
        end
        logp[NN] = sum(log.(1 .- V[1:(NN-1)]))
        p = exp.(logp)

        theta_tilde = zeros(NN)
        phi_tilde = zeros(NN)
        for l in 1:NN
            G0_phi = Gamma(a_phi, b_phi_save[i])
            G0_theta = Normal(mu_theta_save[i], sqrt(sigma2_theta))
            phi_tilde[l] = sqrt(rand(G0_phi, 1)[1])
            theta_tilde[l] = exp(rand(G0_theta, 1)[1])
        end 

        nrep = 100
        epsilon_rep = rand(LogNormal(0, sqrt(sigma2_e_save[i])), nrep)
        for g in eachindex(grids) 
            for i_rep in 1:nrep 
                for l in 1:NN
                    dens[i,g] += 1/nrep * q[end] * p[l] * pdf(LogLogistic(theta_tilde[l]/epsilon_rep[i_rep], phi_tilde[l]), grids[g])
                    surv[i,g] += 1/nrep * q[end] * p[l] * ccdf(LogLogistic(theta_tilde[l]/epsilon_rep[i_rep], phi_tilde[l]), grids[g])
                end
                # for l in 1:k
                for l in eachindex(ind)
                    dens[i,g] += 1/nrep * q[l] * pdf(LogLogistic(theta_star[l]/epsilon_rep[i_rep], phi_star[l]), grids[g])
                    surv[i,g] += 1/nrep * q[l] * ccdf(LogLogistic(theta_star[l]/epsilon_rep[i_rep], phi_star[l]), grids[g])
                end
            end
        end
    end 

    return Dict("d" => dens, "s" => surv)

end

function survival_functional_prior(
    c_e, 
    C_e,
    a_alpha, 
    b_alpha, 
    a_phi, 
    r_phi, 
    R_phi, 
    s_theta, 
    S_theta, 
    sigma2_theta, 
    grids
    ) 

    nMC = 1000
    alpha = rand(Gamma(a_alpha, b_alpha), nMC) 
    b_phi = rand(InverseGamma(r_phi, R_phi), nMC)
    mu_theta = rand(Normal(s_theta, sqrt(S_theta)), nMC)
    Sigma_e = rand(InverseWishart(c_e, C_e), nMC)

    ngrids = length(grids) 
    d = zeros(nMC, ngrids)
    s = zeros(nMC, ngrids)
    N = 40 

    @showprogress for i in 1:nMC 
        V = rand(Beta(1, alpha[i]), N-1)
        logp = zeros(N) 
        logp[1] = log(V[1])
        for r in 2:(N-1)
            logp[r] = log(V[r]) + sum(log.(1 .- V[1:(r-1)]))
        end
        logp[N] = sum(log.(1 .- V[1:(N-1)]))
        p = exp.(logp)

        phi = sqrt.(rand(Gamma(a_phi, b_phi[i]), N))
        theta = exp.(rand(Normal(mu_theta[i], sqrt(sigma2_theta)), N))

        nrep = 100
        epsilon = rand(LogNormal(0, Sigma_e[i][1,1]), nrep)
        for g in eachindex(grids)  
            tmp = 0.0
            tmps = 0.0 
            for l in 1:N 
                for i_r in 1:nrep
                    tmp += 1/nrep * p[l] * pdf(LogLogistic(theta[l]/epsilon[i_r], phi[l]), grids[g]) 
                    tmps += 1/nrep * p[l] * ccdf(LogLogistic(theta[l]/epsilon[i_r], phi[l]), grids[g]) 
                end
            end
            d[i, g] = tmp  
            s[i, g] = tmps 
        end
    end

    return Dict("d" => d, "s" => s) 
end 


function conditional_survival_probability(
    Sigma_e_save, 
    a_eta,
    eta_save, 
    lambda_save, 
    zeta_save, 
    ml_save, 
    b_eta_save, 
    mu_lambda_save, 
    sigma2_lambda, 
    phi_save, 
    theta_save, 
    alpha_save,
    nl_save, 
    a_phi,
    b_phi_save, 
    mu_theta_save, 
    sigma2_theta, 
    grids    
    )
    # Pr(T \ge t | T > t0, N(t_0) \ge 1, Θ)
    
    nrep = 100
    nkeep = length(alpha_save)
    ngrids = length(grids) 
    
    surv_surv = zeros(nkeep, ngrids, nrep)
    gap_surv0 = zeros(nkeep, nrep)

    @showprogress for i in 1:nkeep 

        Sigma_e_star = Sigma_e_save[i,:,:]
        re = rand(MvLogNormal(zeros(2), Sigma_e_star), nrep)
        epsilon_reps = re[1,:]
        xi_reps = re[2,:]

        ml = ml_save[i]

        ind_gap = findall(ml .> 0)
        lambda_star = lambda_save[i][ind_gap]
        eta_star = eta_save[i][ind_gap]

        zeta_star = zeta_save[i]

        q_gap = rand(Dirichlet(vcat(ml[ind_gap], zeta_star)), 1)[:,1]

        N_gap = 20
        V_gap = rand(Beta(1, zeta_star), N_gap-1)
        logp_gap = zeros(N_gap) 
        logp_gap[1] = log(V_gap[1])
        for r in 2:(N_gap-1)
            logp_gap[r] = log(V_gap[r]) + sum(log.(1 .- V_gap[1:(r-1)]))
        end
        logp_gap[N_gap] = sum(log.(1 .- V_gap[1:(N_gap-1)]))
        p_gap = exp.(logp_gap)

        lambda_tilde = zeros(N_gap)
        eta_tilde = zeros(N_gap)
        for l in 1:N_gap
            G0_eta = Gamma(a_eta, b_eta_save[i])
            G0_lambda = Normal(mu_lambda_save[i], sqrt(sigma2_lambda))
            eta_tilde[l] = sqrt(rand(G0_eta, 1)[1])
            lambda_tilde[l] = exp(rand(G0_lambda, 1)[1])
        end 


        nl = nl_save[i]

        ind_surv = findall(nl .> 0)
        theta_star = theta_save[i][ind_surv]
        phi_star = phi_save[i][ind_surv]

        alpha_star = alpha_save[i]

        q_surv = rand(Dirichlet(vcat(nl[ind_surv], alpha_star)), 1)[:,1]

        N_surv = 20
        V_surv = rand(Beta(1, alpha_star), N_surv-1)
        logp_surv = zeros(N_surv) 
        logp_surv[1] = log(V_surv[1])
        for r in 2:(N_surv-1)
            logp_surv[r] = log(V_surv[r]) + sum(log.(1 .- V_surv[1:(r-1)]))
        end
        logp_surv[N_surv] = sum(log.(1 .- V_surv[1:(N_surv-1)]))
        p_surv = exp.(logp_surv)

        theta_tilde = zeros(N_surv)
        phi_tilde = zeros(N_surv)
        for l in 1:N_surv
            G0_phi = Gamma(a_phi, b_phi_save[i])
            G0_theta = Normal(mu_theta_save[i], sqrt(sigma2_theta))
            phi_tilde[l] = sqrt(rand(G0_phi, 1)[1])
            theta_tilde[l] = exp(rand(G0_theta, 1)[1])
        end 

        for i_rep in 1:nrep 
            for l in 1:N_gap
                gap_surv0[i,i_rep] += q_gap[end] * p_gap[l] * ccdf(LogLogistic(lambda_tilde[l]/xi_reps[i_rep], eta_tilde[l]), grids[1])
            end
            for l in eachindex(ind_gap)
                gap_surv0[i,i_rep] += q_gap[l] * ccdf(LogLogistic(lambda_star[l]/xi_reps[i_rep], eta_star[l]), grids[1])
            end
        end

        for g in eachindex(grids) 
            for i_rep in 1:nrep
                for l in 1:N_surv 
                    surv_surv[i,g,i_rep] += q_surv[end] * p_surv[l] * ccdf(LogLogistic(theta_tilde[l]/epsilon_reps[i_rep], phi_tilde[l]), grids[g])
                end
                for l in eachindex(ind_surv)
                    surv_surv[i,g,i_rep] += q_surv[l] * ccdf(LogLogistic(theta_star[l]/epsilon_reps[i_rep], phi_star[l]), grids[g])
                end 
            end # for i_rep 
        end # for g 
    end # for i 

    cond_surv_N0 = zeros(nkeep, ngrids)
    for i in 1:nkeep 
        lower_N0 = 0.0
        upper_N0 = zeros(ngrids)
        for i_rep in 1:nrep 
            lower_N0 += 1/nrep * surv_surv[i,1,i_rep] * gap_surv0[i,i_rep]
            upper_N0[1] = lower_N0
            for g in 2:ngrids 
                upper_N0[g] += 1/nrep * surv_surv[i,g,i_rep] * gap_surv0[i,i_rep]
            end
        end 
        for g in 1:ngrids
            cond_surv_N0[i,g] = upper_N0[g] / lower_N0
        end
    end 
    return cond_surv_N0
end # funciton 

function predict(Sigma_e_matrix, theta_list, phi_list, nl, lambda_list, eta_list, ml)
    nsim = length(nl)

    survival = zeros(nsim)
    gap = []
    arrival = []

    theta = zeros(nsim) 
    phi = zeros(nsim)
    lambda = zeros(nsim)
    eta = zeros(nsim)
    
    @showprogress for i in 1:nsim 
        re = rand(MvLogNormal(zeros(2), Sigma_e_matrix[i,:,:]), 1)
        epsilon = re[1]
        xi = re[2]

        surv_index = sample([1:1:length(nl[i]);], StatsBase.fweights(nl[i]), 1)[1]
        surv_dist = LogLogistic(theta_list[i][surv_index]/epsilon, phi_list[i][surv_index])
        survival[i] = rand(surv_dist, 1)[1]

        theta[i] = theta_list[i][surv_index]
        phi[i] = phi_list[i][surv_index]

        gap_index = sample([1:1:length(ml[i]);], StatsBase.fweights(ml[i]), 1)[1]
        gap_dist = LogLogistic(lambda_list[i][gap_index]/xi, eta_list[i][gap_index])

        lambda[i] = lambda_list[i][gap_index]
        eta[i] = eta_list[i][gap_index]

        tmp_arrival = zeros(1)
        tmp_gap = zeros(0)
        while true 
            gap_tmp = rand(gap_dist, 1)[1]
            if tmp_arrival[end] + gap_tmp > survival[i]
                break 
            end 

            push!(tmp_gap, gap_tmp)
            push!(tmp_arrival, tmp_arrival[end] + gap_tmp)
        end

        push!(gap, tmp_gap)
        push!(arrival, tmp_arrival[2:end])
    end 

    return [survival, gap, arrival, theta, phi, lambda, eta]
end


function survival_functional_estimation_regression(
    Sigma_e_save,
    phi_save, 
    theta_save, 
    beta_save,
    alpha_save,
    nl_save, 
    a_phi,
    b_phi_save, 
    mu_theta_save, 
    sigma2_theta, 
    mu_beta_save,
    Sigma_beta,
    grids, 
    x0
    )

    pdim = size(mu_beta_save)[2]
    nkeep = length(alpha_save)
    ngrids = length(grids) 

    dens = zeros(nkeep, ngrids)
    surv = zeros(nkeep, ngrids)

    @showprogress for i in 1:nkeep

        phi_star = phi_save[i]
        theta_star = theta_save[i] 
        beta_star = beta_save[i]

        alpha = alpha_save[i]

        nl = nl_save[i]
        ind = findall(nl .> 0)
        q = rand(Dirichlet(vcat(nl[ind], alpha)), 1)[:,1]

        NN = 20
        V = rand(Beta(1, alpha), NN-1)
        logp = zeros(NN) 
        logp[1] = log(V[1])
        for r in 2:(NN-1)
            logp[r] = log(V[r]) + sum(log.(1 .- V[1:(r-1)]))
        end
        logp[NN] = sum(log.(1 .- V[1:(NN-1)]))
        p = exp.(logp)

        theta_tilde = zeros(NN)
        beta_tilde = zeros(NN, pdim)
        phi_tilde = zeros(NN)
        for l in 1:NN
            G0_phi = Gamma(a_phi, b_phi_save[i])
            G0_theta = Normal(mu_theta_save[i], sqrt(sigma2_theta))
            G0_beta = MvNormal(mu_beta_save[i,:], Sigma_beta)
            phi_tilde[l] = sqrt(rand(G0_phi, 1)[1])
            theta_tilde[l] = exp(rand(G0_theta, 1)[1])
            beta_tilde[l,:] = rand(G0_beta, 1)
        end 

        nrep = 100
        epsilon_rep = rand(LogNormal(0, sqrt(Sigma_e_save[i,1,1])), nrep)
        for g in eachindex(grids) 
            for i_rep in 1:nrep 
                for l in 1:NN
                    surv_model = LogLogistic(theta_tilde[l]*exp(beta_tilde[l,:]'*x0)/epsilon_rep[i_rep], phi_tilde[l])
                    dens[i,g] += 1/nrep * q[end] * p[l] * pdf(surv_model, grids[g])
                    surv[i,g] += 1/nrep * q[end] * p[l] * ccdf(surv_model, grids[g])
                end
                # for l in 1:k
                for l in eachindex(ind)
                    surv_model = LogLogistic(theta_star[l]*exp(beta_star[l,:]'*x0)/epsilon_rep[i_rep], phi_star[l])
                    dens[i,g] += 1/nrep * q[l] * pdf(surv_model, grids[g])
                    surv[i,g] += 1/nrep * q[l] * ccdf(surv_model, grids[g])
                end
            end
        end
    end 

    return Dict("d" => dens, "s" => surv)

end

function conditional_survival_probability_regression(
    Sigma_e_save, 
    a_eta,
    eta_save, 
    lambda_save, 
    gamma_save, 
    zeta_save, 
    ml_save, 
    b_eta_save, 
    mu_lambda_save, 
    mu_gamma_save, 
    sigma2_lambda, 
    Sigma_gamma,
    phi_save, 
    theta_save, 
    beta_save,
    alpha_save,
    nl_save, 
    a_phi,
    b_phi_save, 
    mu_theta_save, 
    mu_beta_save,
    sigma2_theta, 
    Sigma_beta,
    grids,    
    x0,
    z0
    )
    # Pr(T \ge t | T > t0, N(t_0) \ge 1, Θ)
    
    nrep = 200
    nkeep = length(alpha_save)
    ngrids = length(grids) 

    p = length(x0)
    q = length(z0)
    
    surv_surv = zeros(nkeep, ngrids, nrep)
    gap_surv0 = zeros(nkeep, nrep)

    @showprogress for i in 1:nkeep 

        Sigma_e_star = Sigma_e_save[i,:,:]
        re = rand(MvLogNormal(zeros(2), Sigma_e_star), nrep)
        epsilon_reps = re[1,:]
        xi_reps = re[2,:]

        lambda_star = lambda_save[i]
        gamma_star = gamma_save[i]
        eta_star = eta_save[i]
        zeta_star = zeta_save[i]

        ml = ml_save[i]
        ind_gap = findall(ml .> 0)
        q_gap = rand(Dirichlet(vcat(ml[ind_gap], zeta_star)), 1)[:,1]

        N_gap = 20
        V_gap = rand(Beta(1, zeta_star), N_gap-1)
        logp_gap = zeros(N_gap) 
        logp_gap[1] = log(V_gap[1])
        for r in 2:(N_gap-1)
            logp_gap[r] = log(V_gap[r]) + sum(log.(1 .- V_gap[1:(r-1)]))
        end
        logp_gap[N_gap] = sum(log.(1 .- V_gap[1:(N_gap-1)]))
        p_gap = exp.(logp_gap)

        lambda_tilde = zeros(N_gap)
        gamma_tilde = zeros(N_gap, q)
        eta_tilde = zeros(N_gap)
        for l in 1:N_gap
            G0_eta = Gamma(a_eta, b_eta_save[i])
            G0_lambda = Normal(mu_lambda_save[i], sqrt(sigma2_lambda))
            G0_gamma = MvNormal(mu_gamma_save[i,:], Sigma_gamma)
            eta_tilde[l] = sqrt(rand(G0_eta, 1)[1])
            lambda_tilde[l] = exp(rand(G0_lambda, 1)[1])
            gamma_tilde[l,:] = rand(G0_gamma, 1)
        end 


        theta_star = theta_save[i]
        beta_star = beta_save[i]
        phi_star = phi_save[i]
        alpha_star = alpha_save[i]

        nl = nl_save[i]
        ind_surv = findall(nl .> 0)
        q_surv = rand(Dirichlet(vcat(nl[ind_surv], alpha_star)), 1)[:,1]

        N_surv = 20
        V_surv = rand(Beta(1, alpha_star), N_surv-1)
        logp_surv = zeros(N_surv) 
        logp_surv[1] = log(V_surv[1])
        for r in 2:(N_surv-1)
            logp_surv[r] = log(V_surv[r]) + sum(log.(1 .- V_surv[1:(r-1)]))
        end
        logp_surv[N_surv] = sum(log.(1 .- V_surv[1:(N_surv-1)]))
        p_surv = exp.(logp_surv)

        theta_tilde = zeros(N_surv)
        beta_tilde = zeros(N_surv, p)
        phi_tilde = zeros(N_surv)
        for l in 1:N_surv
            G0_phi = Gamma(a_phi, b_phi_save[i])
            G0_theta = Normal(mu_theta_save[i], sqrt(sigma2_theta))
            G0_beta = MvNormal(mu_beta_save[i,:], Sigma_beta)
            phi_tilde[l] = sqrt(rand(G0_phi, 1)[1])
            theta_tilde[l] = exp(rand(G0_theta, 1)[1])
            beta_tilde[l,:] = rand(G0_beta, 1)
        end 

        for i_rep in 1:nrep 
            for l in 1:N_gap
                scale_param = lambda_tilde[l] * exp(gamma_tilde[l,:]' * z0) /xi_reps[i_rep]
                gap_surv0[i,i_rep] += 1/nrep * q_gap[end] * p_gap[l] * ccdf(LogLogistic(scale_param, eta_tilde[l]), grids[1])
            end
            # for l in 1:g
            for l in eachindex(ind_gap)
                scale_param = lambda_star[l] * exp(gamma_star[l,:]' * z0) /xi_reps[i_rep]
                gap_surv0[i,i_rep] += 1/nrep * q_gap[l] * ccdf(LogLogistic(scale_param, eta_star[l]), grids[1])
            end
        end
        for g in eachindex(grids) 
            for i_rep in 1:nrep
                for l in 1:N_surv 
                    surv_surv[i,g,i_rep] += 1/nrep * q_surv[end] * p_surv[l] * ccdf(LogLogistic(theta_tilde[l] * exp(beta_tilde[l,:]'*x0) /epsilon_reps[i_rep], phi_tilde[l]), grids[g])
                end
                # for l in 1:k
                for l in eachindex(ind_surv)
                    surv_surv[i,g,i_rep] += 1/nrep * q_surv[l] * ccdf(LogLogistic(theta_star[l] * exp(beta_star[l,:]'*x0) /epsilon_reps[i_rep], phi_star[l]), grids[g])
                end 
            end # for i_rep 
        end # for g 
    end # for i 

    cond_surv_N0 = zeros(nkeep, ngrids)
    for i in 1:nkeep 
        lower_N0 = 0.0
        upper_N0 = zeros(ngrids)
        for i_rep in 1:nrep 
            lower_N0 += 1/nrep * surv_surv[i,1,i_rep] * gap_surv0[i,i_rep]
            upper_N0[1] = lower_N0
            for g in 2:ngrids 
                upper_N0[g] += 1/nrep * surv_surv[i,g,i_rep] * gap_surv0[i,i_rep]
            end
        end 
        for g in 1:ngrids
            cond_surv_N0[i,g] = upper_N0[g] / lower_N0
        end
    end 
    return cond_surv_N0
end # funciton 


function predict_cov1(Sigma_e_matrix, theta_list, beta_list, phi_list, nl, lambda_list, gamma_list, eta_list, ml)
    nsim = length(nl)

    survivalC = zeros(nsim)
    gapC = []
    arrivalC = []

    survivalT = zeros(nsim)
    gapT = []
    arrivalT = []
    
    theta = zeros(nsim)
    beta = zeros(nsim)
    phi = zeros(nsim) 

    lambda = zeros(nsim)
    gamma = zeros(nsim) 
    eta = zeros(nsim)
    
    @showprogress for i in 1:nsim 
        re = rand(MvLogNormal(zeros(2), Sigma_e_matrix[i,:,:]), 1)
        epsilon = re[1]
        xi = re[2]

        surv_index = sample([1:1:length(nl[i]);], StatsBase.fweights(nl[i]), 1)[1]
        theta[i] = theta_list[i][surv_index]
        beta[i] = beta_list[i][surv_index,1]
        phi[i] = phi_list[i][surv_index]
        surv_distC = LogLogistic(theta_list[i][surv_index] /epsilon, phi_list[i][surv_index])
        surv_distT = LogLogistic(theta_list[i][surv_index]*exp(beta_list[i][surv_index,1])/epsilon, phi_list[i][surv_index])
        survivalC[i] = rand(surv_distC, 1)[1]
        survivalT[i] = rand(surv_distT, 1)[1]

        gap_index = sample([1:1:length(ml[i]);], StatsBase.fweights(ml[i]), 1)[1]
        lambda[i] = lambda_list[i][gap_index]
        gamma[i] = gamma_list[i][gap_index,1]
        eta[i] = eta_list[i][gap_index]
        gap_distC = LogLogistic(lambda_list[i][gap_index]/xi, eta_list[i][gap_index])
        gap_distT = LogLogistic(lambda_list[i][gap_index]*exp(gamma_list[i][gap_index,1])/xi, eta_list[i][gap_index])

        tmp_arrivalC = zeros(1)
        tmp_gapC = zeros(0)
        while true 
            gap_tmpC = rand(gap_distC, 1)[1]
            if tmp_arrivalC[end] + gap_tmpC > survivalC[i]
                break 
            end 

            push!(tmp_gapC, gap_tmpC)
            push!(tmp_arrivalC, tmp_arrivalC[end] + gap_tmpC)
        end

        push!(gapC, tmp_gapC)
        push!(arrivalC, tmp_arrivalC[2:end])

        tmp_arrivalT = zeros(1)
        tmp_gapT = zeros(0)
        while true 
            gap_tmpT = rand(gap_distT, 1)[1]
            if tmp_arrivalT[end] + gap_tmpT > survivalT[i]
                break 
            end 

            push!(tmp_gapT, gap_tmpT)
            push!(tmp_arrivalT, tmp_arrivalT[end] + gap_tmpT)
        end

        push!(gapT, tmp_gapT)
        push!(arrivalT, tmp_arrivalT[2:end])
    end 

    return [survivalC, gapC, arrivalC, theta, beta, phi, 
            survivalT, gapT, arrivalT, lambda, gamma, eta]
end


function predict_cov1_re(Sigma_e_1_matrix, Sigma_e_2_matrix, theta_list, beta_list, phi_list, nl, lambda_list, gamma_list, eta_list, ml)
    nsim = length(nl)

    survivalC = zeros(nsim)
    gapC = []
    arrivalC = []

    survivalT = zeros(nsim)
    gapT = []
    arrivalT = []
    
    theta = zeros(nsim)
    beta = zeros(nsim)
    phi = zeros(nsim) 

    lambda = zeros(nsim)
    gamma = zeros(nsim) 
    eta = zeros(nsim)
    
    @showprogress for i in 1:nsim 
        re_1 = rand(MvLogNormal(zeros(2), Sigma_e_1_matrix[i,:,:]), 1)
        epsilon_1 = re_1[1]
        xi_1 = re_1[2]

        re_2 = rand(MvLogNormal(zeros(2), Sigma_e_2_matrix[i,:,:]), 1)
        epsilon_2 = re_2[1]
        xi_2 = re_2[2]

        surv_index = sample([1:1:length(nl[i]);], StatsBase.fweights(nl[i]), 1)[1]
        theta[i] = theta_list[i][surv_index]
        beta[i] = beta_list[i][surv_index,1]
        phi[i] = phi_list[i][surv_index]
        surv_distC = LogLogistic(theta_list[i][surv_index] / epsilon_1, phi_list[i][surv_index])
        surv_distT = LogLogistic(theta_list[i][surv_index]*exp(beta_list[i][surv_index,1])/epsilon_2, phi_list[i][surv_index])
        survivalC[i] = rand(surv_distC, 1)[1]
        survivalT[i] = rand(surv_distT, 1)[1]

        gap_index = sample([1:1:length(ml[i]);], StatsBase.fweights(ml[i]), 1)[1]
        lambda[i] = lambda_list[i][gap_index]
        gamma[i] = gamma_list[i][gap_index,1]
        eta[i] = eta_list[i][gap_index]
        gap_distC = LogLogistic(lambda_list[i][gap_index]/xi_1, eta_list[i][gap_index])
        gap_distT = LogLogistic(lambda_list[i][gap_index]*exp(gamma_list[i][gap_index,1])/xi_2, eta_list[i][gap_index])

        tmp_arrivalC = zeros(1)
        tmp_gapC = zeros(0)
        while true 
            gap_tmpC = rand(gap_distC, 1)[1]
            if tmp_arrivalC[end] + gap_tmpC > survivalC[i]
                break 
            end 

            push!(tmp_gapC, gap_tmpC)
            push!(tmp_arrivalC, tmp_arrivalC[end] + gap_tmpC)
        end

        push!(gapC, tmp_gapC)
        push!(arrivalC, tmp_arrivalC[2:end])

        tmp_arrivalT = zeros(1)
        tmp_gapT = zeros(0)
        while true 
            gap_tmpT = rand(gap_distT, 1)[1]
            if tmp_arrivalT[end] + gap_tmpT > survivalT[i]
                break 
            end 

            push!(tmp_gapT, gap_tmpT)
            push!(tmp_arrivalT, tmp_arrivalT[end] + gap_tmpT)
        end

        push!(gapT, tmp_gapT)
        push!(arrivalT, tmp_arrivalT[2:end])
    end 

    return [survivalC, gapC, arrivalC, theta, beta, phi, 
            survivalT, gapT, arrivalT, lambda, gamma, eta]
end


function predict_cov_re(Sigma_e_1_matrix, Sigma_e_2_matrix, theta_list, beta_list, phi_list, nlC, nlT, lambda_list, gamma_list, eta_list, mlC, mlT)
    nsim = length(nlC)

    survivalC = zeros(nsim)
    gapC = []
    arrivalC = []

    survivalT = zeros(nsim)
    gapT = []
    arrivalT = []
    
    thetaC = zeros(nsim)
    betaC = zeros(nsim)
    phiC = zeros(nsim) 

    lambdaC = zeros(nsim)
    gammaC = zeros(nsim) 
    etaC = zeros(nsim)
    
    thetaT = zeros(nsim)
    betaT = zeros(nsim)
    phiT = zeros(nsim) 

    lambdaT = zeros(nsim)
    gammaT = zeros(nsim) 
    etaT = zeros(nsim)
    
    @showprogress for i in 1:nsim 
        re_1 = rand(MvLogNormal(zeros(2), Sigma_e_1_matrix[i,:,:]), 1)
        epsilon_1 = re_1[1]
        xi_1 = re_1[2]

        re_2 = rand(MvLogNormal(zeros(2), Sigma_e_2_matrix[i,:,:]), 1)
        epsilon_2 = re_2[1]
        xi_2 = re_2[2]

        surv_indexC = sample([1:1:length(nlC[i]);], StatsBase.fweights(nlC[i]), 1)[1]
        thetaC[i] = theta_list[i][surv_indexC]
        betaC[i] = beta_list[i][surv_indexC,1]
        phiC[i] = phi_list[i][surv_indexC]
        surv_distC = LogLogistic(thetaC[i] / epsilon_1, phiC[i])
        survivalC[i] = rand(surv_distC, 1)[1]

        surv_indexT = sample([1:1:length(nlT[i]);], StatsBase.fweights(nlT[i]), 1)[1]
        thetaT[i] = theta_list[i][surv_indexT]
        betaT[i] = beta_list[i][surv_indexT,1]
        phiT[i] = phi_list[i][surv_indexT]
        surv_distT = LogLogistic(thetaT[i]*exp(betaT[i])/epsilon_2, phiT[i])
        survivalT[i] = rand(surv_distT, 1)[1]

        gap_indexC = sample([1:1:length(mlC[i]);], StatsBase.fweights(mlC[i]), 1)[1]
        lambdaC[i] = lambda_list[i][gap_indexC]
        gammaC[i] = gamma_list[i][gap_indexC,1]
        etaC[i] = eta_list[i][gap_indexC]
        gap_distC = LogLogistic(lambdaC[i] / xi_1, etaC[i])

        gap_indexT = sample([1:1:length(mlT[i]);], StatsBase.fweights(mlT[i]), 1)[1]
        lambdaT[i] = lambda_list[i][gap_indexT]
        gammaT[i] = gamma_list[i][gap_indexT,1]
        etaT[i] = eta_list[i][gap_indexT]
        gap_distT = LogLogistic(lambdaT[i]*exp(gammaT[i])/xi_2, etaT[i])

        tmp_arrivalC = zeros(1)
        tmp_gapC = zeros(0)
        while true 
            gap_tmpC = rand(gap_distC, 1)[1]
            if tmp_arrivalC[end] + gap_tmpC > survivalC[i]
                break 
            end 

            push!(tmp_gapC, gap_tmpC)
            push!(tmp_arrivalC, tmp_arrivalC[end] + gap_tmpC)
        end

        push!(gapC, tmp_gapC)
        push!(arrivalC, tmp_arrivalC[2:end])

        tmp_arrivalT = zeros(1)
        tmp_gapT = zeros(0)
        while true 
            gap_tmpT = rand(gap_distT, 1)[1]
            if tmp_arrivalT[end] + gap_tmpT > survivalT[i]
                break 
            end 

            push!(tmp_gapT, gap_tmpT)
            push!(tmp_arrivalT, tmp_arrivalT[end] + gap_tmpT)
        end

        push!(gapT, tmp_gapT)
        push!(arrivalT, tmp_arrivalT[2:end])
    end 

    return [survivalC, gapC, arrivalC, thetaC, betaC, phiC, lambdaC, gammaC, etaC,
            survivalT, gapT, arrivalT, thetaT, betaT, phiT, lambdaT, gammaT, etaT]
end


function survival_functional_estimation_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    phi_save, 
    theta_save, 
    beta_save,
    alpha_saveC,
    alpha_saveT,
    nl_saveC, 
    nl_saveT,
    a_phi,
    b_phi_save, 
    mu_theta_save, 
    sigma2_theta, 
    mu_beta_save,
    sigma2_beta,
    grids 
    )

    nkeep = length(alpha_saveC)
    ngrids = length(grids) 

    densC = zeros(nkeep, ngrids)
    survC = zeros(nkeep, ngrids)
    densT = zeros(nkeep, ngrids)
    survT = zeros(nkeep, ngrids)

    @showprogress for i in 1:nkeep

        phi_star = phi_save[i]
        theta_star = theta_save[i] 
        beta_star = beta_save[i]

        alphaC = alpha_saveC[i]
        alphaT = alpha_saveT[i]

        nlC = nl_saveC[i]
        nlT = nl_saveT[i]

        indC = findall(nlC .> 0)
        qC = rand(Dirichlet(vcat(nlC[indC], alphaC)), 1)[:,1]
        indT = findall(nlT .> 0)
        qT = rand(Dirichlet(vcat(nlT[indT], alphaT)), 1)[:,1]

        NN = 20
        VC = rand(Beta(1, alphaC), NN-1)
        VT = rand(Beta(1, alphaT), NN-1)

        logpC = zeros(NN) 
        logpT = zeros(NN) 
        logpC[1] = log(VC[1])
        logpT[1] = log(VT[1])
        for r in 2:(NN-1)
            logpC[r] = log(VC[r]) + sum(log.(1 .- VC[1:(r-1)]))
            logpT[r] = log(VT[r]) + sum(log.(1 .- VT[1:(r-1)]))
        end
        logpC[NN] = sum(log.(1 .- VC[1:(NN-1)]))
        logpT[NN] = sum(log.(1 .- VT[1:(NN-1)]))
        pC = exp.(logpC)
        pT = exp.(logpT)

        theta_tilde = zeros(NN)
        beta_tilde = zeros(NN)
        phi_tilde = zeros(NN)
        for l in 1:NN
            G0_phi = Gamma(a_phi, b_phi_save[i])
            G0_theta = Normal(mu_theta_save[i], sqrt(sigma2_theta))
            G0_beta = Normal(mu_beta_save[i], sqrt(sigma2_beta))
            phi_tilde[l] = sqrt(rand(G0_phi, 1)[1])
            theta_tilde[l] = exp(rand(G0_theta, 1)[1])
            beta_tilde[l] = rand(G0_beta, 1)[1]
        end 

        nrep = 100
        epsilon_repC = rand(LogNormal(0, sqrt(Sigma_e_1_save[i,1,1])), nrep)
        epsilon_repT = rand(LogNormal(0, sqrt(Sigma_e_2_save[i,1,1])), nrep)
        for g in eachindex(grids) 
            for i_rep in 1:nrep 
                for l in 1:NN
                    surv_modelC = LogLogistic(theta_tilde[l]/epsilon_repC[i_rep], phi_tilde[l])
                    densC[i,g] += 1/nrep * qC[end] * pC[l] * pdf(surv_modelC, grids[g])
                    survC[i,g] += 1/nrep * qC[end] * pC[l] * ccdf(surv_modelC, grids[g])
                    surv_modelT = LogLogistic(theta_tilde[l]*exp(beta_tilde[l])/epsilon_repT[i_rep], phi_tilde[l])
                    densT[i,g] += 1/nrep * qT[end] * pT[l] * pdf(surv_modelT, grids[g])
                    survT[i,g] += 1/nrep * qT[end] * pT[l] * ccdf(surv_modelT, grids[g])
                end
                # for l in 1:k
                for l in eachindex(indC)
                    surv_modelC = LogLogistic(theta_star[l]/epsilon_repC[i_rep], phi_star[l])
                    densC[i,g] += 1/nrep * qC[l] * pdf(surv_modelC, grids[g])
                    survC[i,g] += 1/nrep * qC[l] * ccdf(surv_modelC, grids[g])
                end
                for l in eachindex(indT)
                    surv_modelT = LogLogistic(theta_star[l]*exp(beta_star[l])/epsilon_repT[i_rep], phi_star[l])
                    densT[i,g] += 1/nrep * qT[l] * pdf(surv_modelT, grids[g])
                    survT[i,g] += 1/nrep * qT[l] * ccdf(surv_modelT, grids[g])
                end
            end
        end
    end 

    return Dict("dC" => densC, "sC" => survC, "dT" => densT, "sT" => survT)

end


function functional_estimation_blocked_gibbs_cov(
    sigma2_e_1, 
    sigma2_e_2,
    theta,
    beta,
    phi,
    logpC,
    logpT,
    x0,
    x1,
    grids,
    B # truncation level
    )

    nkeep = length(sigma2_e_1)
    ngrids = length(grids)
    densC = zeros(nkeep, ngrids)
    survC = zeros(nkeep, ngrids)
    densT = zeros(nkeep, ngrids)
    survT = zeros(nkeep, ngrids)
    nrep = 200 

    @showprogress for i in 1:nkeep

        pC_i = exp.(logpC[i,:])
        pT_i = exp.(logpT[i,:])

        theta_i = theta[i,:]
        beta_i = beta[i,:,:]
        phi_i = phi[i,:]

        epsilon_predC = rand(LogNormal(0, sqrt(sigma2_e_1[i])), nrep)
        epsilon_predT = rand(LogNormal(0, sqrt(sigma2_e_2[i])), nrep)
        for g in eachindex(grids)

            for i_rep in 1:nrep
                for l in 1:B
                    if pC_i[l] > 1e-10
                        densC[i,g] += 1/nrep * pC_i[l] * pdf(LogLogistic(theta_i[l]*exp(beta_i[l,:]'*x0)/epsilon_predC[i_rep], phi_i[l]), grids[g])
                        survC[i,g] += 1/nrep * pC_i[l] * ccdf(LogLogistic(theta_i[l]*exp(beta_i[l,:]'*x0)/epsilon_predC[i_rep], phi_i[l]), grids[g])
                    end
                    if pT_i[l] > 1e-10
                        densT[i,g] += 1/nrep * pT_i[l] * pdf(LogLogistic(theta_i[l]*exp(beta_i[l,:]'*x1)/epsilon_predT[i_rep], phi_i[l]), grids[g])
                        survT[i,g] += 1/nrep * pT_i[l] * ccdf(LogLogistic(theta_i[l]*exp(beta_i[l,:]'*x1)/epsilon_predT[i_rep], phi_i[l]), grids[g])
                    end
                end
            end
        end
    end

    return Dict("densC" => densC, "survC" => survC, "densT" => densT, "survT" => survT)
end



function conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1, 
    Sigma_e_2,
    theta,
    beta,
    phi,
    logpC,
    logpT,
    x0,
    x1,
    BG, # truncation level
    lambda,
    gamma,
    eta,
    logomegaC,
    logomegaT,
    z0,
    z1,
    BH, # truncation level
    grids
    )

    nkeep = size(Sigma_e_1)[1]
    ngrids = length(grids)
    nrep = 200

    surv_survC = zeros(nkeep, ngrids, nrep)
    gap_survC0 = zeros(nkeep, nrep)

    surv_survT = zeros(nkeep, ngrids, nrep)
    gap_survT0 = zeros(nkeep, nrep)

    @showprogress for i in 1:nkeep

        re_predC = rand(MvLogNormal(zeros(2), Sigma_e_1[i,:,:]), nrep)
        epsilon_predC = re_predC[1,:]
        xi_predC = re_predC[2,:]
        re_predT = rand(MvLogNormal(zeros(2), Sigma_e_2[i,:,:]), nrep)
        epsilon_predT = re_predT[1,:]
        xi_predT = re_predT[2,:]

        pC_i = exp.(logpC[i,:])
        pT_i = exp.(logpT[i,:])

        theta_i = theta[i,:]
        beta_i = beta[i,:,:]
        phi_i = phi[i,:]

        omegaC_i = exp.(logomegaC[i,:])
        omegaT_i = exp.(logomegaT[i,:])

        lambda_i = lambda[i,:]
        gamma_i = gamma[i,:,:]
        eta_i = eta[i,:]

        for i_rep in 1:nrep
            for l in 1:BH
                if omegaC_i[l] > 1e-5
                    gap_survC0[i,i_rep] += omegaC_i[l] * ccdf(LogLogistic(lambda_i[l]*exp(gamma_i[l,:]'*z0)/xi_predC[i_rep], eta_i[l]), grids[1])
                end
                if omegaT_i[l] > 1e-5
                    gap_survT0[i,i_rep] += omegaT_i[l] * ccdf(LogLogistic(lambda_i[l]*exp(gamma_i[l,:]'*z1)/xi_predT[i_rep], eta_i[l]), grids[1])
                end
            end
        end

        for g in eachindex(grids)
            for i_rep in 1:nrep
                for l in 1:BG
                    if pC_i[l] > 1e-5
                        surv_survC[i,g,i_rep] += pC_i[l] * ccdf(LogLogistic(theta_i[l]*exp(beta_i[l,:]'*x0)/epsilon_predC[i_rep], phi_i[l]), grids[g])
                    end
                    if pT_i[l] > 1e-5
                        surv_survT[i,g,i_rep] += pT_i[l] * ccdf(LogLogistic(theta_i[l]*exp(beta_i[l,:]'*x1)/epsilon_predT[i_rep], phi_i[l]), grids[g])
                    end
                end
            end
        end
    end

    condC = zeros(nkeep, ngrids)
    condT = zeros(nkeep, ngrids)
    @showprogress for i in 1:nkeep
        lowerC = 0.0
        upperC = zeros(ngrids)
        lowerT = 0.0
        upperT = zeros(ngrids)

        for i_rep in 1:nrep
            lowerC += 1/nrep * surv_survC[i,1,i_rep] * gap_survC0[i,i_rep]
            upperC[1] = lowerC
            lowerT += 1/nrep * surv_survT[i,1,i_rep] * gap_survT0[i,i_rep]
            upperT[1] = lowerT
            for g in 2:ngrids
                upperC[g] += 1/nrep * surv_survC[i,g,i_rep] * gap_survC0[i,i_rep]
                upperT[g] += 1/nrep * surv_survT[i,g,i_rep] * gap_survT0[i,i_rep]
            end
        end
        for g in 1:ngrids
            condC[i,g] = upperC[g] / lowerC
            condT[i,g] = upperT[g] / lowerT
        end
    end

    return Dict("condC" => condC, "condT" => condT)
end


function G_corr_calc(alpha, alpha0)
    n = length(alpha)
    corr = zeros(n)

    for i in 1:n
        corr[i] = (2 - alpha0[i]) / (2 + 2*alpha[i] + alpha0[i])
    end

    return corr
end

function predict_blocked_gibbs_cov(
    logpC,
    logpT,
    theta,
    beta,
    phi,
    BG,
    logomegaC,
    logomegaT,
    lambda,
    gamma,
    eta,
    BH
    )
    nkeep = size(theta)[1]
    p = size(beta)[3]
    q = size(gamma)[3]
    
    theta_predC = zeros(nkeep)
    theta_predT = zeros(nkeep)
    beta_predC = zeros(nkeep, p)
    beta_predT = zeros(nkeep, p)
    phi_predC = zeros(nkeep)
    phi_predT = zeros(nkeep)

    lambda_predC = zeros(nkeep)
    lambda_predT = zeros(nkeep)
    gamma_predC = zeros(nkeep, q)
    gamma_predT = zeros(nkeep, q)
    eta_predC = zeros(nkeep)
    eta_predT = zeros(nkeep)

    for i in 1:nkeep
        pC = exp.(logpC[i,:])
        pT = exp.(logpT[i,:])
        surv_indC = sample([1:1:BG;], StatsBase.pweights(pC))
        surv_indT = sample([1:1:BG;], StatsBase.pweights(pT))

        theta_predC[i] = theta[i,surv_indC]
        theta_predT[i] = theta[i,surv_indT]
        beta_predC[i,:] = beta[i,surv_indC,:]
        beta_predT[i,:] = beta[i,surv_indT,:]
        phi_predC[i] = phi[i,surv_indC]
        phi_predT[i] = phi[i,surv_indT]

        omegaC = exp.(logomegaC[i,:])
        omegaT = exp.(logomegaT[i,:])
        gap_indC = sample([1:1:BH;], StatsBase.pweights(omegaC))
        gap_indT = sample([1:1:BH;], StatsBase.pweights(omegaT))
        lambda_predC[i] = lambda[i,gap_indC]
        lambda_predT[i] = lambda[i,gap_indT]
        gamma_predC[i,:] = gamma[i,gap_indC,:]
        gamma_predT[i,:] = gamma[i,gap_indT,:]
        eta_predC[i] = eta[i,gap_indC]
        eta_predT[i] = eta[i,gap_indT]
    end

    nkeep = size(thetaC)[1]

    survival_predC = zeros(nkeep)
    survival_predT = zeros(nkeep)
    gap_predC = []
    gap_predT = []
    Nvec_predC = zeros(Int64, nkeep)
    Nvec_predT = zeros(Int64, nkeep)

    for i in 1:nkeep
        survival_predC[i] = rand(LogLogistic(thetaC[i] * exp(x0' * betaC[i,:])/epsilonC[i], phiC[i]), 1)[1]

        tmp_arrivalC = zeros(1)
        tmp_gapC = zeros(0)
        while true
            gap_tmpC = rand(LogLogistic(lambdaC[i]*exp(z0' * gammaC[i,:])/xiC[i], etaC[i]), 1)[1]
            if tmp_arrivalC[end] + gap_tmpC > survival_predC[i]
                break 
            end 

            push!(tmp_gapC, gap_tmpC)
            push!(tmp_arrivalC, tmp_arrivalC[end] + gap_tmpC)
        end
        push!(gap_predC, tmp_gapC)
        Nvec_predC[i] = length(tmp_gapC)

        survival_predT[i] = rand(LogLogistic(thetaT[i] * exp(x1' * betaT[i,:])/epsilonT[i], phiT[i]), 1)[1]

        tmp_arrivalT = zeros(1)
        tmp_gapT = zeros(0)
        while true
            gap_tmpT = rand(LogLogistic(lambdaT[i]*exp(z1'*gammaT[i,:])/xiT[i], etaT[i]), 1)[1]
            if tmp_arrivalT[end] + gap_tmpT > survival_predT[i]
                break 
            end 

            push!(tmp_gapT, gap_tmpT)
            push!(tmp_arrivalT, tmp_arrivalT[end] + gap_tmpT)
        end
        push!(gap_predT, tmp_gapT)
        Nvec_predT[i] = length(tmp_gapT)
    end 
    return Dict("survivalC" => survival_predC, "survivalT" => survival_predT,
                "gapC" => gap_predC, "gapT" => gap_predT, 
                "NvecC" => Nvec_predC, "NvecT" => Nvec_predT)
end

function functional_estimation_blocked_gibbs_common_atoms(
    sigma2_e_1, 
    sigma2_e_2,
    theta,
    phi,
    logpC,
    logpT,
    grids,
    B # truncation level
    )

    nkeep = length(sigma2_e_1)
    ngrids = length(grids)
    densC = zeros(nkeep, ngrids)
    survC = zeros(nkeep, ngrids)
    densT = zeros(nkeep, ngrids)
    survT = zeros(nkeep, ngrids)
    nrep = 200

    @showprogress for i in 1:nkeep

        epsilon_predC = rand(LogNormal(0, sqrt(sigma2_e_1[i])), nrep)
        epsilon_predT = rand(LogNormal(0, sqrt(sigma2_e_2[i])), nrep)

        pC_i = exp.(logpC[i,:])
        pT_i = exp.(logpT[i,:])

        theta_i = theta[i,:]
        phi_i = phi[i,:]

        for g in eachindex(grids)
            for i_rep in 1:nrep
                for l in 1:B
                    if pC_i[l] > 1e-10
                        densC[i,g] += pC_i[l] * pdf(LogLogistic(theta_i[l]/epsilon_predC[i_rep], phi_i[l]), grids[g])
                        survC[i,g] += pC_i[l] * ccdf(LogLogistic(theta_i[l]/epsilon_predC[i_rep], phi_i[l]), grids[g])
                    end
                    if pT_i[l] > 1e-10
                        densT[i,g] += pT_i[l] * pdf(LogLogistic(theta_i[l]/epsilon_predT[i_rep], phi_i[l]), grids[g])
                        survT[i,g] += pT_i[l] * ccdf(LogLogistic(theta_i[l]/epsilon_predT[i_rep], phi_i[l]), grids[g])
                    end
                end
            end
        end
    end
    densC = densC ./ nrep
    densT = densT ./ nrep
    survC = survC ./ nrep
    survT = survT ./ nrep

    return Dict("densC" => densC, "survC" => survC, "densT" => densT, "survT" => survT)
end


function conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1, 
    Sigma_e_2,
    theta,
    phi,
    logpC,
    logpT,
    BG, # truncation level
    lambda,
    eta,
    logomegaC,
    logomegaT,
    BH, # truncation level
    grids
    )

    nkeep = size(Sigma_e_1)[1]
    ngrids = length(grids)
    nrep = 200

    surv_survC = zeros(nkeep, ngrids, nrep)
    gap_survC0 = zeros(nkeep, nrep)

    surv_survT = zeros(nkeep, ngrids, nrep)
    gap_survT0 = zeros(nkeep, nrep)

    @showprogress for i in 1:nkeep

        re_predC = rand(MvLogNormal(zeros(2), Sigma_e_1[i,:,:]), nrep)
        epsilon_predC = re_predC[1,:]
        xi_predC = re_predC[2,:]

        re_predT = rand(MvLogNormal(zeros(2), Sigma_e_2[i,:,:]), nrep)
        epsilon_predT = re_predT[1,:]
        xi_predT = re_predT[2,:]

        pC_i = exp.(logpC[i,:])
        pT_i = exp.(logpT[i,:])

        theta_i = theta[i,:]
        phi_i = phi[i,:]

        omegaC_i = exp.(logomegaC[i,:])
        omegaT_i = exp.(logomegaT[i,:])

        lambda_i = lambda[i,:]
        eta_i = eta[i,:]

        for i_rep in 1:nrep
            for l in 1:BH
                if omegaC_i[l] > 1e-10
                    gap_survC0[i,i_rep] += omegaC_i[l] * ccdf(LogLogistic(lambda_i[l]/xi_predC[i_rep], eta_i[l]), grids[1])
                end
                if omegaT_i[l] > 1e-10
                    gap_survT0[i,i_rep] += omegaT_i[l] * ccdf(LogLogistic(lambda_i[l]/xi_predT[i_rep], eta_i[l]), grids[1])
                end
            end
        end

        for g in eachindex(grids)
            for i_rep in 1:nrep
                for l in 1:BG
                    if pC_i[l] > 1e-10
                        surv_survC[i,g,i_rep] += pC_i[l] * ccdf(LogLogistic(theta_i[l]/epsilon_predC[i_rep], phi_i[l]), grids[g])
                    end
                    if pT_i[l] > 1e-10
                        surv_survT[i,g,i_rep] += pT_i[l] * ccdf(LogLogistic(theta_i[l]/epsilon_predT[i_rep], phi_i[l]), grids[g])
                    end
                end
            end
        end
    end

    condC = zeros(nkeep, ngrids)
    condT = zeros(nkeep, ngrids)
    @showprogress for i in 1:nkeep
        lowerC = 0.0
        upperC = zeros(ngrids)
        lowerT = 0.0
        upperT = zeros(ngrids)

        for i_rep in 1:nrep
            lowerC += 1/nrep * surv_survC[i,1,i_rep] * gap_survC0[i,i_rep]
            upperC[1] = lowerC
            lowerT += 1/nrep * surv_survT[i,1,i_rep] * gap_survT0[i,i_rep]
            upperT[1] = lowerT
            for g in 2:ngrids
                upperC[g] += 1/nrep * surv_survC[i,g,i_rep] * gap_survC0[i,i_rep]
                upperT[g] += 1/nrep * surv_survT[i,g,i_rep] * gap_survT0[i,i_rep]
            end
        end
        for g in 1:ngrids
            condC[i,g] = upperC[g] / lowerC
            condT[i,g] = upperT[g] / lowerT
        end
    end

    return Dict("condC" => condC, "condT" => condT, "surv_survC" => surv_survC, "gap_survC0" => gap_survC0, "surv_survT" => surv_survT, "gap_survT0" => gap_survT0) # , "upperC" => upperC, "lowerC" => lowerC, "upperT" => upperT, "lowerT" => lowerT
end


function predict_blocked_gibbs_common_atoms(
    logpC,
    logpT,
    theta,
    phi,
    BG,
    logomegaC,
    logomegaT,
    lambda,
    eta,
    BH
    )
    nkeep = size(theta)[1]
    
    theta_predC = zeros(nkeep)
    theta_predT = zeros(nkeep)
    phi_predC = zeros(nkeep)
    phi_predT = zeros(nkeep)

    lambda_predC = zeros(nkeep)
    lambda_predT = zeros(nkeep)
    eta_predC = zeros(nkeep)
    eta_predT = zeros(nkeep)

    for i in 1:nkeep
        pC = exp.(logpC[i,:])
        pT = exp.(logpT[i,:])
        surv_indC = sample([1:1:BG;], StatsBase.pweights(pC))
        surv_indT = sample([1:1:BG;], StatsBase.pweights(pT))

        theta_predC[i] = theta[i,surv_indC]
        theta_predT[i] = theta[i,surv_indT]
        phi_predC[i] = phi[i,surv_indC]
        phi_predT[i] = phi[i,surv_indT]

        omegaC = exp.(logomegaC[i,:])
        omegaT = exp.(logomegaT[i,:])
        gap_indC = sample([1:1:BH;], StatsBase.pweights(omegaC))
        gap_indT = sample([1:1:BH;], StatsBase.pweights(omegaT))

        lambda_predC[i] = lambda[i,gap_indC]
        lambda_predT[i] = lambda[i,gap_indT]
        eta_predC[i] = eta[i,gap_indC]
        eta_predT[i] = eta[i,gap_indT]
    end

    return Dict("thetaC" => theta_predC, "thetaT" => theta_predT,
                "phiC" => phi_predC, "phiT" => phi_predT,
                "lambdaC" => lambda_predC, "lambdaT" => lambda_predT,
                "etaC" => eta_predC, "etaT" => eta_predT)
end


function predict_recurrent(lambda_list, eta_list, ml, xi_list, survival)
    nkeep = length(ml)

    arrival = []
    Nvec = zeros(nkeep)
    
    @showprogress for i in 1:nkeep
        gap_index = sample([1:1:length(ml[i]);], StatsBase.fweights(ml[i]), 1)[1]
        gap_dist = LogLogistic(lambda_list[i][gap_index]/xi_list[i], eta_list[i][gap_index])

        tmp_arrival = zeros(1)
        tmp_gap = zeros(0)
        while true 
            gap_tmp = rand(gap_dist, 1)[1]
            if tmp_arrival[end] + gap_tmp > survival
                break 
            end 

            Nvec[i] += 1
            push!(tmp_gap, gap_tmp)
            push!(tmp_arrival, tmp_arrival[end] + gap_tmp)
        end

        push!(arrival, tmp_arrival[2:end])
    end

    return [arrival, Nvec]
end



function predict_recurrent_blocked_gibbs_cov(lambda, eta, gamma, logomega, xi_list, z0, survival)

    arrival = []
    gap = []
    nkeep = size(logomega)[1]
    N = size(logomega)[2]
    
    Nvec = zeros(nkeep)
    @showprogress for i in 1:nkeep
        gap_index = sample([1:1:N;], StatsBase.pweights(exp.(logomega[i,:])), 1)[1]
        gap_dist = LogLogistic(lambda[i,gap_index] * exp(gamma[i,gap_index,:]' * z0) /xi_list[i], eta[i, gap_index])

        tmp_arrival = zeros(1)
        tmp_gap = zeros(0)
        while true 
            gap_tmp = rand(gap_dist, 1)[1]

            Nvec[i] += 1
            push!(tmp_gap, gap_tmp)
            push!(tmp_arrival, tmp_arrival[end] + gap_tmp)

            if tmp_arrival[end] + gap_tmp > survival
                break 
            end 
        end
        
        push!(gap, tmp_gap)
        push!(arrival, tmp_arrival[2:end])
    end

    return [arrival, gap, Nvec]
end
