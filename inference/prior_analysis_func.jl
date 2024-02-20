function prior_dp(alpha::Real, mu::Real, sigma2::Real, a::Real, b::Real)
    """
    LogLogistic(theta, phi) 
    log(theta) ~ N(mu ,sigma)
    phi^2 ~ Ga(a, b)
    """

    N = 50
 
    V = rand(Beta(1, alpha), N-1)
    logp = zeros(N) 
    logp[1] = log(V[1])
    for r in 2:(N-1)
        logp[r] = log(V[r]) + sum(log.(1 .- V[1:(r-1)]))
    end
    logp[N] = sum(log.(1 .- V[1:(N-1)]))
    weights = exp.(logp)

    baseline_dists = [Normal(mu, sqrt(sigma2)), Gamma(a, b)]
    atoms_list = [] 
    for baseline_dist in baseline_dists
        atoms = rand(baseline_dist, N)
        push!(atoms_list, atoms)
    end
 
    res = Dict()
    res["weights"] = weights
    res["atoms"] = atoms_list

    return res 
end 


function prior_dp_regression(alpha::Real, mu::Real, sigma2::Real, mu_beta::Vector{Float64}, Sigma_beta::Matrix{Float64}, a::Real, b::Real)
    """
    LogLogistic(theta, phi) 
    log(theta) ~ N(mu ,sigma)
    phi^2 ~ Ga(a, b)
    """

    N = 50
 
    V = rand(Beta(1, alpha), N-1)
    logp = zeros(N) 
    logp[1] = log(V[1])
    for r in 2:(N-1)
        logp[r] = log(V[r]) + sum(log.(1 .- V[1:(r-1)]))
    end
    logp[N] = sum(log.(1 .- V[1:(N-1)]))
    weights = exp.(logp)

    baseline_dists = [Normal(mu, sqrt(sigma2)), MvNormal(mu_beta, Sigma_beta), Gamma(a, b)]
    atoms_list = [] 
    for baseline_dist in baseline_dists
        atoms = rand(baseline_dist, N)
        push!(atoms_list, atoms)
    end
 
    res = Dict()
    res["weights"] = weights
    res["atoms"] = atoms_list

    return res 
end 


function prior_random_effects_dp(upsilon, Sigma_e)
    """
    (ϵ,ξ) ~ DP(υ, M₀)
    M₀ = LN₂((0,0)', Sigma_e)
    """

    N = 50 
    V = rand(Beta(1, upsilon), N-1)
    logp = zeros(N) 
    logp[1] = log(V[1])
    for r in 2:(N-1)
        logp[r] = log(V[r]) + sum(log.(1 .- V[1:(r-1)]))
    end
    logp[N] = sum(log.(1 .- V[1:(N-1)]))
    weights = exp.(logp)

    baseline_dist = MvLogNormal(zeros(2), Sigma_e) 
    atoms = rand(baseline_dist, N)
 
    res = Dict()
    res["weights"] = weights
    res["atoms"] = atoms

    return res  
end



function sample_random_effects_dp(weights, atoms, n)

    idx = sample([1:1:length(weights);], StatsBase.pweights(weights), n)
    re = atoms[:,idx]

    return re
end



# 1
function conditional_probability_fixed(Sigma_e, grids, theta, phi, lambda, eta)
    # single kernel functions for survival and gap times with fixed parameters 
    # tuning Σₑ

    n_re=50000

    re = rand(MvLogNormal(zeros(2), Sigma_e), n_re)
    epsilon = re[1,:]
    xi = re[2,:]

    ngrids = length(grids)

    surv_surv = zeros(n_re, ngrids)
    gap_surv0 = zeros(n_re)
    gap_cdf0 = zeros(n_re)
    @showprogress for i_re in 1:n_re 
        surv_model = LogLogistic(theta/epsilon[i_re], phi)
        gap_model = LogLogistic(lambda/xi[i_re], eta)
        gap_surv0[i_re] = ccdf(gap_model, grids[1])
        gap_cdf0[i_re] = 1 - gap_surv0[i_re] 
        for g in eachindex(grids)
            surv_surv[i_re, g] = ccdf(surv_model, grids[g])
        end
    end

    cond_surv_N0 = zeros(ngrids)
    cond_surv_N1 = zeros(ngrids)
    for g in eachindex(grids)
        cond_surv_N0[g] = mean(surv_surv[:,g] .* gap_surv0) / mean(surv_surv[:,1] .* gap_surv0)
        cond_surv_N1[g] = mean(surv_surv[:,g] .* gap_cdf0) / mean(surv_surv[:,1] .* gap_cdf0)
    end
    
    return [cond_surv_N0, cond_surv_N1]
end


# 2
function dpm_conditional_probability_fixed(Sigma_e, grids, theta_list, phi_list, weights1, lambda_list, eta_list, weights2)
    # single kernel functions for survival and gap times with fixed parameters 
    # tuning Σₑ

    n_re=50000

    re = rand(MvLogNormal(zeros(2), Sigma_e), n_re)
    epsilon = re[1,:]
    xi = re[2,:]

    N = length(weights1)

    ngrids = length(grids)

    surv_surv = zeros(n_re, ngrids)
    gap_surv0 = zeros(n_re)
    gap_cdf0 = zeros(n_re)
    @showprogress for i_re in 1:n_re 
        for l in 1:N
            surv_model = LogLogistic(theta_list[l]/epsilon[i_re], phi_list[l])
            gap_model = LogLogistic(lambda_list[l]/xi[i_re], eta_list[l])

            gap_surv0[i_re] += weights2[l] * ccdf(gap_model, grids[1])
            gap_cdf0[i_re] += 1 - gap_surv0[i_re] 
            for g in eachindex(grids)
                surv_surv[i_re, g] += weights1[l] * ccdf(surv_model, grids[g])
            end
        end
    end

    cond_surv_N0 = zeros(ngrids)
    cond_surv_N1 = zeros(ngrids)
    for g in eachindex(grids)
        cond_surv_N0[g] = mean(surv_surv[:,g] .* gap_surv0) / mean(surv_surv[:,1] .* gap_surv0)
        cond_surv_N1[g] = mean(surv_surv[:,g] .* gap_cdf0) / mean(surv_surv[:,1] .* gap_cdf0)
    end
    
    return [cond_surv_N0, cond_surv_N1]
end



# 3
function shared_conditional_probability_fixed(sigma2_e, power, grids, theta, phi, lambda, eta)

    n_re=50000

    xi = rand(LogNormal(0, sqrt(sigma2_e)), n_re)
    epsilon = xi.^power 

    ngrids = length(grids)

    surv_surv = zeros(n_re, ngrids)
    gap_surv0 = zeros(n_re)
    gap_cdf0 = zeros(n_re)
    @showprogress for i_re in 1:n_re 
        surv_model = LogLogistic(theta/epsilon[i_re], phi)
        gap_model = LogLogistic(lambda/xi[i_re], eta)
        gap_surv0[i_re] = ccdf(gap_model, grids[1])
        gap_cdf0[i_re] = 1 - gap_surv0[i_re] 
        for g in eachindex(grids)
            surv_surv[i_re, g] = ccdf(surv_model, grids[g])
        end
    end

    cond_surv_N0 = zeros(ngrids)
    cond_surv_N1 = zeros(ngrids)
    for g in eachindex(grids)
        cond_surv_N0[g] = mean(surv_surv[:,g] .* gap_surv0) / mean(surv_surv[:,1] .* gap_surv0)
        cond_surv_N1[g] = mean(surv_surv[:,g] .* gap_cdf0) / mean(surv_surv[:,1] .* gap_cdf0)
    end
    
    return [cond_surv_N0, cond_surv_N1]
end



# 4
function shared_dpm_conditional_probability_fixed(sigma2_e, power, grids, theta_list, phi_list, weights1, lambda_list, eta_list, weights2)
    # single kernel functions for survival and gap times with fixed parameters 
    # tuning Σₑ

    n_re=50000

    xi = rand(LogNormal(0, sqrt(sigma2_e)), n_re)
    epsilon = xi.^power 

    N = length(weights1)

    ngrids = length(grids)

    surv_surv = zeros(n_re, ngrids)
    gap_surv0 = zeros(n_re)
    gap_cdf0 = zeros(n_re)
    @showprogress for i_re in 1:n_re 
        for l in 1:N
            surv_model = LogLogistic(theta_list[l]/epsilon[i_re], phi_list[l])
            gap_model = LogLogistic(lambda_list[l]/xi[i_re], eta_list[l])

            gap_surv0[i_re] += weights2[l] * ccdf(gap_model, grids[1])
            gap_cdf0[i_re] += 1 - gap_surv0[i_re] 
            for g in eachindex(grids)
                surv_surv[i_re, g] += weights1[l] * ccdf(surv_model, grids[g])
            end
        end
    end

    cond_surv_N0 = zeros(ngrids)
    cond_surv_N1 = zeros(ngrids)
    for g in eachindex(grids)
        cond_surv_N0[g] = mean(surv_surv[:,g] .* gap_surv0) / mean(surv_surv[:,1] .* gap_surv0)
        cond_surv_N1[g] = mean(surv_surv[:,g] .* gap_cdf0) / mean(surv_surv[:,1] .* gap_cdf0)
    end
    
    return [cond_surv_N0, cond_surv_N1]
end



# 5
function dpm_conditional_probability_dpre(DP_dict, grids, theta_list, phi_list, weights1, lambda_list, eta_list, weights2)

    n_re=50000

    re = sample_random_effects_dp(DP_dict["weights"], DP_dict["atoms"], n_re)
    epsilon = re[1,:]
    xi = re[2,:]

    N = length(weights1)

    ngrids = length(grids)

    surv_surv = zeros(n_re, ngrids)
    gap_surv0 = zeros(n_re)
    gap_cdf0 = zeros(n_re)
    @showprogress for i_re in 1:n_re 
        for l in 1:N
            surv_model = LogLogistic(theta_list[l]/epsilon[i_re], phi_list[l])
            gap_model = LogLogistic(lambda_list[l]/xi[i_re], eta_list[l])

            gap_surv0[i_re] += weights2[l] * ccdf(gap_model, grids[1])
            gap_cdf0[i_re] += 1 - gap_surv0[i_re] 
            for g in eachindex(grids)
                surv_surv[i_re, g] += weights1[l] * ccdf(surv_model, grids[g])
            end
        end
    end

    cond_surv_N0 = zeros(ngrids)
    cond_surv_N1 = zeros(ngrids)
    for g in eachindex(grids)
        cond_surv_N0[g] = mean(surv_surv[:,g] .* gap_surv0) / mean(surv_surv[:,1] .* gap_surv0)
        cond_surv_N1[g] = mean(surv_surv[:,g] .* gap_cdf0) / mean(surv_surv[:,1] .* gap_cdf0)
    end
    
    return [cond_surv_N0, cond_surv_N1]
end


# 6
function dpm_conditional_probability_fixed_regression(Sigma_e, grids, x0, z0, theta_list, beta_list, phi_list, weights1, lambda_list, gamma_list, eta_list, weights2)
    # single kernel functions for survival and gap times with fixed parameters 
    # tuning Σₑ

    n_re = 50000

    re = rand(MvLogNormal(zeros(2), Sigma_e), n_re)
    epsilon = re[1,:]
    xi = re[2,:]

    N = length(weights1)

    ngrids = length(grids)

    surv_surv = zeros(n_re, ngrids)
    gap_surv0 = zeros(n_re)
    gap_cdf0 = zeros(n_re)
    @showprogress for i_re in 1:n_re 
        for l in 1:N
            surv_model = LogLogistic(theta_list[l]*exp(beta_list[:,l]'*x0)/epsilon[i_re], phi_list[l])
            gap_model = LogLogistic(lambda_list[l]*exp(gamma_list[:,l]'*z0)/xi[i_re], eta_list[l])

            gap_surv0[i_re] += weights2[l] * ccdf(gap_model, grids[1])
            gap_cdf0[i_re] += 1 - gap_surv0[i_re] 
            for g in eachindex(grids)
                surv_surv[i_re, g] += weights1[l] * ccdf(surv_model, grids[g])
            end
        end
    end

    cond_surv_N0 = zeros(ngrids)
    cond_surv_N1 = zeros(ngrids)
    for g in eachindex(grids)
        cond_surv_N0[g] = mean(surv_surv[:,g] .* gap_surv0) / mean(surv_surv[:,1] .* gap_surv0)
        cond_surv_N1[g] = mean(surv_surv[:,g] .* gap_cdf0) / mean(surv_surv[:,1] .* gap_cdf0)
    end
    
    return [cond_surv_N0, cond_surv_N1]
end


# 7
function dpm_conditional_probability_dpre_regression(DP_dict, grids, x0, z0, theta_list, beta_list, phi_list, weights1, lambda_list, gamma_list, eta_list, weights2)
    # single kernel functions for survival and gap times with fixed parameters 
    # tuning Σₑ

    n_re = 50000

    re = sample_random_effects_dp(DP_dict["weights"], DP_dict["atoms"], n_re)

    epsilon = re[1,:]
    xi = re[2,:]

    N = length(weights1)

    ngrids = length(grids)

    surv_surv = zeros(n_re, ngrids)
    gap_surv0 = zeros(n_re)
    gap_cdf0 = zeros(n_re)
    @showprogress for i_re in 1:n_re 
        for l in 1:N
            surv_model = LogLogistic(theta_list[l]*exp(beta_list[:,l]'*x0)/epsilon[i_re], phi_list[l])
            gap_model = LogLogistic(lambda_list[l]*exp(gamma_list[:,l]'*z0)/xi[i_re], eta_list[l])

            gap_surv0[i_re] += weights2[l] * ccdf(gap_model, grids[1])
            gap_cdf0[i_re] += 1 - gap_surv0[i_re] 
            for g in eachindex(grids)
                surv_surv[i_re, g] += weights1[l] * ccdf(surv_model, grids[g])
            end
        end
    end

    cond_surv_N0 = zeros(ngrids)
    cond_surv_N1 = zeros(ngrids)
    for g in eachindex(grids)
        cond_surv_N0[g] = mean(surv_surv[:,g] .* gap_surv0) / mean(surv_surv[:,1] .* gap_surv0)
        cond_surv_N1[g] = mean(surv_surv[:,g] .* gap_cdf0) / mean(surv_surv[:,1] .* gap_cdf0)
    end
    
    return [cond_surv_N0, cond_surv_N1]
end


# 1
function density_fixed(Sigma_e, grids, theta, phi, lambda, eta)

    n_re=50000

    re = rand(MvLogNormal(zeros(2), Sigma_e), n_re)
    epsilon = re[1,:]
    xi = re[2,:]

    ngrids = length(grids)

    surv_dens = zeros(ngrids)
    surv_surv = zeros(ngrids) 
    gap_dens = zeros(ngrids) 
    gap_surv = zeros(ngrids)

    @showprogress for i_re in 1:n_re 
        surv_model = LogLogistic(theta/epsilon[i_re], phi)
        gap_model = LogLogistic(lambda/xi[i_re], eta)

        for g in eachindex(grids)
            surv_dens[g] += pdf(surv_model, grids[g]) / n_re
            surv_surv[g] += ccdf(surv_model, grids[g]) / n_re
            gap_dens[g] += pdf(gap_model, grids[g]) / n_re
            gap_surv[g] += ccdf(gap_model, grids[g]) / n_re
        end
    end

    return [surv_dens, surv_surv, gap_dens, gap_surv]
end



# 2
function dpm_density_fixed(Sigma_e, grids, theta_list, phi_list, weights1, lambda_list, eta_list, weights2)

    n_re=50000

    re = rand(MvLogNormal(zeros(2), Sigma_e), n_re)
    epsilon = re[1,:]
    xi = re[2,:]

    ngrids = length(grids)

    surv_dens = zeros(ngrids)
    surv_surv = zeros(ngrids) 
    gap_dens = zeros(ngrids) 
    gap_surv = zeros(ngrids)

    @showprogress for i_re in 1:n_re 
        surv_params = tuple.(theta_list ./ epsilon[i_re], phi_list)
        surv_model = MixtureModel(LogLogistic, surv_params, weights1)

        gap_params = tuple.(lambda_list ./ xi[i_re], eta_list)
        gap_model = MixtureModel(LogLogistic, gap_params, weights2)

        for g in eachindex(grids)
            surv_dens[g] += pdf(surv_model, grids[g]) / n_re
            surv_surv[g] += ccdf(surv_model, grids[g]) / n_re
            gap_dens[g] += pdf(gap_model, grids[g]) / n_re
            gap_surv[g] += ccdf(gap_model, grids[g]) / n_re
        end
    end

    return [surv_dens, surv_surv, gap_dens, gap_surv]
end



# 3
function shared_density_fixed(sigma2_e, power, grids, theta, phi, lambda, eta)

    n_re=50000

    xi = rand(LogNormal(0, sqrt(sigma2_e)), n_re)
    epsilon = xi.^power 

    ngrids = length(grids)

    surv_dens = zeros(ngrids)
    surv_surv = zeros(ngrids) 
    gap_dens = zeros(ngrids) 
    gap_surv = zeros(ngrids)

    @showprogress for i_re in 1:n_re 
        surv_model = LogLogistic(theta/epsilon[i_re], phi)
        gap_model = LogLogistic(lambda/xi[i_re], eta)

        for g in eachindex(grids)
            surv_dens[g] += pdf(surv_model, grids[g]) / n_re
            surv_surv[g] += ccdf(surv_model, grids[g]) / n_re
            gap_dens[g] += pdf(gap_model, grids[g]) / n_re
            gap_surv[g] += ccdf(gap_model, grids[g]) / n_re
        end
    end

    return [surv_dens, surv_surv, gap_dens, gap_surv]
end



# 4
function shared_dpm_density_fixed(sigma2_e, power, grids, theta_list, phi_list, weights1, lambda_list, eta_list, weights2)

    n_re=50000

    xi = rand(LogNormal(0, sqrt(sigma2_e)), n_re)
    epsilon = xi.^power 

    ngrids = length(grids)

    surv_dens = zeros(ngrids)
    surv_surv = zeros(ngrids) 
    gap_dens = zeros(ngrids) 
    gap_surv = zeros(ngrids)

    @showprogress for i_re in 1:n_re 
        surv_params = tuple.(theta_list ./ epsilon[i_re], phi_list)
        surv_model = MixtureModel(LogLogistic, surv_params, weights1)

        gap_params = tuple.(lambda_list ./ xi[i_re], eta_list)
        gap_model = MixtureModel(LogLogistic, gap_params, weights2)

        for g in eachindex(grids)
            surv_dens[g] += pdf(surv_model, grids[g]) / n_re
            surv_surv[g] += ccdf(surv_model, grids[g]) / n_re
            gap_dens[g] += pdf(gap_model, grids[g]) / n_re
            gap_surv[g] += ccdf(gap_model, grids[g]) / n_re
        end
    end

    return [surv_dens, surv_surv, gap_dens, gap_surv]
end



# 5
function dpm_density_dpre(DP_dict, grids, theta_list, phi_list, weights1, lambda_list, eta_list, weights2)

    n_re=50000

    re = sample_random_effects_dp(DP_dict["weights"], DP_dict["atoms"], n_re)
    epsilon = re[1,:]
    xi = re[2,:]

    ngrids = length(grids)

    surv_dens = zeros(ngrids)
    surv_surv = zeros(ngrids) 
    gap_dens = zeros(ngrids) 
    gap_surv = zeros(ngrids)

    @showprogress for i_re in 1:n_re 
        surv_params = tuple.(theta_list ./ epsilon[i_re], phi_list)
        surv_model = MixtureModel(LogLogistic, surv_params, weights1)

        gap_params = tuple.(lambda_list ./ xi[i_re], eta_list)
        gap_model = MixtureModel(LogLogistic, gap_params, weights2)

        for g in eachindex(grids)
            surv_dens[g] += pdf(surv_model, grids[g]) / n_re
            surv_surv[g] += ccdf(surv_model, grids[g]) / n_re
            gap_dens[g] += pdf(gap_model, grids[g]) / n_re
            gap_surv[g] += ccdf(gap_model, grids[g]) / n_re
        end
    end

    return [surv_dens, surv_surv, gap_dens, gap_surv]
end



# 1
function data_generator_fixed(Sigma_e, theta, phi, lambda, eta, n)
    
    re = rand(MvLogNormal(zeros(2), Sigma_e), n)
    epsilon = re[1,:]
    xi = re[2,:]

    survival = zeros(n)
    gap = [] 
    arrival = []

    @showprogress for i in 1:n 
        survival[i] = rand(LogLogistic(theta/epsilon[i], phi), 1)[1]
        
        gap_dist = LogLogistic(lambda/xi[i], eta)
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

    return [survival, gap, arrival]
end 



# 2
function dpm_data_generator_fixed(Sigma_e, theta, phi, w, lambda, eta, p, n)

    re = rand(MvLogNormal(zeros(2), Sigma_e), n)
    epsilon = re[1,:]
    xi = re[2,:]

    survival = zeros(n)
    gap = [] 
    arrival = []

    @showprogress for i in 1:n 
        surv_param = tuple.(theta ./ epsilon[i], phi)
        surv_dist = MixtureModel(LogLogistic, surv_param, w)
        survival[i] = rand(surv_dist, 1)[1]
        
        gap_param = tuple.(lambda ./ xi[i], eta)
        gap_dist = MixtureModel(LogLogistic, gap_param, p) 

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

    return [survival, gap, arrival]
end 



# 3
function shared_data_generator_fixed(sigma2_e, power, theta, phi, lambda, eta, n)
    
    xi = rand(LogNormal(0, sqrt(sigma2_e)), n)
    epsilon = xi.^power 

    survival = zeros(n)
    gap = [] 
    arrival = []

    @showprogress for i in 1:n 
        survival[i] = rand(LogLogistic(theta/epsilon[i], phi), 1)[1]
        
        gap_dist = LogLogistic(lambda/xi[i], eta)
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

    return [survival, gap, arrival]
end 



# 4
function shared_dpm_data_generator_fixed(sigma2_e, power, theta, phi, w, lambda, eta, p, n)

    xi = rand(LogNormal(0, sqrt(sigma2_e)), n)
    epsilon = xi.^power 

    survival = zeros(n)
    gap = [] 
    arrival = []

    @showprogress for i in 1:n 
        surv_param = tuple.(theta ./ epsilon[i], phi)
        surv_dist = MixtureModel(LogLogistic, surv_param, w)
        survival[i] = rand(surv_dist, 1)[1]
        
        gap_param = tuple.(lambda ./ xi[i], eta)
        gap_dist = MixtureModel(LogLogistic, gap_param, p) 

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

    return [survival, gap, arrival]
end 



# 5
function dpm_data_generator_dpre(DP_dict, theta, phi, w, lambda, eta, p, n)

    re = sample_random_effects_dp(DP_dict["weights"], DP_dict["atoms"], n)
    epsilon = re[1,:]
    xi = re[2,:]

    survival = zeros(n)
    gap = [] 
    arrival = []

    @showprogress for i in 1:n 
        surv_param = tuple.(theta ./ epsilon[i], phi)
        surv_dist = MixtureModel(LogLogistic, surv_param, w)
        survival[i] = rand(surv_dist, 1)[1]
        
        gap_param = tuple.(lambda ./ xi[i], eta)
        gap_dist = MixtureModel(LogLogistic, gap_param, p) 

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

    return [survival, gap, arrival]
end 