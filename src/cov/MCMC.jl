using JLD2
using RCall 
using Random
using TOML 
using StatsBase 
using LinearAlgebra
using Distributions 
using ProgressMeter
using ProximalOperators

Random.seed!(20231001)

include("../utils.jl")
include("../loglogistic.jl")
include("./update_L.jl")
include("./update_surv_mixing.jl") 
include("./update_surv_hyper.jl") 
include("./update_alpha.jl") 
include("./update_U.jl")
include("./update_gap_mixing.jl") 
include("./update_gap_hyper.jl") 
include("./update_zeta.jl") 
include("./update_re.jl")


function MCMC(config_file)
	"""
	config_file: a TOML configuration file in the folder ./configs
	""" 
	
	config = TOML.parsefile(config_file)

	datC = load(config["data_fileC"])
	datT = load(config["data_fileT"])

	hyper = config["hyper"]
    hyper["C_e"] =  mapreduce(permutedims, vcat, hyper["C_e"])
    # hyper["Sigma_beta"] = mapreduce(permutedims, vcat, hyper["Sigma_beta"])
    # hyper["Sigma_beta_inv"] = svd2inv(hyper["Sigma_beta"])
    # hyper["S_beta"] = mapreduce(permutedims, vcat, hyper["S_beta"])
    # hyper["S_beta_inv"] = svd2inv(hyper["S_beta"])
    # hyper["Sigma_gamma"] = mapreduce(permutedims, vcat, hyper["Sigma_gamma"])
    # hyper["Sigma_gamma_inv"] = svd2inv(hyper["Sigma_gamma"])
    # hyper["S_gamma"] = mapreduce(permutedims, vcat, hyper["S_gamma"])
    # hyper["S_gamma_inv"] = svd2inv(hyper["S_gamma"])

    nC = datC["n"]
    nT = datT["n"]
    NC = datC["N"]
    NT = datT["N"]

    # if !haskey(datC, "x")
    #     datC["x"] = zeros(nC, 1)
    #     datC["p"] = 1
    # end
    # if !haskey(datT, "x")
    #     datT["x"] = ones(nT, 1)
    # end
    # if !haskey(datC, "z")
    #     datC["z"] = zeros(nC, 1)
    #     datC["q"] = 1
    # end
    # if !haskey(datT, "z")
    #     datT["z"] = ones(nT, 1)
    # end

    # p, q = datC["p"], datC["q"]

	cur = Dict(
		# initialization of all parameters 
		"L" => [ones(Int64, nC), ones(Int64, nT)],
		"phi" => [1],
		"theta" => [1],
        # "beta" => zeros(1, p),
        "beta" => [0],
        "k" => 1,
        "nl" => [[nC],[nT]],
		"mu_theta" => 1,
        "mu_beta" => 0, # zeros(p),
		"b_phi" => 1,
		"alpha" => [1,1],

        "tU" => [ones(Int64, NC+nC), ones(Int64, NT+nT)],
        "eta" => [1],
        "lambda" => [1],
        # "gamma" => zeros(1,q),
        "gamma" => [0],
        "g" => 1,
        "ml" => [[NC+nC], [NT+nT]],
        "mu_lambda" => 1,
        # "mu_gamma" => zeros(q),
        "mu_gamma" => 0,
        "b_eta" => 1,
        "zeta" => [1,1],

        "epsilon" => [ones(nC), ones(nT)],
        "xi" => [ones(nC), ones(nT)],
        "Sigma_e_1" => Matrix(Diagonal(ones(2))),
        "Sigma_e_2" => Matrix(Diagonal(ones(2)))
		)

    
    batch_size = 50
    nbatch = config["nbatch"]
    nsam = nbatch * batch_size

    # posterior values
    pos = Dict(
		"L" => [],
		"phi" => [], # Matrix{Float64}(undef, nsam, N),
		"theta" => [], # Array{Float64}(undef, nsam, N, J),
        "beta" => [],
        "k" => zeros(Int64, nsam),
        "nl" => [],

		"mu_theta" => zeros(nsam),
        # "mu_beta" => zeros(nsam, p),
        "mu_beta" => zeros(nsam),
		"b_phi" => zeros(nsam),
		"alpha" => zeros(nsam, 2),

        "tU" => [],
        "eta" => [],
        "lambda" => [],
        "gamma" => [],
        "g" => zeros(Int64, nsam),
        "ml" => [],

        "mu_lambda" => zeros(nsam),
        # "mu_gamma" => zeros(nsam, q),
        "mu_gamma" => zeros(nsam),
        "b_eta" => zeros(nsam),
        "zeta" => zeros(nsam, 2),

        "epsilon" => [zeros(nsam, nC), zeros(nsam, nT)], # zeros(nsam, n),
        "xi" => [zeros(nsam, nC), zeros(nsam, nT)], # zeros(nsam, n),
        "Sigma_e_1" => zeros(nsam, 2, 2),
        "Sigma_e_2" => zeros(nsam, 2, 2)
	)

    SigC = []
    SigT = []
    for _ in 1:nC
        push!(SigC, 0.01*[1.0 0.0; 0.0 1.0])
    end
    for _ in 1:nT
        push!(SigT, 0.01*[1.0 0.0; 0.0 1.0])
    end
    Sig = [SigC, SigT]

    @showprogress for n_b in 1:nbatch
		# Gibbs sampler 

        for i_t in 1:batch_size

            i = (n_b-1)*batch_size + i_t

            # survival
            tmp_L = update_L(datC, datT, cur, hyper)
            cur["L"] = tmp_L["L"]
            push!(pos["L"], cur["L"])
            cur["theta"] = tmp_L["theta"]
            cur["beta"] = tmp_L["beta"]
            cur["phi"] = tmp_L["phi"]
            cur["nl"] = tmp_L["nl"]
            push!(pos["nl"], cur["nl"])
            pos["k"][i] = cur["k"] = tmp_L["k"]

            cur["u"] = update_surv_pg_lantent(datC, datT, cur) # temporary augmented parameter 
            cur["theta"] = update_theta(datC, datT, cur, hyper)
            push!(pos["theta"], cur["theta"])
            cur["beta"] = update_beta(datC, datT, cur, hyper)
            push!(pos["beta"], cur["beta"])
            cur["phi"] = update_phi(datC, datT, cur, hyper)
            push!(pos["phi"], cur["phi"])

            pos["alpha"][i,:] = cur["alpha"] = update_alpha(datC, datT, cur, hyper)
            pos["mu_theta"][i] = cur["mu_theta"] = update_mu_theta(cur, hyper)
            # pos["mu_beta"][i,:] = cur["mu_beta"] = update_mu_beta(cur, hyper)
            pos["mu_beta"][i] = cur["mu_beta"] = update_mu_beta(cur, hyper)
            pos["b_phi"][i] = cur["b_phi"] = update_b_phi(cur, hyper)

            # gap times 
            tmp_U = update_U(datC, datT, cur, hyper)
            cur["tU"] = tmp_U["tU"]
            push!(pos["tU"], cur["tU"])
            cur["lambda"] = tmp_U["lambda"]
            cur["gamma"] = tmp_U["gamma"]
            cur["eta"] = tmp_U["eta"]
            cur["ml"] = tmp_U["ml"]
            push!(pos["ml"], cur["ml"])
            pos["g"][i] = cur["g"] = tmp_U["g"]

            cur["varsigma"] = update_gap_pg_lantent(datC, datT, cur) # temporary augmented parameter 
            cur["lambda"] = update_lambda(datC, datT, cur, hyper)
            push!(pos["lambda"], cur["lambda"])
            cur["gamma"] = update_gamma(datC, datT, cur, hyper)
            push!(pos["gamma"], cur["gamma"])
            cur["eta"] = update_eta(datC, datT, cur, hyper)
            push!(pos["eta"], cur["eta"])

            pos["zeta"][i,:] = cur["zeta"] = update_zeta(datC, datT, cur, hyper)
            pos["mu_lambda"][i] = cur["mu_lambda"] = update_mu_lambda(cur, hyper)
            # pos["mu_gamma"][i,:] = cur["mu_gamma"] = update_mu_gamma(cur, hyper)
            pos["mu_gamma"][i] = cur["mu_gamma"] = update_mu_gamma(cur, hyper)
            pos["b_eta"][i] = cur["b_eta"] = update_b_eta(cur, hyper)

            cur["epsilon"], cur["xi"] = update_random_effects(datC, datT, cur, hyper, Sig)
            pos["epsilon"][1][i,:] = cur["epsilon"][1]
            pos["epsilon"][2][i,:] = cur["epsilon"][2]
            pos["xi"][1][i,:] = cur["xi"][1]
            pos["xi"][2][i,:] = cur["xi"][2]
            pos["Sigma_e_1"][i,:,:], pos["Sigma_e_2"][i,:,:] = cur["Sigma_e_1"], cur["Sigma_e_2"] = update_Sigma_e(datC, datT, cur, hyper)

            # println(cur["k"], " ", cur["g"])
        end

        for i in 1:nC
            Sig[1][i] = cov(hcat(log.(pos["epsilon"][1][1:(n_b*batch_size),i]), log.(pos["xi"][1][1:(n_b*batch_size),i]))) + 1.0e-10*Diagonal(ones(2)) # adaptive MCMC for theta
        end
        for i in 1:nT
            Sig[2][i] = cov(hcat(log.(pos["epsilon"][2][1:(n_b*batch_size),i]), log.(pos["xi"][2][1:(n_b*batch_size),i]))) + 1.0e-10*Diagonal(ones(2)) # adaptive MCMC for theta
        end

	end 

	result = Dict("pos" => pos,
				  "hyper" => hyper)

	savefile = config["save_path"]
	save(savefile, result) 
	
end;