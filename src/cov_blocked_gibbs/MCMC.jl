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

include("//Users/yunzheli/Packages/BNPJointModel/src/utils.jl")
include("//Users/yunzheli/Packages/BNPJointModel/src/loglogistic.jl")
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_L.jl")
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_p.jl")
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_surv_mixing.jl") 
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_surv_hyper.jl") 
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_alpha.jl") 
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_U.jl")
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_omega.jl")
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_gap_mixing.jl") 
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_gap_hyper.jl") 
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_zeta.jl") 
include("//Users/yunzheli/Packages/BNPJointModel/src/cov_blocked_gibbs/update_re.jl")


function MCMC(config_file)
	"""
	config_file: a TOML configuration file in the folder ./configs
	""" 
	
	config = TOML.parsefile(config_file)

	datC = load(config["data_fileC"])
	datT = load(config["data_fileT"])

	hyper = config["hyper"]
    hyper["C_e"] =  mapreduce(permutedims, vcat, hyper["C_e"])
    hyper["Sigma_beta"] = mapreduce(permutedims, vcat, hyper["Sigma_beta"])
    hyper["Sigma_beta_inv"] = svd2inv(hyper["Sigma_beta"])
    hyper["S_beta"] = mapreduce(permutedims, vcat, hyper["S_beta"])
    hyper["S_beta_inv"] = svd2inv(hyper["S_beta"])
    hyper["Sigma_gamma"] = mapreduce(permutedims, vcat, hyper["Sigma_gamma"])
    hyper["Sigma_gamma_inv"] = svd2inv(hyper["Sigma_gamma"])
    hyper["S_gamma"] = mapreduce(permutedims, vcat, hyper["S_gamma"])
    hyper["S_gamma_inv"] = svd2inv(hyper["S_gamma"])

    nC = datC["n"]
    nT = datT["n"]
    NC = datC["N"]
    NT = datT["N"]

    if !haskey(datC, "x")
        datC["x"] = zeros(nC, 1)
        datC["p"] = 1
    end
    if !haskey(datT, "x")
        datT["x"] = ones(nT, 1)
    end
    if !haskey(datC, "z")
        datC["z"] = zeros(nC, 1)
        datC["q"] = 1
    end
    if !haskey(datT, "z")
        datT["z"] = ones(nT, 1)
    end

    p, q = datC["p"], datC["q"]
    BG, BH = hyper["BG"], hyper["BH"]

    if isfile(config["save_path"])
        println("Warm start")
        cur = load(config["save_path"])
        cur = cur["warm_start"]
    else
        println("Cold start")
        cur = Dict(
            # initialization of all parameters 
            "LC" => sample(1:1:BG, nC), # ones(Int64, nC),
            "LT" => sample(1:1:BG, nC), # ones(Int64, nT),
            "phi" => ones(BG),
            "theta" => ones(BG),
            "beta" => zeros(BG, p),
            "mu_theta" => 1,
            "mu_beta" => zeros(p), 
            "b_phi" => 1,
            "alpha" => 3,
            "alpha0" => 0.9,
            "logpC" => log(1/BG) .* ones(BG),
            "logpT" => log(1/BG) .* ones(BG),
            "latent_surv" => ones(BG, 4) ./ 4,
            "uC" => ones(nC),
            "uT" => ones(nT),

            "tUC" => ones(Int64, NC+nC),
            "tUT" => ones(Int64, NT+nT),
            "eta" => ones(BH),
            "lambda" => ones(BH),
            "gamma" => zeros(BH, q),
            "mu_lambda" => 1,
            "mu_gamma" => zeros(q),
            "b_eta" => 1,
            "zeta" => 3,
            "zeta0" => 0.5,
            "logomegaC" => log(1/BH) .* ones(BH),
            "logomegaT" => log(1/BH) .* ones(BH),
            "latent_gap" => ones(BH, 4) ./ 4,

            "epsilonC" => ones(nC),
            "epsilonT" => ones(nT),
            "xiC" => ones(nC),
            "xiT" => ones(nT),
            "Sigma_e_1" => Matrix(Diagonal(ones(2))),
            "Sigma_e_2" => Matrix(Diagonal(ones(2)))
            )
    end

    
    batch_size = 50
    nbatch = config["nbatch"]
    nsam = nbatch * batch_size

    # posterior values
    pos = Dict(
		"LC" => zeros(Int64, nsam, nC),
		"LT" => zeros(Int64, nsam, nT),
		"phi" => zeros(nsam, BG), # Matrix{Float64}(undef, nsam, N),
		"theta" => zeros(nsam, BG), # Array{Float64}(undef, nsam, N, J),
        "beta" => zeros(nsam, BG, p),
		"mu_theta" => zeros(nsam),
        "mu_beta" => zeros(nsam, p),
		"b_phi" => zeros(nsam),
		"alpha" => zeros(nsam),
        "alpha0" => zeros(nsam),
        "logpC" => zeros(nsam, BG),
        "logpT" => zeros(nsam, BG),

        "tUC" => zeros(Int64, nsam, NC+nC),
        "tUT" => zeros(Int64, nsam, NT+nT),
        "eta" => zeros(nsam, BH),
        "lambda" => zeros(nsam, BH),
        "gamma" => zeros(nsam, BH, p),

        "mu_lambda" => zeros(nsam),
        "mu_gamma" => zeros(nsam, q),
        "b_eta" => zeros(nsam),
        "zeta" => zeros(nsam),
        "zeta0" => zeros(nsam),
        "logomegaC" => zeros(nsam, BH),
        "logomegaT" => zeros(nsam, BH),

        "epsilonC" => zeros(nsam, nC),
        "epsilonT" => zeros(nsam, nT), # zeros(nsam, n),
        "xiC" => zeros(nsam, nC),
        "xiT" => zeros(nsam, nT), # zeros(nsam, n),
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
            pos["LC"][i,:] = cur["LC"] = tmp_L["LC"]
            pos["LT"][i,:] = cur["LT"] = tmp_L["LT"]

            tmp_p = update_p(cur, hyper)
            pos["logpC"][i,:] = cur["logpC"] = tmp_p["logpC"]
            pos["logpT"][i,:] = cur["logpT"] = tmp_p["logpT"]
            cur["latent_surv"] = tmp_p["latent"]
            if haskey(hyper, "alpha")
                pos["alpha"][i] = cur["alpha"] = hyper["alpha"]
                pos["alpha0"][i] = cur["alpha0"] = update_alpha0(cur, hyper)
            else
                pos["alpha"][i], pos["alpha0"][i] = cur["alpha"], cur["alpha0"] = update_alpha(cur, hyper)
            end

            tmp_u = update_surv_pg_lantent(datC, datT, cur) # temporary augmented parameter 
            cur["uC"] = tmp_u["uC"]
            cur["uT"] = tmp_u["uT"]
            pos["theta"][i,:] = cur["theta"] = update_theta(datC, datT, cur, hyper)
            pos["beta"][i,:,:] = cur["beta"] = update_beta(datC, datT, cur, hyper)
            pos["phi"][i,:] = cur["phi"] = update_phi(datC, datT, cur, hyper)

            pos["mu_beta"][i,:] = cur["mu_beta"] = update_mu_beta(cur, hyper)
            pos["mu_theta"][i] = cur["mu_theta"] = update_mu_theta(cur, hyper)
            pos["b_phi"][i] = cur["b_phi"] = update_b_phi(cur, hyper)

            # gap times 
            tmp_U = update_U(datC, datT, cur, hyper)
            pos["tUC"][i,:] = cur["tUC"] = tmp_U["tUC"]
            pos["tUT"][i,:] = cur["tUT"] = tmp_U["tUT"]

            tmp_omega = update_omega(cur, hyper)
            pos["logomegaC"][i,:] = cur["logomegaC"] = tmp_omega["logomegaC"]
            pos["logomegaT"][i,:] = cur["logomegaT"] = tmp_omega["logomegaT"]
            cur["latent_gap"] = tmp_omega["latent"]
            if haskey(hyper, "zeta")
                pos["zeta"][i] = cur["zeta"] = hyper["zeta"]
                pos["zeta0"][i] = cur["zeta0"] = update_zeta0(cur, hyper)
            else
                pos["zeta"][i], pos["zeta0"][i] = cur["zeta"], cur["zeta0"] = update_zeta(cur, hyper)
            end

            tmp_varsigma = update_gap_pg_lantent(datC, datT, cur) # temporary augmented parameter 
            cur["varsigmaC"] = tmp_varsigma["varsigmaC"]
            cur["varsigmaT"] = tmp_varsigma["varsigmaT"]
            pos["lambda"][i,:] = cur["lambda"] = update_lambda(datC, datT, cur, hyper)
            pos["gamma"][i,:,:] = cur["gamma"] = update_gamma(datC, datT, cur, hyper)
            pos["eta"][i,:] = cur["eta"] = update_eta(datC, datT, cur, hyper)

            pos["mu_lambda"][i] = cur["mu_lambda"] = update_mu_lambda(cur, hyper)
            pos["mu_gamma"][i,:] = cur["mu_gamma"] = update_mu_gamma(cur, hyper)
            pos["b_eta"][i] = cur["b_eta"] = update_b_eta(cur, hyper)

            tmp_re = update_random_effects(datC, datT, cur, hyper, Sig)
            pos["epsilonC"][i,:] = cur["epsilonC"] = tmp_re["epsilonC"]
            pos["epsilonT"][i,:] = cur["epsilonT"] = tmp_re["epsilonT"]
            pos["xiC"][i,:] = cur["xiC"] = tmp_re["xiC"]
            pos["xiT"][i,:] = cur["xiT"] = tmp_re["xiT"]
            pos["Sigma_e_1"][i,:,:], pos["Sigma_e_2"][i,:,:] = cur["Sigma_e_1"], cur["Sigma_e_2"] = update_Sigma_e(datC, datT, cur, hyper)

        end

        for i in 1:nC
            Sig[1][i] = cov(hcat(log.(pos["epsilonC"][1:(n_b*batch_size),i]), log.(pos["xiC"][1:(n_b*batch_size),i]))) + 1.0e-10*Diagonal(ones(2)) # adaptive MCMC for theta
        end
        for i in 1:nT
            Sig[2][i] = cov(hcat(log.(pos["epsilonT"][1:(n_b*batch_size),i]), log.(pos["xiT"][1:(n_b*batch_size),i]))) + 1.0e-10*Diagonal(ones(2)) # adaptive MCMC for theta
        end

	end 

	result = Dict("pos" => pos,
				  "hyper" => hyper,
                  "warm_start" => cur)

	savefile = config["save_path"]
	save(savefile, result) 
	
end;