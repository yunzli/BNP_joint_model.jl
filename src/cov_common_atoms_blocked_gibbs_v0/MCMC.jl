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

work_dir = "//Users/yunzheli/Packages/BNPJointModel/src/" 
model_dir = "cov_common_atoms_blocked_gibbs_v0/"
include(work_dir * "utils.jl")
include(work_dir * "loglogistic.jl")
include(work_dir * model_dir * "update_L.jl")
include(work_dir * model_dir * "update_p.jl")
include(work_dir * model_dir * "update_surv_mixing.jl") 
include(work_dir * model_dir * "update_surv_hyper.jl") 
include(work_dir * model_dir * "update_alpha.jl") 
include(work_dir * model_dir * "update_U.jl")
include(work_dir * model_dir * "update_omega.jl")
include(work_dir * model_dir * "update_gap_mixing.jl") 
include(work_dir * model_dir * "update_gap_hyper.jl") 
include(work_dir * model_dir * "update_zeta.jl") 
include(work_dir * model_dir * "update_re.jl")


function MCMC(config_file)
	"""
	config_file: a TOML configuration file in the folder ./configs
	""" 
	
	config = TOML.parsefile(config_file)

	datC = load(config["data_fileC"])
	datT = load(config["data_fileT"])

	hyper = config["hyper"]
    hyper["C_e"] =  mapreduce(permutedims, vcat, hyper["C_e"])

    nC = datC["n"]
    nT = datT["n"]
    NC = datC["N"]
    NT = datT["N"]

    BG, BH = hyper["BG"], hyper["BH"]

	cur = Dict(
		# initialization of all parameters 
		"LC" => sample(1:1:BG, nC), # ones(Int64, nC),
        "LT" => sample(1:1:BG, nC), # ones(Int64, nT),
		"theta" => ones(BG),
		"phi" => ones(BG),
		"mu_theta" => 1,
		"b_phi" => 1,
		"alphaC" => 1,
        "alphaT" => 1,
        "logpC" => log(1/BG) .* ones(BG),
        "logpT" => log(1/BG) .* ones(BG),

        "tUC" => ones(Int64, NC+nC),
        "tUT" => ones(Int64, NT+nT),
        "lambda" => ones(BH),
        "eta" => ones(BH),
        "mu_lambda" => 1,
        "b_eta" => 1,
        "zetaC" => 1,
        "zetaT" => 1,
        "logomegaC" => log(1/BH) .* ones(BH),
        "logomegaT" => log(1/BH) .* ones(BH),

        "epsilonC" => ones(nC),
        "epsilonT" => ones(nT),
        "xiC" => ones(nC),
        "xiT" => ones(nT),
        "Sigma_e_1" => Matrix(Diagonal(ones(2))),
        "Sigma_e_2" => Matrix(Diagonal(ones(2)))
		)

    
    batch_size = 50
    nbatch = config["nbatch"]
    nsam = nbatch * batch_size

    # posterior values
    pos = Dict(
		"LC" => zeros(Int64, nsam, nC),
		"LT" => zeros(Int64, nsam, nT),
		"phi" => zeros(nsam, BG), # Matrix{Float64}(undef, nsam, N),
		"theta" => zeros(nsam, BG), # Array{Float64}(undef, nsam, N, J),
		"mu_theta" => zeros(nsam),
		"b_phi" => zeros(nsam),
		"alphaC" => zeros(nsam),
        "alphaT" => zeros(nsam),
        "logpC" => zeros(nsam, BG),
        "logpT" => zeros(nsam, BG),

        "tUC" => zeros(Int64, nsam, NC+nC),
        "tUT" => zeros(Int64, nsam, NT+nT),
        "eta" => zeros(nsam, BH),
        "lambda" => zeros(nsam, BH),
        "mu_lambda" => zeros(nsam),
        "b_eta" => zeros(nsam),
        "zetaC" => zeros(nsam),
        "zetaT" => zeros(nsam),
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
            pos["alphaC"][i], pos["alphaT"][i] = cur["alphaC"], cur["alphaT"] = update_alpha(cur, hyper)

            tmp_u = update_surv_pg_lantent(datC, datT, cur) # temporary augmented parameter 
            cur["uC"] = tmp_u["uC"]
            cur["uT"] = tmp_u["uT"]
            pos["theta"][i,:] = cur["theta"] = update_theta(datC, datT, cur, hyper)
            pos["phi"][i,:] = cur["phi"] = update_phi(datC, datT, cur, hyper)

            pos["mu_theta"][i] = cur["mu_theta"] = update_mu_theta(cur, hyper)
            pos["b_phi"][i] = cur["b_phi"] = update_b_phi(cur, hyper)

            # gap times 
            tmp_U = update_U(datC, datT, cur, hyper)
            pos["tUC"][i,:] = cur["tUC"] = tmp_U["tUC"]
            pos["tUT"][i,:] = cur["tUT"] = tmp_U["tUT"]

            tmp_omega = update_omega(cur, hyper)
            pos["logomegaC"][i,:] = cur["logomegaC"] = tmp_omega["logomegaC"]
            pos["logomegaT"][i,:] = cur["logomegaT"] = tmp_omega["logomegaT"]
            pos["zetaC"][i], pos["zetaT"][i] = cur["zetaC"], cur["zetaT"] = update_zeta(cur, hyper)

            tmp_varsigma = update_gap_pg_lantent(datC, datT, cur) # temporary augmented parameter 
            cur["varsigmaC"] = tmp_varsigma["varsigmaC"]
            cur["varsigmaT"] = tmp_varsigma["varsigmaT"]
            pos["lambda"][i,:] = cur["lambda"] = update_lambda(datC, datT, cur, hyper)
            pos["eta"][i,:] = cur["eta"] = update_eta(datC, datT, cur, hyper)

            pos["mu_lambda"][i] = cur["mu_lambda"] = update_mu_lambda(cur, hyper)
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
				  "hyper" => hyper)

	savefile = config["save_path"]
	save(savefile, result) 
	
end;