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

	# nsam = config["nsam"]
	datC = load(config["data_fileC"])
	datT = load(config["data_fileT"])
	hyper = config["hyper"]
    hyper["C_e"] =  mapreduce(permutedims, vcat, hyper["C_e"])

    nC = datC["n"]
    nT = datT["n"]
    NC = datC["N"]
    NT = datT["N"]

	cur = Dict(
		# initialization of all parameters 
		"L" => [ones(Int64, nC), ones(Int64, nT)],
		"phi" => [1],
		"theta" => [1],
        "k" => 1,
        "nl" => [[nC],[nT]], 
		"mu_theta" => 1,
		"b_phi" => 1,
		"alpha" => [1,1],

        "tU" => [ones(Int64, NC+nC), ones(Int64, NT+nT)],
        "eta" => [1],
        "lambda" => [1],
        "g" => 1,
        "ml" => [[NC+nC], [NT+nT]],
        "mu_lambda" => 1,
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
        "k" => zeros(Int64, nsam), 
        "nl" => [], 

		"mu_theta" => zeros(nsam), 
		"b_phi" => zeros(nsam), 
		"alpha" => zeros(nsam, 2), 

        "tU" => [],
        "eta" => [],
        "lambda" => [],
        "g" => zeros(Int64, nsam),
        "ml" => [],

        "mu_lambda" => zeros(nsam), 
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
            cur["phi"] = tmp_L["phi"]
            cur["nl"] = tmp_L["nl"]
            push!(pos["nl"], cur["nl"])
            pos["k"][i] = cur["k"] = tmp_L["k"]

            cur["u"] = update_surv_pg_lantent(datC, datT, cur) # temporary augmented parameter 
            cur["theta"] = update_theta(datC, datT, cur, hyper)
            push!(pos["theta"], cur["theta"])
            cur["phi"] = update_phi(datC, datT, cur, hyper)
            push!(pos["phi"], cur["phi"]) 

            pos["alpha"][i,:] = cur["alpha"] = update_alpha(datC, datT, cur, hyper)
            pos["mu_theta"][i] = cur["mu_theta"] = update_mu_theta(cur, hyper)
            pos["b_phi"][i] = cur["b_phi"] = update_b_phi(cur, hyper)  

            # # gap times 
            tmp_U = update_U(datC, datT, cur, hyper)
            cur["tU"] = tmp_U["tU"]
            push!(pos["tU"], cur["tU"])
            cur["lambda"] = tmp_U["lambda"]
            cur["eta"] = tmp_U["eta"]
            cur["ml"] = tmp_U["ml"]
            push!(pos["ml"], cur["ml"])
            pos["g"][i] = cur["g"] = tmp_U["g"]

            cur["varsigma"] = update_gap_pg_lantent(datC, datT, cur) # temporary augmented parameter 
            cur["lambda"] = update_lambda(datC, datT, cur, hyper)
            push!(pos["lambda"], cur["lambda"])
            cur["eta"] = update_eta(datC, datT, cur, hyper)
            push!(pos["eta"], cur["eta"]) 

            pos["zeta"][i,:] = cur["zeta"] = update_zeta(datC, datT, cur, hyper)
            pos["mu_lambda"][i] = cur["mu_lambda"] = update_mu_lambda(cur, hyper)
            pos["b_eta"][i] = cur["b_eta"] = update_b_eta(cur, hyper)  

            cur["epsilon"], cur["xi"] = update_random_effects(datC, datT, cur, hyper, Sig)
            pos["epsilon"][1][i,:] = cur["epsilon"][1]
            pos["epsilon"][2][i,:] = cur["epsilon"][2]
            pos["xi"][1][i,:] = cur["xi"][1]
            pos["xi"][2][i,:] = cur["xi"][2]
            pos["Sigma_e_1"][i,:,:], pos["Sigma_e_2"][i,:,:] = cur["Sigma_e_1"], cur["Sigma_e_2"] = update_Sigma_e(datC, datT, cur, hyper)

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

