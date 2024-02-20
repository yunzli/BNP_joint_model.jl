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
	dat = load(config["data_file"])
	hyper = config["hyper"]
    hyper["C_e"] =  mapreduce(permutedims, vcat, hyper["C_e"])

    n = dat["n"]
    N = dat["N"]

    if isfile(config["save_path"])
        println("Warm start")
        cur = load(config["save_path"])
        cur = cur["warm_start"]
    else
        println("Cold start")
        cur = Dict(
            # initialization of all parameters 
            "L" => ones(Int64, n), 
            "phi" => [1], 
            "theta" => [1],
            "k" => 1, 
            "nl" => [n], 
            "mu_theta" => 1, 
            "b_phi" => 1,
            "alpha" => 1,

            "tU" => ones(Int64, N+n),
            "eta" => [1],
            "lambda" => [1], 
            "g" => 1, 
            "ml" => [N+n],
            "mu_lambda" => 1,
            "b_eta" => 1,
            "zeta" => 1,

            "epsilon" => ones(n),
            "xi" => ones(n),
            "Sigma_e" => Matrix(Diagonal(ones(2)))
            )
    end

    
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
		"alpha" => zeros(nsam), 

        "tU" => [],
        "eta" => [],
        "lambda" => [],
        "g" => zeros(Int64, nsam),
        "ml" => [],

        "mu_lambda" => zeros(nsam), 
        "b_eta" => zeros(nsam), 
        "zeta" => zeros(nsam),

        "epsilon" => zeros(nsam, n),
        "xi" => zeros(nsam, n),
        "Sigma_e" => zeros(nsam, 2, 2)
	)

    Sig = []
    for _ in 1:n 
        push!(Sig, 0.01*[1.0 0.0; 0.0 1.0]) 
    end  
	# @showprogress for i in (1:nsam)
    @showprogress for n_b in 1:nbatch
		# Gibbs sampler 

        for i_t in 1:batch_size 

            i = (n_b-1)*batch_size + i_t 

            # survival 
            tmp_L = update_L(dat, cur, hyper)
            cur["L"] = tmp_L["L"]
            push!(pos["L"], cur["L"])
            cur["theta"] = tmp_L["theta"]
            cur["phi"] = tmp_L["phi"]
            cur["nl"] = tmp_L["nl"]
            push!(pos["nl"], cur["nl"])
            pos["k"][i] = cur["k"] = tmp_L["k"]
        

            cur["u"] = update_surv_pg_lantent(dat, cur) # temporary augmented parameter 
            cur["theta"] = update_theta(dat, cur, hyper)
            push!(pos["theta"], cur["theta"])
            cur["phi"] = update_phi(dat, cur, hyper)
            push!(pos["phi"], cur["phi"]) 

            pos["alpha"][i] = cur["alpha"] = update_alpha(dat, cur, hyper)
            pos["mu_theta"][i] = cur["mu_theta"] = update_mu_theta(cur, hyper)
            pos["b_phi"][i] = cur["b_phi"] = update_b_phi(cur, hyper)  

            # gap times 
            tmp_U = update_U(dat, cur, hyper)
            cur["tU"] = tmp_U["tU"]
            push!(pos["tU"], cur["tU"])
            cur["lambda"] = tmp_U["lambda"]
            cur["eta"] = tmp_U["eta"]
            cur["ml"] = tmp_U["ml"]
            push!(pos["ml"], cur["ml"])
            pos["g"][i] = cur["g"] = tmp_U["g"]

            cur["varsigma"] = update_gap_pg_lantent(dat, cur) # temporary augmented parameter 
            cur["lambda"] = update_lambda(dat, cur, hyper)
            push!(pos["lambda"], cur["lambda"])
            cur["eta"] = update_eta(dat, cur, hyper)
            push!(pos["eta"], cur["eta"]) 

            pos["zeta"][i] = cur["zeta"] = update_zeta(dat, cur, hyper)
            pos["mu_lambda"][i] = cur["mu_lambda"] = update_mu_lambda(cur, hyper)
            pos["b_eta"][i] = cur["b_eta"] = update_b_eta(cur, hyper)  

            cur["epsilon"], cur["xi"] = update_random_effects(dat, cur, hyper, Sig)
            # cur["epsilon"], cur["xi"] = update_random_effects_direct(dat, cur, hyper)
            pos["epsilon"][i,:] = cur["epsilon"]
            pos["xi"][i,:] = cur["xi"]
            pos["Sigma_e"][i,:,:] = cur["Sigma_e"] = update_Sigma_e(dat, cur, hyper)

        end

        for i in 1:n
            Sig[i] = cov(hcat(log.(pos["epsilon"][1:(n_b*batch_size),i]), log.(pos["xi"][1:(n_b*batch_size),i]))) + 1.0e-10*Diagonal(ones(2)) # adaptive MCMC for theta
        end

	end 

	result = Dict("pos" => pos,
				  "hyper" => hyper,
                  "warm_start" => cur)

	savefile = config["save_path"] 
	save(savefile, result) 
	
end;

