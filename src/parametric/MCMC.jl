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
include("//Users/yunzheli/Packages/BNPJointModel/src/parametric/update_surv_param.jl")
include("//Users/yunzheli/Packages/BNPJointModel/src/parametric/update_gap_param.jl")
include("//Users/yunzheli/Packages/BNPJointModel/src/parametric/update_re.jl")

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
            "phi" => 1, 
            "theta" => 1,
            "eta" => 1,
            "lambda" => 1, 
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
		"phi" => zeros(nsam), 
		"theta" => zeros(nsam), # Array{Float64}(undef, nsam, N, J),

        "eta" => zeros(nsam),
        "lambda" => zeros(nsam),

        "epsilon" => zeros(nsam, n),
        "xi" => zeros(nsam, n),
        "Sigma_e" => zeros(nsam, 2, 2)
	)

    Sig = []
    for _ in 1:n 
        push!(Sig, 0.01*[1.0 0.0; 0.0 1.0]) 
    end  

    l_rate = -1 .* ones(2)
	sig = 0.01 .* ones(2)

    @showprogress for n_b in 1:nbatch
		# Gibbs sampler 
		count = zeros(2)
        for i_t in 1:batch_size 

            i = (n_b-1)*batch_size + i_t 

            # survival
            pos["theta"][i], _ = cur["theta"], tmp1 = update_theta(dat, cur, hyper, sig[1])
            count[1] += tmp1

            pos["phi"][i], _ = cur["phi"], tmp2 = update_phi(dat, cur, hyper, sig[2])
            count[2] += tmp2

            # gap times
            pos["lambda"][i] = cur["lambda"] = update_lambda(dat, cur, hyper)

            pos["eta"][i] = cur["eta"] = update_eta(dat, cur, hyper)

            # random effects
            cur["epsilon"], cur["xi"] = update_random_effects(dat, cur, hyper, Sig)
            pos["epsilon"][i,:] = cur["epsilon"]
            pos["xi"][i,:] = cur["xi"]
            pos["Sigma_e"][i,:,:] = cur["Sigma_e"] = update_Sigma_e(dat, cur, hyper)
        end

		acc = count ./ batch_size
		delta_n = minimum([0.01, n_b^(-1/2)])
		for i in 1:2
			if acc[i] < 0.44
				l_rate[i] -= delta_n
			else
				l_rate[i] += delta_n
			end
			sig[i] = exp(2 * l_rate[i])
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