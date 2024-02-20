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

"""
This MCMC is for LDDP-based joint model with treatment-specific random effects.
"""

package_dir = "//Users/yunzheli/Packages/BNPJointModel/"
include(package_dir*"src/utils.jl")
include(package_dir*"src/loglogistic.jl")
include(package_dir*"src/cov_re/update_L.jl")
include(package_dir*"src/cov_re/update_surv_mixing.jl") 
include(package_dir*"src/cov_re/update_surv_hyper.jl") 
include(package_dir*"src/cov_re/update_alpha.jl") 
include(package_dir*"src/cov_re/update_U.jl")
include(package_dir*"src/cov_re/update_gap_mixing.jl") 
include(package_dir*"src/cov_re/update_gap_hyper.jl") 
include(package_dir*"src/cov_re/update_zeta.jl") 
include(package_dir*"src/cov_re/update_re.jl")


function MCMC(config_file)
	"""
	config_file: a TOML configuration file in the folder ./configs
	""" 
	
	config = TOML.parsefile(config_file)

	nsam = config["nsam"]
	dat = load(config["data_file"])
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

    n = dat["n"]
    N = dat["N"]
    p = dat["p"]
    q = dat["q"]

	cur = Dict(
		# initialization of all parameters 
		"L" => ones(Int64, n), 
		"phi" => [1], 
		"theta" => [1],
        "beta" => zeros(1,p),
        "k" => 1, 
        "nl" => [n], 
		"mu_theta" => 1, 
        "mu_beta" => zeros(p),
		"b_phi" => 1,
		"alpha" => 1,

        "tU" => ones(Int64, N+n),
        "eta" => [1],
        "lambda" => [1], 
        "gamma" => zeros(1,q), 
        "g" => 1, 
        "ml" => [N+n],
        "mu_lambda" => 1,
        "mu_gamma" => zeros(q),
        "b_eta" => 1,
        "zeta" => 1,

        "epsilon" => ones(n),
        "xi" => ones(n),
        "Sigma_e_1" => Matrix(Diagonal(ones(2))),
        "Sigma_e_2" => Matrix(Diagonal(ones(2)))
		)

    # posterior values
    pos = Dict(
		"L" => [],
		"phi" => [], # Matrix{Float64}(undef, nsam, N),
		"theta" => [], # Array{Float64}(undef, nsam, N, J),
        "beta" => [],
        "k" => zeros(Int64, nsam), 
        "nl" => [], 

		"mu_theta" => zeros(nsam), 
        "mu_beta" => zeros(nsam, p), 
		"b_phi" => zeros(nsam), 
		"alpha" => zeros(nsam), 

        "tU" => [],
        "eta" => [],
        "lambda" => [],
        "gamma" => [], 
        "g" => zeros(Int64, nsam),
        "ml" => [],

        "mu_lambda" => zeros(nsam), 
        "mu_gamma" => zeros(nsam, q), 
        "b_eta" => zeros(nsam), 
        "zeta" => zeros(nsam),

        "epsilon" => zeros(nsam, n),
        "xi" => zeros(nsam, n),
        "Sigma_e_1" => zeros(nsam, 2, 2),
        "Sigma_e_2" => zeros(nsam, 2, 2)
	)

	@showprogress for i in (1:nsam)
		# Gibbs sampler 

        # survival 
        tmp_L = update_L(dat, cur, hyper)
		cur["L"] = tmp_L["L"]
		push!(pos["L"], cur["L"])
        cur["theta"] = tmp_L["theta"]
        cur["beta"] = tmp_L["beta"]
        cur["phi"] = tmp_L["phi"]
		cur["nl"] = tmp_L["nl"]
        push!(pos["nl"], cur["nl"])
        pos["k"][i] = cur["k"] = tmp_L["k"]
    

		cur["u"] = update_surv_pg_lantent(dat, cur) # temporary augmented parameter 
		cur["theta"] = update_theta(dat, cur, hyper)
        push!(pos["theta"], cur["theta"])
        cur["beta"] = update_beta(dat, cur, hyper)
        push!(pos["beta"], cur["beta"])
        cur["phi"] = update_phi(dat, cur, hyper)
		push!(pos["phi"], cur["phi"]) 

        pos["alpha"][i] = cur["alpha"] = update_alpha(dat, cur, hyper)
		pos["mu_theta"][i] = cur["mu_theta"] = update_mu_theta(cur, hyper)
        pos["mu_beta"][i,:] = cur["mu_beta"] = update_mu_beta(cur, hyper)
		pos["b_phi"][i] = cur["b_phi"] = update_b_phi(cur, hyper)  

        # gap times 
        tmp_U = update_U(dat, cur, hyper)
		cur["tU"] = tmp_U["tU"]
		push!(pos["tU"], cur["tU"])
        cur["lambda"] = tmp_U["lambda"]
        cur["gamma"] = tmp_U["gamma"]
        cur["eta"] = tmp_U["eta"]
		cur["ml"] = tmp_U["ml"]
        push!(pos["ml"], cur["ml"])
        pos["g"][i] = cur["g"] = tmp_U["g"]

		cur["varsigma"] = update_gap_pg_lantent(dat, cur) # temporary augmented parameter 
		cur["lambda"] = update_lambda(dat, cur, hyper)
        push!(pos["lambda"], cur["lambda"])
		cur["gamma"] = update_gamma(dat, cur, hyper)
        push!(pos["gamma"], cur["gamma"])
        cur["eta"] = update_eta(dat, cur, hyper)
		push!(pos["eta"], cur["eta"]) 

        pos["zeta"][i] = cur["zeta"] = update_zeta(dat, cur, hyper)
		pos["mu_lambda"][i] = cur["mu_lambda"] = update_mu_lambda(cur, hyper)
		pos["mu_gamma"][i,:] = cur["mu_gamma"] = update_mu_gamma(cur, hyper)
		pos["b_eta"][i] = cur["b_eta"] = update_b_eta(cur, hyper)  

        cur["epsilon"], cur["xi"] = update_random_effects(dat, cur, hyper)
        pos["epsilon"][i,:] = cur["epsilon"]
        pos["xi"][i,:] = cur["xi"]
        pos["Sigma_e_1"][i,:,:], pos["Sigma_e_2"][i,:,:] = cur["Sigma_e_1"], cur["Sigma_e_2"] = update_Sigma_e(dat, cur, hyper)
	end 

	result = Dict("pos" => pos,
				  "hyper" => hyper)

	savefile = config["save_path"] 
	save(savefile, result) 
	
end;

