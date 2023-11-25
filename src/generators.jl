# generate survival times and recurrent events

using Distributions
using LinearAlgebra
using StatsBase

struct DP
   # this is a customized DP type 
   weights::Vector{Float64}
   atoms::Vector{Any}
end

function dp(weights, atoms)
   # this is a customized DP constructor 
   return DP(weights, atoms)
end

function dp_generator(alpha::Real, baseline_dists::Vector, N::Int64)
   """
   This function generates from a DP prior

   Args:
      baseline_dist: baseline distribution function 
      N: truncation level

   Output: 
      DP 
   """

   V = rand(Beta(1, alpha), N-1)
   logp = zeros(N) 
   logp[1] = log(V[1])
   for r in 2:(N-1)
       logp[r] = log(V[r]) + sum(log.(1 .- V[1:(r-1)]))
   end
   logp[N] = sum(log.(1 .- V[1:(N-1)]))
   weights = exp.(logp)
   atoms_list = [] 
   for baseline_dist in baseline_dists
       atoms = rand(baseline_dist, N)
       push!(atoms_list, atoms)
   end

   print(typeof(atoms_list))
   return dp(weights, atoms_list)
end 

function prior_generator(hyperparams, n::Int64)
   """
   This function generates recurrent and survival data jointly
   from the model. 

   Args:
      hyperparams: dictionary of hyperparameters
      n: number of observations to generate
   """

   αₛ = hyperparams["alpha_s"]
   μₛ = hyperparams["mu_s"]
   σₛ = hyperparams["sigma_s"]
   aₛ = hyperparams["a_s"]
   bₛ = hyperparams["b_s"]
   base_dists_surv = []
   push!(base_dists_surv, Normal(μₛ, σₛ))
   push!(base_dists_surv, Gamma(aₛ, bₛ))

   αᵣ = hyperparams["alpha_r"]
   μᵣ = hyperparams["mu_r"]
   σᵣ = hyperparams["sigma_r"]
   aᵣ = hyperparams["a_r"]
   bᵣ = hyperparams["b_r"]
   base_dists_gap = []
   push!(base_dists_gap, Normal(μᵣ, σᵣ))
   push!(base_dists_gap, Gamma(aᵣ, bᵣ))

   αₑ = hyperparams["alpha_e"]
   Σₑ = hyperparams["Sigma_e"]
   base_dists_re = [] 
   push!(base_dists_re, MvLogNormal([0,0], Σₑ))

   Gₛ = dp_generator(αₛ, base_dists_surv, 40)
   Gᵣ = dp_generator(αᵣ, base_dists_gap, 40)
   Gₑ = dp_generator(αₑ, base_dists_re, 40)

   re_params = tuple.(Gₑ.atoms[1][1,:], Gₑ.atoms[1][2,:]) 
   random_effects = sample(re_params, Weights(Gₑ.weights), n)
   
   survivals = zeros(n) # rand(survival_dist, n)
   gap_times = []
   arrivival_times = []
   counts = zeros(Int64, n)
   for i in range(1,n)
      tmp_gap = zeros(0)
      tmp_arrival = zeros(1)

      surv_params = tuple.(exp.(Gₛ.atoms[1].-random_effects[i][1]), sqrt.(Gₛ.atoms[2]))
      survival_dist = MixtureModel(LogLogistic, surv_params, Gₛ.weights)
      survivals[i] = rand(survival_dist, 1)[1]

      recur_params = tuple.(exp.(Gᵣ.atoms[1].-random_effects[i][2]), sqrt.(Gᵣ.atoms[2]))
      recur_dist = MixtureModel(LogLogistic, recur_params, Gᵣ.weights)

      while true 
         gap = rand(recur_dist, 1)[1]
         if tmp_arrival[end] + gap > survivals[i]
            break
         end

         push!(tmp_gap, gap)
         push!(tmp_arrival, tmp_arrival[end]+gap)
      end
      counts[i] = length(tmp_gap)
      push!(gap_times, tmp_gap)
      push!(arrivival_times, tmp_arrival[2:end])
   end

   res = Dict("counts" => counts,
              "gap_times" => gap_times, 
              "arrivival_times" => arrivival_times,
              "survival_times" => survivals,
              "dp_surv" => Gₛ, 
              "dp_gap" => Gᵣ, 
              "dp_re" => Gₑ)

   return res
end