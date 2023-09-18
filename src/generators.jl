# generate survival times and recurrent events

using Distributions

struct DP
   # this is a customized DP type 
   weights::Vector{Float64}
   θₛ::Vector{Float64}
   ϕₛ::Vector{Float64}
   θᵣ::Vector{Float64}
   ϕᵣ::Vector{Float64}
end

function dp(weights, θₛ, ϕₛ, θᵣ, ϕᵣ)
   # this is a customized DP constructor 
   return DP(weights, θₛ, ϕₛ, θᵣ, ϕᵣ)
end

function dp_generator(alpha::Real, baseline_dists::Dict, N::Int64)
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
   θₛ = exp.(rand(baseline_dists["log_theta_s"], N))
   ϕₛ = sqrt.(rand(baseline_dists["phi2_s"], N))
   θᵣ = exp.(rand(baseline_dists["log_theta_r"], N))
   ϕᵣ = sqrt.(rand(baseline_dists["phi2_r"], N))

   return dp(weights, θₛ, ϕₛ, θᵣ, ϕᵣ)
end 

function prior_generator(hyperparams, n::Int64)
   """
   This function generates recurrent and survival data jointly
   from the model. 

   Args:
      hyperparams: dictionary of hyperparameters
      n: number of observations to generate
   """

   α = hyperparams["alpha"]
   μₛ = hyperparams["mu_s"]
   σₛ = hyperparams["sigma_s"]
   aₛ = hyperparams["a_s"]
   bₛ = hyperparams["b_s"]
   μᵣ = hyperparams["mu_r"]
   σᵣ = hyperparams["sigma_r"]
   aᵣ = hyperparams["a_r"]
   bᵣ = hyperparams["b_r"]

   base_dists = Dict() 
   base_dists["log_theta_s"] = Normal(μₛ, σₛ)
   base_dists["phi2_s"] = Gamma(aₛ, bₛ)
   base_dists["log_theta_r"] = Normal(μᵣ, σᵣ)
   base_dists["phi2_r"] = Gamma(aᵣ, bᵣ)

   G = dp_generator(α, base_dists, 40)

   surv_params = tuple.(G.θₛ, G.ϕₛ)
   survival_dist = MixtureModel(LogLogistic, surv_params, G.weights)

   recur_params = tuple.(G.θᵣ, G.ϕᵣ)
   recur_dist = MixtureModel(LogLogistic, recur_params, G.weights)

   survivals = rand(survival_dist, n)
   gap_times = []
   arrivival_times = []
   counts = zeros(Int64, n)
   for i in range(1,n)
      tmp_gap = [] 
      tmp_arrival = [0.0]
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
              "dp" => G)

   return res
end