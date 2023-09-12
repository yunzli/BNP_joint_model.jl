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
   θₛ = rand(baseline_dists["theta_s"], N)
   ϕₛ = rand(baseline_dists["phi_s"], N)
   θᵣ = rand(baseline_dists["theta_r"], N)
   ϕᵣ = rand(baseline_dists["phi_r"], N)

   return dp(weights, θₛ, ϕₛ, θᵣ, ϕᵣ)
end 

function prior_generator(hyperparams, n::Int64)

   α = hyperparams["alpha"]
   G = dp()
   a_s = hyperparams["theta_s"]
   b_s = hyperparams["phi_s"]
   theta_r = hyperparams["theta_r"]
   phi_r = hyperparams["phi_r"]

   survivals = rand(LogLogistic(theta_s, phi_s), n)
   gap_times = []
   arrivival_times = []
   for i in range(1,n)
      tmp_gap = [] 
      tmp_arrival = []
      while isempty(tmp_gap) || tmp_arrival[end] < survivals[i]
         gap = rand(LogLogistic(theta_r, phi_r), 1)[1]
         push!(tmp_gap, gap)
         push!(tmp_arrival, tmp_arrival[end]+tmp_gap[end])
      end
      push!(gap_times, tmp_gap)
      push!(arrivival_times, tmp_arrival)
   end

   return Dict("gaps_times" <= gap_times, 
               "arrivival_times" <= arrivival_times,
               "survival_times" <= survivals)
end