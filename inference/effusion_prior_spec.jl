using TOML
using RCall 
using JLD2 
using Distributions 
using ProgressMeter
using LinearAlgebra
using StatsBase
using Random 

Random.seed!(100)

dataC = load("//Users/yunzheli/Research/BNPJoint/data/effusionC.jld2")
gapC = dataC["gap"]
survivalC = dataC["survival"]
arrivalC = dataC["arrival"]
nuC = dataC["nu"]

dataT = load("//Users/yunzheli/Research/BNPJoint/data/effusionT.jld2")
gapT = dataT["gap"]
survivalT = dataT["survival"]
arrivalT = dataT["arrival"]
nuT = dataT["nu"]

@rput gapC survivalC nuC 
@rput gapT survivalT nuT 
R"""

var_survC = var(survivalC[which(nuC==1)])
var_survT = var(survivalT[which(nuT==1)])

var_gapC = var(unlist(gapC))
var_gapT = var(unlist(gapT))

calc_var = function(x, factor){
    # getting variance for random effects y
    # y ~ LN(0, s2)
    # var(y) = (exp(s2)-1)exp(s2)
    # let var(y) = x/10
    # x/10 = (exp(s2)-1)exp(s2)
    # let z = exp(s2)
    # x/10 = (z-1)z 
    # x/10 = z^2 - z
    # z^2 - z - x/10 = 0 and z > 0
    # z = -b + sqrt(b^2 - 4ac) / 2a
    # or 
    # z = -b - sqrt(b^2 - 4ac) / 2a (not applicable)
    # a = 1, b = -1, c = -x/10 
    # s2 = log(z)
    tmp = (1 + sqrt(1 + 4*x/factor)) / 2
    return(log(tmp))
}

spec_factor = 30
surv_specC = calc_var(var_survC, spec_factor)
gap_specC = calc_var(var_gapC, spec_factor)
surv_specT = calc_var(var_survT, spec_factor)
gap_specT = calc_var(var_gapT, spec_factor)

print(surv_specC)
print(gap_specC)
print(surv_specT)
print(gap_specT)
"""

@rget surv_specC gap_specC surv_specT gap_specT  

c_e = 13
C_eC = (c_e - 3) * [[surv_specC, 0], [0, gap_specC]]
C_eC = mapreduce(permutedims, vcat, C_eC)
C_eT = (c_e - 3) * [[surv_specT, 0], [0, gap_specT]]
C_eT = mapreduce(permutedims, vcat, C_eT)

print(C_eC)
print(C_eT)


nsim = 5000
epsilon_priorC = zeros(nsim)
xi_priorC = zeros(nsim)
for i in 1:nsim 
    Sigma_e_priorC = rand(InverseWishart(c_e, C_eC), 1)[1]
    re = rand(MvLogNormal(zeros(2), Sigma_e_priorC), 1)
    epsilon_priorC[i] = re[1]
    xi_priorC[i] = re[2]
end 

epsilon_priorT = zeros(nsim)
xi_priorT = zeros(nsim)
for i in 1:nsim 
    Sigma_e_priorT = rand(InverseWishart(c_e, C_eT), 1)[1]
    re = rand(MvLogNormal(zeros(2), Sigma_e_priorT), 1)
    epsilon_priorT[i] = re[1]
    xi_priorT[i] = re[2]
end 

@rput epsilon_priorC xi_priorC epsilon_priorT xi_priorT
R"""
par(mfrow=c(2,2))
hist(epsilon_priorC)
hist(xi_priorC)
hist(epsilon_priorT)
hist(xi_priorT)

hist(log(epsilon_priorC))
hist(log(xi_priorC))
hist(log(epsilon_priorT))
hist(log(xi_priorT))
"""
