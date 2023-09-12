using Distributions 
import BNPJointModel

alpha = 1
N = 50 

dists = Dict()
dists["theta_s"] = Gamma(10,1)
dists["phi_s"] = Gamma(10,1)
dists["theta_r"] = Gamma(10,1)
dists["phi_r"] = Gamma(10,1)

res = BNPJointModel.dp_generator(alpha, dists, N)