using Pkg
Pkg.activate(".")

using Distributions 
import BNPJointModel

dists = Dict()
dists["alpha"] = 5.0
dists["mu_s"] = 2.0 
dists["sigma_s"] = 0.1 
dists["a_s"] = 5.0
dists["b_s"] = 1.0
dists["mu_r"] = 2.0 
dists["sigma_r"] = 0.1 
dists["a_r"] = 5.0
dists["b_r"] = 1.0

res = BNPJointModel.prior_generator(dists, 20)

BNPJointModel.data_visualizer(res) 