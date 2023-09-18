using Pkg
Pkg.activate(".")

using Distributions 
using Random 
import BNPJointModel

Random.seed!(10086)

# first set of hyperparameter generation 
dists1 = Dict()
dists1["alpha"] = 5.0
dists1["mu_s"] = 2.0 
dists1["sigma_s"] = 0.1 
dists1["a_s"] = 20.0
dists1["b_s"] = 3.0
dists1["mu_r"] = 0.5 
dists1["sigma_r"] = 0.5
dists1["a_r"] = 20.0
dists1["b_r"] = 3.0

res1 = BNPJointModel.prior_generator(dists1, 50)
BNPJointModel.data_visualizer(res1, "test1")

# second set of hyperparameter generation 
dists2 = Dict()
dists2["alpha"] = 5.0
dists2["mu_s"] = 2.0 
dists2["sigma_s"] = 0.1 
dists2["a_s"] = 20.0
dists2["b_s"] = 3.0
dists2["mu_r"] = 2.0 
dists2["sigma_r"] = 0.1
dists2["a_r"] = 20.0
dists2["b_r"] = 3.0

res2 = BNPJointModel.prior_generator(dists2, 50)
BNPJointModel.data_visualizer(res2, "test2")

# third set of hyperparameter generation 
dists3 = Dict()
dists3["alpha"] = 5.0
dists3["mu_s"] = 0.5
dists3["sigma_s"] = 0.5
dists3["a_s"] = 20.0
dists3["b_s"] = 3.0
dists3["mu_r"] = 2.0 
dists3["sigma_r"] = 0.1
dists3["a_r"] = 20.0
dists3["b_r"] = 3.0

res3 = BNPJointModel.prior_generator(dists3, 50)
BNPJointModel.data_visualizer(res3, "test3")