function update_alpha(dat, cur, hyper)

	n = length(dat["survival"])
	a_alpha = hyper["a_alpha"]
	b_alpha = hyper["b_alpha"]

	k = cur["k"]
	alpha = cur["alpha"]
	alpha_eta = rand(Beta(alpha+1, n),1)[1]

	weight = (a_alpha + k - 1) / (n * (1 / b_alpha - log(alpha_eta)) + a_alpha + k - 1) 

	u = rand(Uniform(0,1), 1)[1] 
	if u < weight
		alpha = rand(Gamma(a_alpha + k, 1 / (1 / b_alpha - log(alpha_eta))), 1)[1]
	else
		alpha = rand(Gamma(a_alpha + k - 1, 1 / (1 / b_alpha - log(alpha_eta))), 1)[1]
	end

	return alpha 

end
