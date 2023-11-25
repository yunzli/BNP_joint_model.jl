function update_zeta(dat, cur, hyper)

	N = sum(dat["Nvec"])
	a_zeta = hyper["a_zeta"]
	b_zeta = hyper["b_zeta"]

	g = cur["g"]
	zeta = cur["zeta"]
	zeta_eta = rand(Beta(zeta+1, N),1)[1]

	weight = (a_zeta + g - 1) / (N * (1 / b_zeta - log(zeta_eta)) + a_zeta + g - 1) 

	u = rand(Uniform(0,1), 1)[1] 
	if u < weight
		zeta = rand(Gamma(a_zeta + g, 1 / (1 / b_zeta - log(zeta_eta))), 1)[1]
	else
		zeta = rand(Gamma(a_zeta + g - 1, 1 / (1 / b_zeta - log(zeta_eta))), 1)[1]
	end

	return zeta 

end
