function update_alpha(datC, datT, cur, hyper)

	nC = datC["n"]
	nT = datT["n"]
	a_alpha = hyper["a_alpha"]
	b_alpha = hyper["b_alpha"]

	nl = cur["nl"]
	nlC = nl[1]
	nlT = nl[2]
	kC = length(findall(nlC .> 0))
	kT = length(findall(nlT .> 0))

	alpha = cur["alpha"]
    alphaC = alpha[1]
	alphaT = alpha[2]

	alpha_etaC = rand(Beta(alphaC+1, nC),1)[1]
	weightC = (a_alpha + kC - 1) / (nC * (1 / b_alpha - log(alpha_etaC)) + a_alpha + kC - 1)
	u = rand(Uniform(0,1), 1)[1]
	if u < weightC
		alphaC = rand(Gamma(a_alpha + kC, 1 / (1 / b_alpha - log(alpha_etaC))), 1)[1]
	else
		alphaC = rand(Gamma(a_alpha + kC - 1, 1 / (1 / b_alpha - log(alpha_etaC))), 1)[1]
	end

	alpha_etaT = rand(Beta(alphaT+1, nT),1)[1]
	weightT = (a_alpha + kT - 1) / (nT * (1 / b_alpha - log(alpha_etaT)) + a_alpha + kT - 1) 
	u = rand(Uniform(0,1), 1)[1] 
	if u < weightT
		alphaT = rand(Gamma(a_alpha + kT, 1 / (1 / b_alpha - log(alpha_etaT))), 1)[1]
	else
		alphaT = rand(Gamma(a_alpha + kT - 1, 1 / (1 / b_alpha - log(alpha_etaT))), 1)[1]
	end

	alpha = [alphaC, alphaT]
	return alpha 
end