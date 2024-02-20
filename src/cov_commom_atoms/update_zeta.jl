function update_zeta(datC, datT, cur, hyper)

	NC = datC["N"]
	NT = datT["N"]
	a_zeta = hyper["a_zeta"]
	b_zeta = hyper["b_zeta"]

	g = cur["g"]

	ml = cur["ml"]
	mlC = ml[1]
	mlT = ml[2]
	gC = length(findall(mlC .> 0))
	gT = length(findall(mlT .> 0))
	zeta = cur["zeta"]
	zetaC = zeta[1]
	zetaT = zeta[2]

	zeta_etaC = rand(Beta(zetaC+1, NC),1)[1]

	weightC = (a_zeta + gC - 1) / (NC * (1 / b_zeta - log(zeta_etaC)) + a_zeta + gC - 1) 

	u = rand(Uniform(0,1), 1)[1] 
	if u < weightC
		zetaC = rand(Gamma(a_zeta + gC, 1 / (1 / b_zeta - log(zeta_etaC))), 1)[1]
	else
		zetaC = rand(Gamma(a_zeta + gC - 1, 1 / (1 / b_zeta - log(zeta_etaC))), 1)[1]
	end

	zeta_etaT = rand(Beta(zetaT+1, NT),1)[1]

	weightT = (a_zeta + gT - 1) / (NT * (1 / b_zeta - log(zeta_etaT)) + a_zeta + gT - 1) 

	u = rand(Uniform(0,1), 1)[1] 
	if u < weightT
		zetaT = rand(Gamma(a_zeta + gT, 1 / (1 / b_zeta - log(zeta_etaT))), 1)[1]
	else
		zetaT = rand(Gamma(a_zeta + gT - 1, 1 / (1 / b_zeta - log(zeta_etaT))), 1)[1]
	end

	zeta = [zetaC, zetaT]
	return zeta 
end
