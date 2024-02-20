function update_omega(cur, hyper)

    BH = hyper["BH"]

    tUC = cur["tUC"]
    tUT = cur["tUT"]
    zetaC = cur["zetaC"]
    zetaT = cur["zetaT"]

    piC = zeros(BH-1)
    piT = zeros(BH-1)

    logomegaC = zeros(BH)
    logomegaT = zeros(BH)

    mlC = zeros(Int64, BH)
    mlT = zeros(Int64, BH)

    for l in 1:BH

        mlC[l] = length(findall(tUC .== l))
        mlT[l] = length(findall(tUT .== l))

    end

    piC[1] = rand(Beta(1+mlC[1], zetaC+sum(mlC[2:end])), 1)[1]
    logomegaC[1] = log(piC[1])

    piT[1] = rand(Beta(1+mlT[1], zetaT+sum(mlT[2:end])), 1)[1]
    logomegaT[1] = log(piT[1])

    for l in 2:(BH-1)
        piC[l] = rand(Beta(1+mlC[l], zetaC+sum(mlC[(l+1):end])), 1)[1]
        logomegaC[l] = log(piC[l]) + sum(log.(1 .- piC[1:(l-1)]))

        piT[l] = rand(Beta(1+mlT[l], zetaT+sum(mlT[(l+1):end])), 1)[1]
        logomegaT[l] = log(piT[l]) + sum(log.(1 .- piT[1:(l-1)]))
    end

    logomegaC[end] =  sum(log.(1 .- piC))
    logomegaT[end] =  sum(log.(1 .- piT))

    for l in 1:BH
        if logomegaC[l] == -Inf
            logomegaC[l] = log(eps(0.0))
        end 
        if logomegaT[l] == -Inf
            logomegaT[l] = log(eps(0.0))
        end 
    end

	@assert sum(exp.(logomegaC)) ≈ 1.0
	@assert sum(exp.(logomegaT)) ≈ 1.0

	return Dict("logomegaC" => logomegaC, "logomegaT" => logomegaT)
end