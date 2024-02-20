function update_p(cur, hyper)

    BG = hyper["BG"]

    LC = cur["LC"]
    LT = cur["LT"]
    alphaC = cur["alphaC"]
    alphaT = cur["alphaT"]

    VC = zeros(BG-1)
    VT = zeros(BG-1)
    logpC = zeros(BG)
    logpT = zeros(BG)
    nlC = zeros(Int64, BG)
    nlT = zeros(Int64, BG)
    
    for l in 1:BG

        nlC[l] = length(findall(LC .== l))
        nlT[l] = length(findall(LT .== l))

    end 

    VC[1] = rand(Beta(1+nlC[1], alphaC+sum(nlC[2:end])), 1)[1]
    logpC[1] = log(VC[1])

    VT[1] = rand(Beta(1+nlT[1], alphaT+sum(nlT[2:end])), 1)[1]
    logpT[1] = log(VT[1])

    for l in 2:(BG-1)
        VC[l] = rand(Beta(1+nlC[l], alphaC+sum(nlC[(l+1):end])), 1)[1]
        logpC[l] = log(VC[l]) + sum(log.(1 .- VC[1:(l-1)]))

        VT[l] = rand(Beta(1+nlT[l], alphaT+sum(nlT[(l+1):end])), 1)[1]
        logpT[l] = log(VT[l]) + sum(log.(1 .- VT[1:(l-1)]))
    end

    logpC[end] = sum(log.(1 .- VC))
    logpT[end] = sum(log.(1 .- VT))

    for l in 1:BG
        if logpC[l] == -Inf
            logpC[l] = log(eps(0.0))
        end 
        if logpT[l] == -Inf
            logpT[l] = log(eps(0.0))
        end 
    end

	return Dict("logpC" => logpC, "logpT" => logpT)
end