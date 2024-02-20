function update_L(datC, datT, cur, hyper)

    survivalC = datC["survival"]
    nuC = datC["nu"]
    nC = datC["n"] 

    survivalT = datT["survival"]
    nuT = datT["nu"]
    nT = datT["n"]

    BG = hyper["BG"]

    theta = cur["theta"]
    phi = cur["phi"]

    epsilonC = cur["epsilonC"]
    epsilonT = cur["epsilonT"]
    logpC = cur["logpC"]
    logpT = cur["logpT"]

    LC = zeros(Int64, nC)
    LT = zeros(Int64, nT)

    for i in 1:nC
        logLprob = zeros(BG)

        for l in 1:BG

            dist = LogLogistic(theta[l] / epsilonC[i], phi[l])

            if nuC[i] == 1
                logLprob[l] = logpC[l] + logpdf(dist, survivalC[i])
            else
                logLprob[l] = logpC[l] + logccdf(dist, survivalC[i])
            end

            if logLprob[l] == -Inf
                logLprob[l] = log(eps(0.0))
            end
        end 

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )

        LC[i] = sample([1:1:BG;], Weights(Lprob))
    end 

    for i in 1:nT
        logLprob = zeros(BG)

        for l in 1:BG

            dist = LogLogistic(theta[l] / epsilonT[i], phi[l])

            if nuT[i] == 1
                logLprob[l] = logpT[l] + logpdf(dist, survivalT[i])
            else
                logLprob[l] = logpT[l] + logccdf(dist, survivalT[i])
            end

            if logLprob[l] == -Inf
                logLprob[l] = log(eps(0.0))
            end
        end 

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )

        LT[i] = sample([1:1:BG;], Weights(Lprob))
    end 

    return Dict("LC" => LC, "LT" => LT)
end 