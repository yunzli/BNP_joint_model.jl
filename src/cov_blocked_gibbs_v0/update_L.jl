function update_L(datC, datT, cur, hyper)

    survivalC = datC["survival"]
    nuC = datC["nu"]
    xC = datC["x"]
    epsilonC = cur["epsilonC"]

    survivalT = datT["survival"]
    nuT = datT["nu"]
    xT = datT["x"]
    epsilonT = cur["epsilonT"]

    BG = hyper["BG"]

    theta = cur["theta"]
    beta = cur["beta"]
    phi = cur["phi"]

    logpC = cur["logpC"]
    logpT = cur["logpT"]

    nC = datC["n"] 
    LC = zeros(Int64, nC)

    nT = datT["n"]
    LT = zeros(Int64, nT)

    for i in 1:nC
        logLprob = zeros(BG)

        for l in 1:BG

            dist = LogLogistic(theta[l] * exp(xC[i,:]'*beta[l,:]) / epsilonC[i], phi[l])

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

            dist = LogLogistic(theta[l] * exp(xT[i,:]'*beta[l,:]) / epsilonT[i], phi[l])

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