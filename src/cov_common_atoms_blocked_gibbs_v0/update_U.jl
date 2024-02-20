function update_U(datC, datT, cur, hyper)
    survivalC = datC["survival"]
    gapC = datC["gap"]
    nC = datC["n"]
    NC = datC["N"]
    NvecC = datC["Nvec"]

    survivalT = datT["survival"]
    gapT = datT["gap"]
    nT = datT["n"]
    NT = datT["N"]
    NvecT = datT["Nvec"]

    BH = hyper["BH"]

    lambda = cur["lambda"]
    eta = cur["eta"]

    logomegaC = cur["logomegaC"]
    logomegaT = cur["logomegaT"]

    xiC = cur["xiC"]
    xiT = cur["xiT"]

    tUC = zeros(Int64, nC+NC)
    tUT = zeros(Int64, nT+NT)

    for h in 1:(NC+nC)
        logLprob = zeros(BH)

        for l in 1:BH

            if h <= NC 
                i,j = h2ij(h, NvecC)
                obs = gapC[i][j]
                dist = LogLogistic(lambda[l] / xiC[i], eta[l])
                
                logLprob[l] = logomegaC[l] + logpdf(dist, obs)
            else 
                i = h - NC
                if NvecC[i] == 0
                    obs = survivalC[i]
                else
                    obs = survivalC[i] - sum(gapC[i])
                end
                dist = LogLogistic(lambda[l] / xiC[i], eta[l])

                logLprob[l] = logomegaC[l] + logccdf(dist, obs)
            end

            if logLprob[l] == -Inf
                logLprob[l] = log(eps(0.0))
            end
        end 

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )

        tUC[h] = sample([1:1:BH;], Weights(Lprob))
    end

    for h in 1:(NT+nT)
        logLprob = zeros(BH)

        for l in 1:BH
            
            if h <= NT 
                i,j = h2ij(h, NvecT)
                obs = gapT[i][j]
                dist = LogLogistic(lambda[l] / xiT[i], eta[l])

                logLprob[l] = logomegaT[l] + logpdf(dist, obs)
            else 
                i = h - NT
                if NvecT[i] == 0
                    obs = survivalT[i]
                else
                    obs = survivalT[i] - sum(gapT[i])
                end
                dist = LogLogistic(lambda[l] / xiT[i], eta[l])

                logLprob[l] = logomegaT[l] + logccdf(dist, obs)
            end
            
            if logLprob[l] == -Inf
                logLprob[l] = log(eps(0.0))
            end
        end 

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )

        tUT[h] = sample([1:1:BH;], Weights(Lprob))
    end

    return Dict("tUC" => tUC, "tUT" => tUT)
end 