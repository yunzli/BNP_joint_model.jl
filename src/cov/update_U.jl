function update_U(datC, datT, cur, hyper)

    tU = cur["tU"]
    tUC = tU[1]
    tUT = tU[2]

    lambda = cur["lambda"]
    gamma = cur["gamma"]
    eta = cur["eta"]
    ml = cur["ml"]
    mlC = ml[1]
    mlT = ml[2]

    xi = cur["xi"]
    xiC = xi[1]
    xiT = xi[2]

    g = cur["g"]

    survivalC = datC["survival"]
    gapC = datC["gap"]
    NvecC = datC["Nvec"] # N1, N2, ..., Nn
    NC = datC["N"]
    nC = datC["n"]

    survivalT = datT["survival"]
    gapT = datT["gap"]
    NvecT = datT["Nvec"] # N1, N2, ..., Nn
    NT = datT["N"]
    nT = datT["n"]

    gstar = hyper["gstar"]
    zeta = cur["zeta"]
    zetaC = zeta[1]
    zetaT = zeta[2]

    sigma2_lambda = hyper["sigma2_lambda"]
    mu_lambda = cur["mu_lambda"]
    sigma2_gamma = hyper["sigma2_gamma"]
    mu_gamma = cur["mu_gamma"]
    b_eta = cur["b_eta"]
    a_eta = hyper["a_eta"]

    for h in 1:(NC+nC)

        tUh = tUC[h]
        ml_h = mlC[tUh]

        _tUC = zeros(Int64, NC+nC)
        _tUT = zeros(Int64, NT+nT)

        if (ml_h == 1) & (mlT[tUh] == 0)
            # a singleton 

            _g = g - 1
            _mlC = zeros(Int64, _g)
            _mlT = zeros(Int64, _g)

            _eta = zeros(_g+gstar)
            _lambda = zeros(_g+gstar)
            _gamma = zeros(_g+gstar)

            for h_prime in 1:(NC+nC) 
                if h_prime != h
                    if tUC[h_prime] < tUh 
                        _tUC[h_prime] = tUC[h_prime]
                    elseif tUC[h_prime] > tUh
                        _tUC[h_prime] = tUC[h_prime] - 1
                    end
                end
            end

            for h_prime in 1:(NT+nT)
                if tUT[h_prime] < tUh 
                    _tUT[h_prime] = tUT[h_prime]
                elseif tUT[h_prime] > tUh
                    _tUT[h_prime] = tUT[h_prime] - 1
                end
            end

            for l in 1:_g
                if l < tUh 
                    _mlC[l] = mlC[l]
                    _mlT[l] = mlT[l]

                    _lambda[l] = lambda[l]
                    _gamma[l] = gamma[l]
                    _eta[l] = eta[l] 
                else
                    _mlC[l] = mlC[l+1]
                    _mlT[l] = mlT[l+1]

                    _lambda[l] = lambda[l+1]
                    _gamma[l] = gamma[l+1]
                    _eta[l] = eta[l+1] 
                end
            end
        else 
            # not a singleton 

            _g = g 
            _mlC = zeros(Int64, _g)
            _mlT = zeros(Int64, _g)

            _eta = zeros(_g+gstar)
            _lambda = zeros(_g+gstar) 
            _gamma = zeros(_g+gstar)

            for h_prime in 1:(NC+nC)
                if h_prime != h 
                    _tUC[h_prime] = tUC[h_prime]
                end
            end

            for h_prime in 1:(NT+nT)
                _tUT[h_prime] = tUT[h_prime]
            end

            for l in 1:_g

                _mlT[l] = mlT[l]
                if l == tUh 
                    _mlC[l] = mlC[l] - 1
                else
                    _mlC[l] = mlC[l]
                end

                _lambda[l] = lambda[l]
                _gamma[l] = gamma[l]
                _eta[l] = eta[l] 
            end
        end

        _indC = findall(_mlC .> 0)
        _gC = length(_indC)

        for l in (_g+1):(_g+gstar)
            _lambda[l] = exp(rand(Normal(mu_lambda, sqrt(sigma2_lambda)), 1)[1])
            _gamma[l] = rand(Normal(mu_gamma, sqrt(sigma2_gamma)), 1)[1]
            _eta[l] = sqrt(rand(Gamma(a_eta, b_eta), 1)[1])
        end

        logLprob = zeros(_g+gstar) 
        for l in 1:(_g+gstar) 
            # if l <= _g
            if l in _indC
                logLprob[l] = log(_mlC[l])
            else
                # logLprob[l] = log(zeta/gstar) 
                logLprob[l] = log(zetaC/(gstar + _g - _gC))
            end

            if h <= NC
                i,j = h2ij(h, NvecC)
                logLprob[l] += logpdf(LogLogistic(_lambda[l] / xiC[i], _eta[l]), gapC[i][j])
            else
                i = h - NC
                if NvecC[i] == 0 
                    logLprob[l] += logccdf(LogLogistic(_lambda[l] / xiC[i], _eta[l]), survivalC[i])
                else
                    logLprob[l] += logccdf(LogLogistic(_lambda[l] / xiC[i], _eta[l]), survivalC[i] - sum(gapC[i]))
                end 
            end
        end

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )
        tUh = sample(Weights(Lprob))

        tUT = zeros(Int64, NT+nT)
        for h_prime in 1:(NT+nT)
            tUT[h_prime] = _tUT[h_prime]
        end

        tUC = zeros(Int64, NC+nC)
        if tUh <= _g 
            # join an existing cluster 

            g = _g 
            mlC = zeros(Int64, g)
            mlT = zeros(Int64, g)

            lambda = zeros(g)
            gamma = zeros(g)
            eta = zeros(g)

            for l in 1:g
                lambda[l] = _lambda[l] 
                gamma[l] = _gamma[l]
                eta[l] = _eta[l]

                mlT[l] = _mlT[l]
                if l == tUh 
                    mlC[l] = _mlC[l] + 1
                else
                    mlC[l] = _mlC[l]
                end
            end 

            for h_prime in 1:(NC+nC)
                if h_prime != h
                    tUC[h_prime] = _tUC[h_prime]
                end
            end
            tUC[h] = tUh
        else
            # create a new cluster

            g = _g + 1 
            mlC = zeros(Int64, g)
            mlT = zeros(Int64, g)

            lambda = zeros(g)
            gamma = zeros(g)
            eta = zeros(g)

            for l in 1:(g-1)
                mlC[l] = _mlC[l]
                mlT[l] = _mlT[l]

                lambda[l] = _lambda[l] 
                gamma[l] = _gamma[l]
                eta[l] = _eta[l]
            end
            mlC[g] = 1
            mlT[g] = 0

            lambda[g] = _lambda[tUh]
            gamma[g] = _gamma[tUh]
            eta[g] = _eta[tUh] 

            for h_prime in 1:(NC+nC)
                if h_prime != h
                    tUC[h_prime] = _tUC[h_prime] 
                end
            end
            tUC[h] = g 
        end 

        for l in 1:g
            @assert mlC[l] == length(findall(tUC .== l)) print("C control \n", length(findall(tUC .== l)), "\n", l, "\n", mlC, "\n")
            @assert mlT[l] == length(findall(tUT .== l)) print("C treatment \n", length(findall(tUT .== l)), "\n", l, "\n", mlT, "\n")
            @assert mlC[l] + mlT[l] > 0 println("control empty cluster\n", mlC, " ", mlT)
        end
    end

    for h in 1:(NT+nT)

        tUh = tUT[h]
        ml_h = mlT[tUh]

        _tUT = zeros(Int64, NT+nT)
        _tUC = zeros(Int64, NC+nC)

        if (ml_h == 1) & (mlC[tUh] == 0)
            # a singleton 

            _g = g - 1
            _mlT = zeros(Int64, _g)
            _mlC = zeros(Int64, _g)

            _eta = zeros(_g+gstar)
            _lambda = zeros(_g+gstar) 
            _gamma = zeros(_g+gstar)

            for h_prime in 1:(NT+nT) 
                if h_prime != h
                    if tUT[h_prime] < tUh 
                        _tUT[h_prime] = tUT[h_prime]
                    elseif tUT[h_prime] > tUh
                        _tUT[h_prime] = tUT[h_prime] - 1
                    end
                end
            end

            for h_prime in 1:(NC+nC)
                if tUC[h_prime] < tUh 
                    _tUC[h_prime] = tUC[h_prime]
                elseif tUC[h_prime] > tUh
                    _tUC[h_prime] = tUC[h_prime] - 1
                end
            end

            for l in 1:_g
                if l < tUh 
                    _mlT[l] = mlT[l]
                    _mlC[l] = mlC[l]

                    _lambda[l] = lambda[l]
                    _gamma[l] = gamma[l]
                    _eta[l] = eta[l] 
                else
                    _mlT[l] = mlT[l+1]
                    _mlC[l] = mlC[l+1]

                    _lambda[l] = lambda[l+1]
                    _gamma[l] = gamma[l+1]
                    _eta[l] = eta[l+1] 
                end
            end
        else 
            # not a singleton 

            _g = g 
            _mlT = zeros(Int64, _g)
            _mlC = zeros(Int64, _g)

            _eta = zeros(_g+gstar)
            _lambda = zeros(_g+gstar) 
            _gamma = zeros(_g+gstar)

            for h_prime in 1:(NT+nT)
                if h_prime != h 
                    _tUT[h_prime] = tUT[h_prime]
                end
            end

            for h_prime in 1:(NC+nC)
                _tUC[h_prime] = tUC[h_prime]
            end

            for l in 1:_g

                _mlC[l] = mlC[l]
                if l == tUh 
                    _mlT[l] = mlT[l] - 1
                else
                    _mlT[l] = mlT[l]
                end

                _lambda[l] = lambda[l]
                _gamma[l] = gamma[l]
                _eta[l] = eta[l] 
            end
        end

        _indT = findall(_mlT .> 0)
        _gT = length(_indT)

        for l in (_g+1):(_g+gstar)
            _lambda[l] = exp(rand(Normal(mu_lambda, sqrt(sigma2_lambda)), 1)[1])
            _gamma[l] = rand(Normal(mu_gamma, sqrt(sigma2_gamma)), 1)[1]
            _eta[l] = sqrt(rand(Gamma(a_eta, b_eta), 1)[1])
        end

        logLprob = zeros(_g+gstar) 
        for l in 1:(_g+gstar) 
            # if l <= _g
            if l in _indT
                logLprob[l] = log(_mlT[l])
            else
                # logLprob[l] = log(zeta/gstar) 
                logLprob[l] = log(zetaT/(gstar + _g - _gT))
            end

            if h <= NT
                i,j = h2ij(h, NvecT)
                logLprob[l] += logpdf(LogLogistic(_lambda[l] * exp(_gamma[l]) / xiT[i], _eta[l]), gapT[i][j])
            else
                i = h - NT
                if NvecT[i] == 0 
                    logLprob[l] += logccdf(LogLogistic(_lambda[l] * exp(_gamma[l]) / xiT[i], _eta[l]), survivalT[i])
                else
                    logLprob[l] += logccdf(LogLogistic(_lambda[l] * exp(_gamma[l]) / xiT[i], _eta[l]), survivalT[i] - sum(gapT[i]))
                end 
            end
        end

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )
        tUh = sample(Weights(Lprob))

        tUC = zeros(Int64, NC+nC)
        for h_prime in 1:(NC+nC)
            tUC[h_prime] = _tUC[h_prime]
        end

        tUT = zeros(Int64, NT+nT)
        if tUh <= _g 
            # join an existing cluster 

            g = _g 
            mlT = zeros(Int64, g)
            mlC = zeros(Int64, g)

            lambda = zeros(g)
            gamma = zeros(g)
            eta = zeros(g)

            for l in 1:g
                lambda[l] = _lambda[l] 
                gamma[l] = _gamma[l]
                eta[l] = _eta[l]

                mlC[l] = _mlC[l]
                if l == tUh 
                    mlT[l] = _mlT[l] + 1
                else
                    mlT[l] = _mlT[l]
                end
            end 

            for h_prime in 1:(NT+nT)
                if h_prime != h
                    tUT[h_prime] = _tUT[h_prime]
                end
            end
            tUT[h] = tUh
        else
            # create a new cluster

            g = _g + 1
            mlT = zeros(Int64, g)
            mlC = zeros(Int64, g)

            lambda = zeros(g)
            gamma = zeros(g)
            eta = zeros(g)

            for l in 1:(g-1)
                mlT[l] = _mlT[l]
                mlC[l] = _mlC[l]

                lambda[l] = _lambda[l] 
                gamma[l] = _gamma[l]
                eta[l] = _eta[l]
            end

            mlT[g] = 1
            mlC[g] = 0

            lambda[g] = _lambda[tUh]
            gamma[g] = _gamma[tUh]
            eta[g] = _eta[tUh] 

            for h_prime in 1:(NT+nT)
                if h_prime != h
                    tUT[h_prime] = _tUT[h_prime] 
                end
            end
            tUT[h] = g 
        end 
        
        for l in 1:g
            @assert mlC[l] == length(findall(tUC .== l)) print("T control \n", length(findall(tUC .== l)), "\n", l, "\n", mlC, "\n")
            @assert mlT[l] == length(findall(tUT .== l)) print("T treatment \n", length(findall(tUT .== l)), "\n", l, "\n", mlT, "\n")
            @assert mlC[l] + mlT[l] > 0 println("control empty cluster\n", mlC, " ", mlT)
        end
    end

    tU = [tUC, tUT]
    ml = [mlC, mlT]

    return Dict("tU" => tU, "lambda" => lambda, "gamma" => gamma, "eta" => eta, "g" => g, "ml" => ml)
end