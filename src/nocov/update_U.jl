function update_U(dat, cur, hyper)

    tU = cur["tU"]
    lambda = cur["lambda"]
    eta = cur["eta"]
    ml = cur["ml"]
    xi = cur["xi"]
    g = cur["g"]

    survival = dat["survival"]
    iota = dat["iota"]
    gap = dat["gap"]
    Nvec = dat["Nvec"] # N1, N2, ..., Nn
    N = dat["N"]
    n = dat["n"]

    gstar = hyper["gstar"]
    zeta = cur["zeta"]

    sigma2_lambda = hyper["sigma2_lambda"]
    mu_lambda = cur["mu_lambda"]
    b_eta = cur["b_eta"]
    a_eta = hyper["a_eta"]

    for h in 1:(N+n)

        tUh = tU[h]
        ml_h = ml[tUh]

        _g = 0
        _tU = zeros(Int64, N+n)
        if ml_h > 1
            # not a singleton 

            _g = g 
            _ml = zeros(Int64, _g)
            _eta = zeros(_g+gstar)
            _lambda = zeros(_g+gstar) 

            for h_prime in 1:(N+n)
                if h_prime != h 
                    _tU[h_prime] = tU[h_prime]
                end
            end

            for l in 1:_g

                if l == tUh 
                    _ml[l] = ml[l] - 1
                else
                    _ml[l] = ml[l]
                end

                _lambda[l] = lambda[l]
                _eta[l] = eta[l] 
            end
        else
            # a singleton 

            _g = g - 1
            _ml = zeros(Int64, _g)
            _eta = zeros(_g+gstar)
            _lambda = zeros(_g+gstar) 

            for h_prime in 1:(N+n) 
                if h_prime != h
                    if tU[h_prime] < tUh 
                        _tU[h_prime] = tU[h_prime]
                    elseif tU[h_prime] > tUh
                        _tU[h_prime] = tU[h_prime] - 1
                    end
                end
            end

            for l in 1:_g
                if l < tUh 
                    _ml[l] = ml[l]
                    _lambda[l] = lambda[l]
                    _eta[l] = eta[l] 
                else
                    _ml[l] = ml[l+1]
                    _lambda[l] = lambda[l+1]
                    _eta[l] = eta[l+1] 
                end
            end
        end

        for l in (_g+1):(_g+gstar)
            _lambda[l] = exp(rand(Normal(mu_lambda, sqrt(sigma2_lambda)), 1)[1])
            _eta[l] = sqrt(rand(Gamma(a_eta, b_eta), 1)[1])
        end

        logLprob = zeros(_g+gstar) 
        for l in 1:(_g+gstar) 
            if l <= _g
                logLprob[l] = log(_ml[l])
            else
                logLprob[l] = log(zeta/gstar) 
            end

            if h <= N
                i,j = h2ij(h, Nvec)
                logLprob[l] += logpdf(LogLogistic(_lambda[l] / xi[i], _eta[l]), gap[i][j])
            else
                i = h - N
                if Nvec[i] == 0 
                    logLprob[l] += logccdf(LogLogistic(_lambda[l] / xi[i], _eta[l]), survival[i])
                else
                    logLprob[l] += logccdf(LogLogistic(_lambda[l] / xi[i], _eta[l]), survival[i] - sum(gap[i]))
                end 
            end
        end

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )
        tUh = sample(Weights(Lprob))

        tU = zeros(Int64, N+n)
        if tUh <= _g 
            # join an existing cluster 

            g = _g 
            lambda = zeros(g)
            eta = zeros(g)
            ml = zeros(Int64, g)

            for l in 1:g
                lambda[l] = _lambda[l] 
                eta[l] = _eta[l]
                if l == tUh 
                    ml[l] = _ml[l] + 1
                else
                    ml[l] = _ml[l]
                end
            end 

            for h_prime in 1:(N+n)
                if h_prime != h
                    tU[h_prime] = _tU[h_prime]
                end
            end
            tU[h] = tUh
        else
            # create a new cluster

            g = _g + 1 
            lambda = zeros(g)
            eta = zeros(g)
            ml = zeros(Int64, g)

            for l in 1:(g-1)
                lambda[l] = _lambda[l] 
                eta[l] = _eta[l]
                ml[l] = _ml[l]
            end
            lambda[g] = _lambda[tUh]
            eta[g] = _eta[tUh] 
            ml[g] = 1

            for h_prime in 1:(N+n) 
                if h_prime != h
                    tU[h_prime] = _tU[h_prime] 
                end
            end
            tU[h] = g 
        end 
    end

    return Dict("tU" => tU, "lambda" => lambda, "eta" => eta, "g" => g, "ml" => ml)
end