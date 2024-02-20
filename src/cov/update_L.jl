function update_L(datC, datT, cur, hyper)

    L = cur["L"]
    LC = L[1]
    LT = L[2]

    theta = cur["theta"]
    beta = cur["beta"]
    phi = cur["phi"]

    nl = cur["nl"]
    nlC = nl[1]
    nlT = nl[2]

    epsilon = cur["epsilon"]
    epsilonC = epsilon[1]
    epsilonT = epsilon[2]
    k = cur["k"]

    nuC = datC["nu"]
    survivalC = datC["survival"]
    nC = datC["n"]

    nuT = datT["nu"]
    survivalT = datT["survival"]
    nT = datT["n"]

    kstar = hyper["kstar"]

    alpha = cur["alpha"]
    alphaC = alpha[1]
    alphaT = alpha[2]

    sigma2_theta = hyper["sigma2_theta"]
    mu_theta = cur["mu_theta"]
    sigma2_beta = hyper["sigma2_beta"]
    mu_beta = cur["mu_beta"]
    a_phi = hyper["a_phi"]
    b_phi = cur["b_phi"]

    for i in 1:nC

        Li = LC[i]
        nl_i = nlC[Li]

        _LC = zeros(Int64, nC)
        _LT = zeros(Int64, nT)

        # println("initial: ", nlC, nlT)
        if (nl_i == 1) & (nlT[Li] == 0)
            # a singleton 
            # cluster LCi == l vanishes

            _k = k - 1
            _nlC = zeros(Int64, _k)
            _nlT = zeros(Int64, _k)

            _phi = zeros(_k+kstar)
            _theta = zeros(_k+kstar) 
            # _beta = zeros(_k+kstar, p)
            _beta = zeros(_k+kstar)

            for i_prime in 1:nC
                if i_prime != i
                    if LC[i_prime] < Li 
                        _LC[i_prime] = LC[i_prime]
                    elseif LC[i_prime] > Li
                        _LC[i_prime] = LC[i_prime] - 1
                    end
                end
            end

            for i_prime in 1:nT
                if LT[i_prime] < Li
                    _LT[i_prime] = LT[i_prime]
                elseif LT[i_prime] > Li
                    _LT[i_prime] = LT[i_prime] - 1
                end
            end

            for l in 1:_k
                if l < Li 
                    _nlC[l] = nlC[l]
                    _nlT[l] = nlT[l]

                    _theta[l] = theta[l]
                    # _beta[l,:] = beta[l,:]
                    _beta[l] = beta[l]
                    _phi[l] = phi[l]
                else
                    _nlC[l] = nlC[l+1]
                    _nlT[l] = nlT[l+1] # nlT[l] should be 0

                    _theta[l] = theta[l+1]
                    # _beta[l,:] = beta[l+1,:]
                    _beta[l] = beta[l+1]
                    _phi[l] = phi[l+1]
                end
            end
        else
            # not a singleton 

            _k = k
            _nlC = zeros(Int64, _k)
            _nlT = zeros(Int64, _k)

            _phi = zeros(_k+kstar)
            _theta = zeros(_k+kstar)
            # _beta = zeros(_k+kstar, p)
            _beta = zeros(_k+kstar)

            for i_prime in 1:nC
                if i_prime != i
                    _LC[i_prime] = LC[i_prime]
                end
            end

            for i_prime in 1:nT
                _LT[i_prime] = LT[i_prime]
            end

            for l in 1:_k

                _nlT[l] = nlT[l]
                if l == Li 
                    _nlC[l] = nlC[l] - 1
                else
                    _nlC[l] = nlC[l]
                end

                _theta[l] = theta[l]
                # _beta[l,:] = beta[l,:]
                _beta[l] = beta[l]
                _phi[l] = phi[l] 
            end
        end

        _indC = findall(_nlC .> 0)
        _kC = length(_indC)

        for l in (_k+1):(_k+kstar)
            # generate candidates from the baseline distribution 
            _theta[l] = exp(rand(Normal(mu_theta, sqrt(sigma2_theta)), 1)[1])
            # _beta[l,:] = rand(MvNormal(mu_beta, Sigma_beta), 1)
            _beta[l] = rand(Normal(mu_beta, sqrt(sigma2_beta)), 1)[1]
            _phi[l] = sqrt(rand(Gamma(a_phi, b_phi), 1)[1])
        end

        logLprob = zeros(_k+kstar) 
        for l in 1:(_k+kstar) 
            # if l <= _k 
            if l in _indC  
                logLprob[l] = log(_nlC[l])
            else
                logLprob[l] = log(alphaC/(kstar + _k - _kC)) 
            end

            if nuC[i] == 1
                logLprob[l] += logpdf(LogLogistic(_theta[l] / epsilonC[i], _phi[l]), survivalC[i])
            else
                logLprob[l] += logccdf(LogLogistic(_theta[l] / epsilonC[i], _phi[l]), survivalC[i])
            end
        end

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )
        Li = sample(Weights(Lprob))


        LT = zeros(Int64, nT)
        for i_prime in 1:nT
            LT[i_prime] = _LT[i_prime]
        end

        LC = zeros(Int64, nC)
        if Li <= _k
            # join an existing cluster 

            k = _k 
            theta = zeros(k)
            phi = zeros(k)
            # beta = zeros(k, p)
            beta = zeros(k)
            nlC = zeros(Int64, k)
            nlT = zeros(Int64, k)

            for l in 1:k
                theta[l] = _theta[l] 
                # beta[l,:] = _beta[l,:]
                beta[l] = _beta[l]
                phi[l] = _phi[l]
                nlT[l] = _nlT[l]
                if l == Li 
                    nlC[l] = _nlC[l] + 1
                else
                    nlC[l] = _nlC[l]
                end
            end 

            for i_prime in 1:nC
                if i_prime != i 
                    LC[i_prime] = _LC[i_prime]
                end
            end
            LC[i] = Li
        else
            # create a new cluster

            k = _k + 1 
            nlC = zeros(Int64, k)
            nlT = zeros(Int64, k)

            theta = zeros(k)
            # beta = zeros(k,p)
            beta = zeros(k)
            phi = zeros(k)

            for l in 1:(k-1)
                nlC[l] = _nlC[l]
                nlT[l] = _nlT[l]

                theta[l] = _theta[l]
                # beta[l,:] = _beta[l,:]
                beta[l] = _beta[l]
                phi[l] = _phi[l]
            end
            nlC[k] = 1
            nlT[k] = 0

            theta[k] = _theta[Li]
            # beta[k,:] = _beta[Li,:]
            beta[k] = _beta[Li]
            phi[k] = _phi[Li]

            for i_prime in 1:nC
                if i_prime != i
                    LC[i_prime] = _LC[i_prime]
                end
            end
            LC[i] = k
        end 

        # println("conclusion: ", nlC, nlT)
        for l in 1:k 
            @assert nlC[l] == length(findall(LC .== l)) print("control \n", length(findall(LC .== l)), "\n", l, "\n", nlC, "\n")
            @assert nlT[l] == length(findall(LT .== l)) print("treatment \n", length(findall(LT .== l)), "\n", l, "\n", nlT, "\n")
            @assert nlC[l] + nlT[l] > 0 println("control empty cluster\n", nlC, " ", nlT)
        end
    end

    for i in 1:nT

        Li = LT[i]
        nl_i = nlT[Li]

        _LT = zeros(Int64, nT)
        _LC = zeros(Int64, nC)

        # println("initial: ", nlC, nlT)
        if (nl_i == 1) & (nlC[Li] == 0)
            # a singleton 
            # cluster LTi == l vanishes

            _k = k - 1
            _nlT = zeros(Int64, _k)
            _nlC = zeros(Int64, _k)

            _phi = zeros(_k+kstar)
            _theta = zeros(_k+kstar)
            # _beta = zeros(_k+kstar, p)
            _beta = zeros(_k+kstar)

            for i_prime in 1:nT
                if i_prime != i
                    if LT[i_prime] < Li
                        _LT[i_prime] = LT[i_prime]
                    elseif LT[i_prime] > Li
                        _LT[i_prime] = LT[i_prime] - 1
                    end
                end
            end

            for i_prime in 1:nC
                if LC[i_prime] < Li
                    _LC[i_prime] = LC[i_prime]
                elseif LC[i_prime] > Li
                    _LC[i_prime] = LC[i_prime] - 1
                end
            end

            for l in 1:_k
                if l < Li 
                    _nlT[l] = nlT[l]
                    _nlC[l] = nlC[l]

                    _theta[l] = theta[l]
                    # _beta[l,:] = beta[l,:]
                    _beta[l] = beta[l]
                    _phi[l] = phi[l] 
                else
                    _nlT[l] = nlT[l+1] 
                    _nlC[l] = nlC[l+1] # nlT[l] should be 0

                    _theta[l] = theta[l+1]
                    # _beta[l,:] = beta[l+1,:]
                    _beta[l] = beta[l+1]
                    _phi[l] = phi[l+1]
                end
            end
        else
            # not a singleton

            _k = k
            _nlT = zeros(Int64, _k)
            _nlC = zeros(Int64, _k)

            _phi = zeros(_k+kstar)
            _theta = zeros(_k+kstar)
            # _beta = zeros(_k+kstar, p)
            _beta = zeros(_k+kstar)

            for i_prime in 1:nT
                if i_prime != i
                    _LT[i_prime] = LT[i_prime]
                end
            end

            for i_prime in 1:nC
                _LC[i_prime] = LC[i_prime]
            end

            for l in 1:_k

                _nlC[l] = nlC[l]
                if l == Li
                    _nlT[l] = nlT[l] - 1
                else
                    _nlT[l] = nlT[l]
                end

                _theta[l] = theta[l]
                # _beta[l,:] = beta[l,:]
                _beta[l] = beta[l]
                _phi[l] = phi[l] 
            end
        end

        for l in (_k+1):(_k+kstar)
            _theta[l] = exp(rand(Normal(mu_theta, sqrt(sigma2_theta)), 1)[1])
            _beta[l] = rand(Normal(mu_beta, sqrt(sigma2_beta)), 1)[1]
            _phi[l] = sqrt(rand(Gamma(a_phi, b_phi), 1)[1])
        end

        _indT = findall(_nlT .> 0)
        _kT = length(_indT)

        logLprob = zeros(_k+kstar) 
        for l in 1:(_k+kstar) 
            # if l <= _k 
            if l in _indT
                logLprob[l] = log(_nlT[l])
            else
                logLprob[l] = log(alphaT/(kstar+_k-_kT))
            end

            if nuT[i] == 1
                logLprob[l] += logpdf(LogLogistic(_theta[l] * exp(_beta[l]) / epsilonT[i], _phi[l]), survivalT[i])
            else
                logLprob[l] += logccdf(LogLogistic(_theta[l] * exp(_beta[l]) / epsilonT[i], _phi[l]), survivalT[i])
            end
        end

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )
        Li = sample(Weights(Lprob))

        # println("treatment")
        # println(_nlC)
        # println(_nlT)
        # println(Lprob)

        LC = zeros(Int64, nC)
        for i_prime in 1:nC
            LC[i_prime] = _LC[i_prime]
        end

        LT = zeros(Int64, nT)
        if Li <= _k 
            # join an existing cluster 

            k = _k
            theta = zeros(k)
            # beta =  zeros(k, p)
            beta = zeros(k)
            phi = zeros(k)
            nlT = zeros(Int64, k)
            nlC = zeros(Int64, k)

            for l in 1:k
                theta[l] = _theta[l]
                # beta[l,:] = _beta[l,:]
                beta[l] = _beta[l]
                phi[l] = _phi[l]
                nlC[l] = _nlC[l]
                if l == Li
                    nlT[l] = _nlT[l] + 1
                else
                    nlT[l] = _nlT[l]
                end
            end

            for i_prime in 1:nT
                if i_prime != i
                    LT[i_prime] = _LT[i_prime]
                end
            end
            LT[i] = Li
        else
            # create a new cluster

            k = _k + 1
            nlT = zeros(Int64, k)
            nlC = zeros(Int64, k)

            theta = zeros(k)
            # beta = zeros(k, p)
            beta = zeros(k)
            phi = zeros(k)

            for l in 1:(k-1)
                nlT[l] = _nlT[l]
                nlC[l] = _nlC[l]

                theta[l] = _theta[l] 
                # beta[l,:] = _beta[l,:]
                beta[l] = _beta[l]
                phi[l] = _phi[l]
            end
            nlT[k] = 1
            nlC[k] = 0

            theta[k] = _theta[Li]
            # beta[k,:] = _beta[Li,:]
            beta[k] = _beta[Li]
            phi[k] = _phi[Li]

            for i_prime in 1:nT
                if i_prime != i
                    LT[i_prime] = _LT[i_prime]
                end
            end
            LT[i] = k
        end

        # println("conclusion: ", nlC, nlT)
        for l in 1:k
            @assert nlC[l] == length(findall(LC .== l)) print("control \n", length(findall(LC .== l)), "\n", l, "\n", nlC, "\n")
            @assert nlT[l] == length(findall(LT .== l)) print("treatment \n", length(findall(LT .== l)), "\n", l, "\n", nlT, "\n")
            @assert nlC[l] + nlT[l] > 0 println("control empty cluster\n", nlC, " ", nlT)
        end
    end

    L = [LC, LT]
    nl = [nlC, nlT]

    # if Base.maximum(LT) > length(theta) 
    #     println("C warning")
    # end
    # if Base.maximum(LC) > length(theta) 
    #     println("T warning")
    # end

    return Dict("L" => L, "theta" => theta, "beta" => beta, "phi" => phi, "k" => k, "nl" => nl)
end