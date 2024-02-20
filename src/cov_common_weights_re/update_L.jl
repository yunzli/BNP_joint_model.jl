function update_L(dat, cur, hyper)

    L = cur["L"]
    theta = cur["theta"]
    beta = cur["beta"]
    phi = cur["phi"]
    nl = cur["nl"]
    epsilon = cur["epsilon"]
    k = cur["k"]

    nu = dat["nu"]
    survival = dat["survival"]
    n = dat["n"]
    x = dat["x"]
    p = dat["p"]

    kstar = hyper["kstar"]
    alpha = cur["alpha"]

    sigma2_theta = hyper["sigma2_theta"]
    Sigma_beta = hyper["Sigma_beta"]
    mu_theta = cur["mu_theta"]
    mu_beta = cur["mu_beta"]
    b_phi = cur["b_phi"]
    a_phi = hyper["a_phi"]

    for i in 1:n

        Li = L[i]
        nl_i = nl[Li]

        _k = 0
        _L = zeros(Int64, n)
        if nl_i > 1
            # not a singleton 

            _k = k 
            _nl = zeros(Int64, _k)
            _phi = zeros(_k+kstar)
            _theta = zeros(_k+kstar) 
            _beta = zeros(_k+kstar, p)

            for i_prime in 1:n 
                if i_prime != i 
                    _L[i_prime] = L[i_prime]
                end
            end

            for l in 1:_k

                if l == Li 
                    _nl[l] = nl[l] - 1
                else
                    _nl[l] = nl[l]
                end

                _theta[l] = theta[l]
                _beta[l,:] = beta[l,:] 
                _phi[l] = phi[l] 
                # print(nl, "\n\n")
            end
        else
            # a singleton 

            _k = k - 1
            _nl = zeros(Int64, _k)
            _phi = zeros(_k+kstar)
            _theta = zeros(_k+kstar) 
            _beta = zeros(_k+kstar, p)

            for i_prime in 1:n 
                if i_prime != i
                    if L[i_prime] < Li 
                        _L[i_prime] = L[i_prime]
                    elseif L[i_prime] > Li
                        _L[i_prime] = L[i_prime] - 1
                    end
                end
            end

            for l in 1:_k
                if l < Li 
                    _nl[l] = nl[l]
                    _theta[l] = theta[l]
                    _beta[l,:] = beta[l,:]
                    _phi[l] = phi[l] 
                else
                    _nl[l] = nl[l+1]
                    _theta[l] = theta[l+1]
                    _beta[l,:] = beta[l+1,:]
                    _phi[l] = phi[l+1] 
                end
            end
        end

        for l in (_k+1):(_k+kstar)
            _theta[l] = exp(rand(Normal(mu_theta, sqrt(sigma2_theta)), 1)[1])
            _beta[l,:] = rand(MvNormal(mu_beta, Sigma_beta), 1)
            _phi[l] = sqrt(rand(Gamma(a_phi, b_phi), 1)[1])
        end

        logLprob = zeros(_k+kstar) 
        for l in 1:(_k+kstar) 
            if l <= _k 
                logLprob[l] = log(_nl[l])
            else
                logLprob[l] = log(alpha/kstar) 
            end

            if nu[i] == 1
                logLprob[l] += logpdf(LogLogistic(_theta[l] * exp(_beta[l,:]' * x[i,:]) / epsilon[i], _phi[l]), survival[i])
            else
                logLprob[l] += logccdf(LogLogistic(_theta[l] * exp(_beta[l,:]' * x[i,:]) / epsilon[i], _phi[l]), survival[i])
            end
        end

        Lprob = exp.(logLprob .- Base.maximum(logLprob)) / sum( exp.(logLprob .- Base.maximum(logLprob)) )
        Li = sample(Weights(Lprob))

        L = zeros(Int64, n)
        if Li <= _k 
            # join an existing cluster 

            k = _k 
            theta = zeros(k)
            beta = zeros(k, p)
            phi = zeros(k)
            nl = zeros(Int64, k)

            for l in 1:k
                theta[l] = _theta[l] 
                beta[l,:] = _beta[l,:]
                phi[l] = _phi[l]
                if l == Li 
                    nl[l] = _nl[l] + 1
                else
                    nl[l] = _nl[l]
                end
            end 

            for i_prime in 1:n 
                if i_prime != i 
                    L[i_prime] = _L[i_prime]
                end
            end
            L[i] = Li
        else
            # create a new cluster

            k = _k + 1 
            theta = zeros(k)
            beta = zeros(k, p)
            phi = zeros(k)
            nl = zeros(Int64, k)

            for l in 1:(k-1)
                theta[l] = _theta[l] 
                beta[l,:] = _beta[l,:]
                phi[l] = _phi[l]
                nl[l] = _nl[l]
            end
            theta[k] = _theta[Li]
            beta[k,:] = _beta[Li,:]
            phi[k] = _phi[Li] 
            nl[k] = 1

            for i_prime in 1:n 
                if i_prime != i 
                    L[i_prime] = _L[i_prime] 
                end
            end
            L[i] = k 
        end 

        for l in 1:k 
            @assert nl[l] == length(findall(L .== l)) print("\n", L, "\n", i, " ", nl, "\n")
        end
               
    end

    return Dict("L" => L, "theta" => theta, "beta" => beta, "phi" => phi, "k" => k, "nl" => nl)
end