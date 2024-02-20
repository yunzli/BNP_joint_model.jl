function update_p(cur, hyper)

    BG = hyper["BG"]

    LC = cur["LC"]
    LT = cur["LT"]
    alpha = cur["alpha"]
    alpha0 = cur["alpha0"]

    VC = zeros(BG-1)
    VT = zeros(BG-1)
    logpC = zeros(BG)
    logpT = zeros(BG)
    nlC = zeros(Int64, BG)
    nlT = zeros(Int64, BG)
    
    latent_cur = cur["latent_surv"]
    latent = zeros(BG-1, 4)

    for l in 1:BG

        nlC[l] = length(findall(LC .== l))
        nlT[l] = length(findall(LT .== l))

    end 

    for l in 1:(BG-1)

        proposal_factor = 100

        latent_pro = rand(Dirichlet(proposal_factor*latent_cur[l,:]), 1)[:,1]
        latent_pro[findall(latent_pro .== 0)] .= 1e-10

        logcur = logpdf(Dirichlet([1-alpha0,alpha0,alpha0,alpha-alpha0]), latent_cur[l,:])
        logcur = logcur - logpdf(Dirichlet(proposal_factor*latent_pro), latent_cur[l,:]) 
        logcur = logcur + nlC[l] * log(latent_cur[l,1] + latent_cur[l,2])
        logcur = logcur + sum(nlC[(l+1):end]) * log(latent_cur[l,3] + latent_cur[l,4])
        logcur = logcur + nlT[l] * log(latent_cur[l,1] + latent_cur[l,3])
        logcur = logcur + sum(nlT[(l+1):end]) * log(latent_cur[l,2] + latent_cur[l,4])

        logpro = logpdf(Dirichlet([1-alpha0,alpha0,alpha0,alpha-alpha0]), latent_pro)
        logpro = logpro - logpdf(Dirichlet(proposal_factor*latent_cur[l,:]), latent_pro) 
        logpro = logpro + nlC[l] * log(latent_pro[1] + latent_pro[2])
        logpro = logpro + sum(nlC[(l+1):end]) * log(latent_pro[3] + latent_pro[4])
        logpro = logpro + nlT[l] * log(latent_pro[1] + latent_pro[3])
        logpro = logpro + sum(nlT[(l+1):end]) * log(latent_pro[2] + latent_pro[4])
        
        if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
           latent[l,:] = latent_pro 
        else
           latent[l,:] = latent_cur[l,:] 
        end 

        VC[l] = latent[l,1] + latent[l,2]
        VT[l] = latent[l,1] + latent[l,3] 

    end

    VC[findall(VC .== 0)] .= 1e-10
    VC[findall(VC .> 1)] .= 1 - 1e-10
    VT[findall(VT .== 0)] .= 1e-10
    VT[findall(VT .> 1)] .= 1 - 1e-10

    logpC[1] = log(VC[1])
    logpT[1] = log(VT[1])

    for l in 2:(BG-1)
        logpC[l] = log(VC[l]) + sum(log.(1 .- VC[1:(l-1)]))
        logpT[l] = log(VT[l]) + sum(log.(1 .- VT[1:(l-1)]))
    end

    logpC[end] =  sum(log.(1 .- VC))
    logpT[end] =  sum(log.(1 .- VT))
    # print(logp[N], "\n")

    for l in 1:BG
        if logpC[l] == -Inf
            # print(logp[l], " ", l, "\n")
            logpC[l] = log(eps(0.0))
        end 
        if logpT[l] == -Inf
            # print(logp[l], " ", l, "\n")
            logpT[l] = log(eps(0.0))
        end 
    end

	@assert sum(exp.(logpC)) ≈ 1.0
	@assert sum(exp.(logpT)) ≈ 1.0

    # println(exp.(logpC))
    # println(exp.(logpT))
    # println(" ")
	return Dict("logpC" => logpC, "logpT" => logpT, "latent" => latent)
end