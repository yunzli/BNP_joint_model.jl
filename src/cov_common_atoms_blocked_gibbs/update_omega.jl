function update_omega(cur, hyper)

    BH = hyper["BH"]

    tUC = cur["tUC"]
    tUT = cur["tUT"]
    zeta = cur["zeta"]
    zeta0 = cur["zeta0"]

    piC = zeros(BH-1)
    piT = zeros(BH-1)

    logomegaC = zeros(BH)
    logomegaT = zeros(BH)
    mlC = zeros(Int64, BH)
    mlT = zeros(Int64, BH)

    latent_cur = cur["latent_gap"]
    latent = zeros(BH-1, 4)

    for l in 1:BH

        mlC[l] = length(findall(tUC .== l))
        mlT[l] = length(findall(tUT .== l))

    end

    for l in 1:(BH-1)

        latent_pro = rand(Dirichlet(10*latent_cur[l,:]), 1)[:,1]
        latent_pro[findall(latent_pro .== 0)] .= 1e-10

        logcur = logpdf(Dirichlet([1-zeta0,zeta0,zeta0,zeta-zeta0]), latent_cur[l,:])
        logcur = logcur - logpdf(Dirichlet(10*latent_pro), latent_cur[l,:]) 
        logcur = logcur + mlC[l] * log(latent_cur[l,1] + latent_cur[l,2])
        logcur = logcur + sum(mlC[(l+1):end]) * log(latent_cur[l,3] + latent_cur[l,4])
        logcur = logcur + mlT[l] * log(latent_cur[l,1] + latent_cur[l,3])
        logcur = logcur + sum(mlT[(l+1):end]) * log(latent_cur[l,2] + latent_cur[l,4])

        logpro = logpdf(Dirichlet([1-zeta0,zeta0,zeta0,zeta-zeta0]), latent_pro)
        logpro = logpro - logpdf(Dirichlet(10*latent_cur[l,:]), latent_pro) 
        logpro = logpro + mlC[l] * log(latent_pro[1] + latent_pro[2])
        logpro = logpro + sum(mlC[(l+1):end]) * log(latent_pro[3] + latent_pro[4])
        logpro = logpro + mlT[l] * log(latent_pro[1] + latent_pro[3])
        logpro = logpro + sum(mlT[(l+1):end]) * log(latent_pro[2] + latent_pro[4])
          
        if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
           latent[l,:] = latent_pro 
        else
           latent[l,:] = latent_cur[l,:] 
        end 

        piC[l] = latent[l,1] + latent[l,2]
        piT[l] = latent[l,1] + latent[l,3]     

    end

    piC[findall(piC .== 0)] .= 1e-10
    piC[findall(piC .> 1)] .= 1 - 1e-10
    piT[findall(piT .== 0)] .= 1e-10
    piT[findall(piT .> 1)] .= 1 - 1e-10

    logomegaC[1] = log(piC[1])
    logomegaT[1] = log(piT[1])

    for l in 2:(BH-1)
        logomegaC[l] = log(piC[l]) + sum(log.(1 .- piC[1:(l-1)]))
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

	return Dict("logomegaC" => logomegaC, "logomegaT" => logomegaT, "latent" => latent)
end