function update_zeta(cur, hyper)

    zeta_cur = cur["zeta"]
    zeta0_cur = cur["zeta0"]
    latent = cur["latent_gap"]
    BH = hyper["BH"]

    a1, b1, c1 = hyper["a1"], hyper["b1"], hyper["c1"]

    proposal_factor = 100
    zeta0_pro = rand(Beta(proposal_factor*zeta0_cur, proposal_factor*(1 - zeta0_cur)), 1)[1]
    zeta_pro = rand(Pareto(2, zeta0_pro), 1)[1]

    logcur = logpdf(Pareto(c1, zeta0_cur), zeta_cur) + logpdf(Beta(a1, b1), zeta0_cur) 
    for l in 1:(BH-1)
        logcur += logpdf(Dirichlet([1-zeta0_cur, zeta0_cur,  zeta0_cur, zeta_cur - zeta0_cur]), latent[l,:])
    end
    logcur = logcur - logpdf(Beta(proposal_factor*zeta0_pro, proposal_factor*(1 - zeta0_pro)), zeta0_cur) - logpdf(Pareto(2, zeta0_cur), zeta_cur)

    logpro = logpdf(Pareto(c1, zeta0_pro), zeta_pro) + logpdf(Beta(a1, b1), zeta0_pro) 
    for l in 1:(BH-1)
        logpro += logpdf(Dirichlet([1-zeta0_pro, zeta0_pro,  zeta0_pro, zeta_pro - zeta0_pro]), latent[l,:])
    end
    logpro = logpro - logpdf(Beta(proposal_factor*zeta0_cur, proposal_factor*(1 - zeta0_cur)), zeta0_pro) - logpdf(Pareto(2, zeta0_pro), zeta_pro)

    if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
        return [zeta_pro, zeta0_pro]
    else
        return [zeta_cur, zeta0_cur]
    end 
end

function update_zeta0(cur, hyper)
    # This function is used only when zeta is fixed
    zeta0_cur = cur["zeta0"]
    latent = cur["latent_gap"]
    BG = hyper["BG"]

    a0, b0 = hyper["a0"], hyper["b0"]
    zeta = hyper["zeta"]

    zeta0_pro = rand(Beta(10*zeta0_cur, 10*(1 - zeta0_cur)), 1)[1]
    
    logcur = logpdf(Beta(a0, b0), zeta0_cur) 
    for l in 1:(BG-1)
        logcur += logpdf(Dirichlet([1-zeta0_cur, zeta0_cur, zeta0_cur, zeta - zeta0_cur]), latent[l,:])
    end
    logcur = logcur - logpdf(Beta(10*zeta0_pro, 10*(1 - zeta0_pro)), zeta0_cur)

    logpro = logpdf(Beta(a0, b0), zeta0_pro) 
    for l in 1:(BG-1)
        logpro += logpdf(Dirichlet([1-zeta0_pro, zeta0_pro, zeta0_pro, zeta - zeta0_pro]), latent[l,:])
    end
    logpro = logpro - logpdf(Beta(10*zeta0_cur, 10*(1 - zeta0_cur)), zeta0_pro)

    if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
        return zeta0_pro
    else
        return zeta0_cur
    end 
end