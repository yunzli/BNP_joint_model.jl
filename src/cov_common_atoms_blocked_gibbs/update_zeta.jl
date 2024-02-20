function update_zeta(cur, hyper)

    zeta_cur = cur["zeta"]
    zeta0_cur = cur["zeta0"]
    latent = cur["latent_gap"]
    BH = hyper["BH"]

    a1, b1, c1 = hyper["a1"], hyper["b1"], hyper["c1"]

    zeta0_pro = rand(Beta(10*zeta0_cur, 10*(1 - zeta0_cur)), 1)[1]
    zeta_pro = rand(Pareto(2, zeta0_pro), 1)[1]

    logcur = logpdf(Pareto(c1, zeta0_cur), zeta_cur) + logpdf(Beta(a1, b1), zeta0_cur) 
    for l in 1:(BH-1)
        logcur += logpdf(Dirichlet([1-zeta0_cur, zeta0_cur,  zeta0_cur, zeta_cur - zeta0_cur]), latent[l,:])
    end
    logcur = logcur - logpdf(Beta(10*zeta0_pro, 10*(1 - zeta0_pro)), zeta0_cur) - logpdf(Pareto(2, zeta0_cur), zeta_cur)

    logpro = logpdf(Pareto(c1, zeta0_pro), zeta_pro) + logpdf(Beta(a1, b1), zeta0_pro) 
    for l in 1:(BH-1)
        logpro += logpdf(Dirichlet([1-zeta0_pro, zeta0_pro,  zeta0_pro, zeta_pro - zeta0_pro]), latent[l,:])
    end
    logpro = logpro - logpdf(Beta(10*zeta0_cur, 10*(1 - zeta0_cur)), zeta0_pro) - logpdf(Pareto(2, zeta0_pro), zeta_pro)

    if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
        return [zeta_pro, zeta0_pro]
    else
        return [zeta_cur, zeta0_cur]
    end 
end