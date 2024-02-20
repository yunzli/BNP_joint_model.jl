function update_alpha(cur, hyper)

    alpha_cur = cur["alpha"]
    alpha0_cur = cur["alpha0"]
    latent = cur["latent_surv"]
    BG = hyper["BG"]

    a0, b0, c0 = hyper["a0"], hyper["b0"], hyper["c0"]

    alpha0_pro = rand(Beta(10*alpha0_cur, 10*(1 - alpha0_cur)), 1)[1]
    alpha_pro = rand(Pareto(2, alpha0_pro), 1)[1]

    logcur = logpdf(Pareto(c0, alpha0_cur), alpha_cur) + logpdf(Beta(a0, b0), alpha0_cur) 
    for l in 1:(BG-1)
        logcur += logpdf(Dirichlet([1-alpha0_cur, alpha0_cur, alpha0_cur, alpha_cur - alpha0_cur]), latent[l,:])
    end
    logcur = logcur - logpdf(Beta(10*alpha0_pro, 10*(1 - alpha0_pro)), alpha0_cur) - logpdf(Pareto(2, alpha0_cur), alpha_cur)

    logpro = logpdf(Pareto(c0, alpha0_pro), alpha_pro) + logpdf(Beta(a0, b0), alpha0_pro) 
    for l in 1:(BG-1)
        logpro += logpdf(Dirichlet([1-alpha0_pro, alpha0_pro, alpha0_pro, alpha_pro - alpha0_pro]), latent[l,:])
    end
    logpro = logpro - logpdf(Beta(10*alpha0_cur, 10*(1 - alpha0_cur)), alpha0_pro) - logpdf(Pareto(2, alpha0_pro), alpha_pro)

    if log(rand(Uniform(0,1),1)[1]) < (logpro - logcur)
        return [alpha_pro, alpha0_pro]
    else
        return [alpha_cur, alpha0_cur]
    end 
end