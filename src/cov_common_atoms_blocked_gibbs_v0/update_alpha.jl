function update_alpha(cur, hyper)

    a_alpha = hyper["a_alpha"]
    b_alpha = hyper["b_alpha"]
    BG = hyper["BG"]

    logpC = cur["logpC"]
    logpT = cur["logpT"]

    a_alpha_new = BG + a_alpha - 1
    b_alphaC_new = 1 / (1 / b_alpha - logpC[end])
    b_alphaT_new = 1 / (1 / b_alpha - logpT[end])

    alphaC = rand(Gamma(a_alpha_new, b_alphaC_new), 1)[1]
    alphaT = rand(Gamma(a_alpha_new, b_alphaT_new), 1)[1]

    return [alphaC, alphaT]
end