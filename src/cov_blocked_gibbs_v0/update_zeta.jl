function update_zeta(cur, hyper)

    a_zeta = hyper["a_zeta"]
    b_zeta = hyper["b_zeta"]
    BH = hyper["BH"]

    logomegaC = cur["logomegaC"]
    logomegaT = cur["logomegaT"]

    a_zeta_new = BH + a_zeta - 1
    b_zetaC_new = 1 / (1 / b_zeta - logomegaC[end])
    b_zetaT_new = 1 / (1 / b_zeta - logomegaT[end])

    zetaC = rand(Gamma(a_zeta_new, b_zetaC_new), 1)[1]
    zetaT = rand(Gamma(a_zeta_new, b_zetaT_new), 1)[1]

    return [zetaC, zetaT]
end