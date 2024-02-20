using TOML
using RCall 
using JLD2 
using Distributions 
using ProgressMeter
using LinearAlgebra
using StatsBase
using Random 

Random.seed!(100)

include("../src/loglogistic.jl")
include("functional_estimates.jl")

# config = TOML.parsefile("../configs/effusion_cov_common_atoms.TOML")
config = TOML.parsefile("../configs/effusion_cov_common_atoms_s2.TOML")

dataC = load(config["data_fileC"])
gapC = dataC["gap"]
survivalC = dataC["survival"]
arrivalC = dataC["arrival"]
nuC = dataC["nu"]

dataT = load(config["data_fileT"])
gapT = dataT["gap"]
survivalT = dataT["survival"]
arrivalT = dataT["arrival"]
nuT = dataT["nu"]


fitdata = load(config["save_path"])
pos = fitdata["pos"]
hyper = fitdata["hyper"]

nsam = size(pos["alpha"])[1]
nburn = div(nsam, 4)
nthin = div(nsam-nburn,2000)
keep_index = [nburn+1:nthin:nsam;]
nkeep = length(keep_index)


n_clusters = pos["k"][keep_index]
theta_save = pos["theta"][keep_index]
phi_save = pos["phi"][keep_index]
mu_theta_save = pos["mu_theta"][keep_index]
b_phi_save = pos["b_phi"][keep_index]

m_clusters = pos["g"][keep_index]
lambda_save = pos["lambda"][keep_index]
eta_save = pos["eta"][keep_index]
mu_lambda_save = pos["mu_lambda"][keep_index]
b_eta_save = pos["b_eta"][keep_index]


alpha_saveC = pos["alpha"][keep_index,1]
L_saveC = [x[1] for x in pos["L"][keep_index]]
nl_saveC = [x[1] for x in pos["nl"][keep_index]]
zeta_saveC = pos["zeta"][keep_index,2]
tU_saveC = [x[1] for x in pos["tU"][keep_index]]
ml_saveC = [x[1] for x in pos["ml"][keep_index]]

alpha_saveT = pos["alpha"][keep_index,2]
L_saveT = [x[2] for x in pos["L"][keep_index]]
nl_saveT = [x[2] for x in pos["nl"][keep_index]]
zeta_saveT = pos["zeta"][keep_index,2]
tU_saveT = [x[2] for x in pos["tU"][keep_index]]
ml_saveT = [x[2] for x in pos["ml"][keep_index]]

Sigma_e_save = pos["Sigma_e"][keep_index,:,:]

sigma2_theta = hyper["sigma2_theta"]
a_phi = hyper["a_phi"]

sigma2_lambda = hyper["sigma2_lambda"]
a_eta = hyper["a_eta"]

fig_path = config["fig_path"]
if !isdir(fig_path)
    print(fig_path,"\n")
	mkdir(fig_path)
end
@rput fig_path 


r_phi, R_phi = hyper["r_phi"], hyper["R_phi"]
s_theta, S_theta = hyper["s_theta"], hyper["S_theta"]
a_alpha, b_alpha = hyper["a_alpha"], hyper["b_alpha"]
@rput r_phi R_phi s_theta S_theta a_alpha b_alpha

r_eta, R_eta = hyper["r_eta"], hyper["R_eta"]
s_lambda, S_lambda = hyper["s_lambda"], hyper["S_lambda"]
a_zeta, b_zeta = hyper["a_zeta"], hyper["b_zeta"]
@rput r_eta R_eta s_lambda S_lambda a_zeta b_zeta

@rput mu_theta_save mu_lambda_save 
@rput b_phi_save b_eta_save 
R"""
library(ggplot2)
png(paste0(fig_path,"mu_theta_trace.png"))
plot(mu_theta_save, type='l', cex.axis=2)
dev.off() 

mu_theta_grids = seq(min(mu_theta_save), max(mu_theta_save), length=200)
mu_theta_dens = dnorm(mu_theta_grids, s_theta, sqrt(S_theta))

png(paste0(fig_path,"mu_theta_hist.png"))
hist(mu_theta_save, main="", cex.axis=2, freq=FALSE)
lines(mu_theta_dens ~ mu_theta_grids, type="l", col="red")
dev.off() 

png(paste0(fig_path,"b_phi_trace.png"))
plot(b_phi_save, type='l', cex.axis=2)
dev.off() 

b_phi_grids = seq(min(b_phi_save), max(b_phi_save), length=200)
b_phi_dens = invgamma::dinvgamma(b_phi_grids, r_phi, R_phi)

png(paste0(fig_path,"b_phi_hist.png"))
hist(b_phi_save, main="", cex.axis=2, freq=FALSE)
lines(b_phi_dens ~ b_phi_grids, type="l", col="red")
dev.off() 

png(paste0(fig_path,"mu_lambda_trace.png"))
plot(mu_lambda_save, type='l', cex.axis=2)
dev.off() 

mu_lambda_grids = seq(min(mu_lambda_save), max(mu_lambda_save), length=200)
mu_lambda_dens = dnorm(mu_lambda_grids, s_lambda, sqrt(S_lambda))

png(paste0(fig_path,"mu_lambda_hist.png"))
hist(mu_lambda_save, main="", cex.axis=2, freq=FALSE)
lines(mu_lambda_dens ~ mu_lambda_grids, type="l", col="red")
dev.off() 

png(paste0(fig_path,"b_eta_trace.png"))
plot(b_eta_save, type='l', cex.axis=2)
dev.off() 

b_eta_grids = seq(min(b_eta_save), max(b_eta_save), length=200)
b_eta_dens = invgamma::dinvgamma(b_eta_grids, r_eta, R_eta)

png(paste0(fig_path,"b_eta_hist.png"))
hist(b_eta_save, main="", cex.axis=2, freq=FALSE)
lines(b_eta_dens ~ b_eta_grids, type="l", col="red")
dev.off() 
"""

@rput Sigma_e_save
R"""
png(paste0(fig_path,"Sigma_e_11_trace.png"))
plot(Sigma_e_save[,1,1], type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_11_hist.png"))
hist(Sigma_e_save[,1,1], main="", cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_12_trace.png"))
plot(Sigma_e_save[,1,2], type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_12_hist.png"))
hist(Sigma_e_save[,1,2], main="", cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_22_trace.png"))
plot(Sigma_e_save[,2,2], type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_22_hist.png"))
hist(Sigma_e_save[,2,2], main="", cex.axis=2)
dev.off() 
"""

@rput alpha_saveC alpha_saveT; 
R"""
png(paste0(fig_path, "alphaC_trace.png"))
plot(alpha_saveC, type='l', cex.axis=2)
dev.off() 

alpha_gridsC = seq(min(alpha_saveC), max(alpha_saveC), length=200)
alpha_densC = dgamma(alpha_gridsC, a_alpha, b_alpha)

png(paste0(fig_path, "alphaC_hist.png"))
hist(alpha_saveC, main="", cex.axis=2, freq=FALSE)
lines(alpha_densC ~ alpha_gridsC, type="l", col="red")
dev.off() 

png(paste0(fig_path, "alphaT_trace.png"))
plot(alpha_saveT, type='l', cex.axis=2)
dev.off() 

alpha_gridsT = seq(min(alpha_saveT), max(alpha_saveT), length=200)
alpha_densT = dgamma(alpha_gridsT, a_alpha, b_alpha)

png(paste0(fig_path, "alphaT_hist.png"))
hist(alpha_saveT, main="", cex.axis=2, freq=FALSE)
lines(alpha_densT ~ alpha_gridsT, type="l", col="red")
dev.off() 
"""

@rput zeta_saveC zeta_saveT; 
R"""
png(paste0(fig_path, "zetaC_trace.png"))
plot(zeta_saveC, type='l', cex.axis=2)
dev.off() 

zeta_gridsC = seq(min(zeta_saveC), max(zeta_saveC), length=200)
zeta_densC = dgamma(zeta_gridsC, a_zeta, b_zeta)

png(paste0(fig_path, "zetaC_hist.png"))
hist(zeta_saveC, main="", cex.axis=2, freq=FALSE)
lines(zeta_densC ~ zeta_gridsC, type="l", col="red")
dev.off() 

png(paste0(fig_path, "zetaT_trace.png"))
plot(zeta_saveT, type='l', cex.axis=2)
dev.off() 

zeta_gridsT = seq(min(zeta_saveT), max(zeta_saveT), length=200)
zeta_densT = dgamma(zeta_gridsT, a_zeta, b_zeta)

png(paste0(fig_path, "zetaT_hist.png"))
hist(zeta_saveT, main="", cex.axis=2, freq=FALSE)
lines(zeta_densT ~ zeta_gridsT, type="l", col="red")
dev.off() 
"""


k_saveC = [length(findall(x .> 0)) for x in nl_saveC] 
k_saveT = [length(findall(x .> 0)) for x in nl_saveT] 

@rput k_saveC k_saveT 
R"""
png(paste0(fig_path, "kC_hist.png"))
hist(k_saveC, main="", xlab="kC", cex.axis=2, freq=FALSE)
dev.off() 

png(paste0(fig_path, "kT_hist.png"))
hist(k_saveT, main="", xlab="kT", cex.axis=2, freq=FALSE)
dev.off() 
"""

g_saveC = [length(findall(x .> 0)) for x in ml_saveC] 
g_saveT = [length(findall(x .> 0)) for x in ml_saveT] 

@rput g_saveC g_saveT 
R"""
png(paste0(fig_path, "gC_hist.png"))
hist(g_saveC, main="", xlab="gC", cex.axis=2, freq=FALSE)
dev.off() 

png(paste0(fig_path, "gT_hist.png"))
hist(g_saveT, main="", xlab="gT", cex.axis=2, freq=FALSE)
dev.off() 
"""



epsilon_pred = zeros(nkeep)
xi_pred = zeros(nkeep)
for i in 1:nkeep
    re = rand(MvLogNormal(zeros(2), Sigma_e_save[i,:,:]), 1)
    epsilon_pred[i] = re[1]
    xi_pred[i] = re[2]
end

c_e = hyper["c_e"]
C_e = hyper["C_e"]

epsilon_prior = zeros(nkeep)
xi_prior = zeros(nkeep)
for i in 1:nkeep
    Sigma_e_prior = rand(InverseWishart(c_e, C_e), 1)[1]
    re = rand(MvLogNormal(zeros(2), Sigma_e_prior), 1)
    epsilon_prior[i] = re[1]
    xi_prior[i] = re[2]
end 

@rput epsilon_pred xi_pred
@rput epsilon_prior xi_prior
R"""
png(paste0(fig_path, "random_effects_pred.png"), height=480, width=480)
plot(epsilon_pred ~ xi_pred, main="", xlab="xi", ylab="epsilon", cex.axis=2)
points(epsilon_prior ~ xi_prior, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
dev.off() 

png(paste0(fig_path, "random_effects_pred_recaled.png"), height=480, width=480)
plot(epsilon_pred ~ xi_pred, main="", xlab="xi", ylab="epsilon", cex.axis=2, xlim=c(0,25), ylim=c(0,25))
points(epsilon_prior ~ xi_prior, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
dev.off() 
"""

R"""
png(paste0(fig_path, "epsilon_pred_hist.png"), height=480, width=480)
hist(epsilon_pred, main="", xlab="xi", ylab="epsilon", cex.axis=2)
dev.off() 

png(paste0(fig_path, "xi_pred_hist.png"), height=480, width=480)
hist(xi_pred, main="", xlab="xi", ylab="xi", cex.axis=2)
dev.off() 

png(paste0(fig_path, "epsilon_prior_hist.png"), height=480, width=480)
hist(epsilon_prior, main="", xlab="xi", ylab="epsilon", cex.axis=2)
dev.off() 

png(paste0(fig_path, "xi_prior_hist.png"), height=480, width=480)
hist(xi_prior, main="", xlab="xi", ylab="xi", cex.axis=2)
dev.off() 
"""

survival_predC, gap_predC, arrival_predC, theta_predC, phi_predC, lambda_predC, eta_predC = predict(Sigma_e_save, theta_save, phi_save, nl_saveC, lambda_save, eta_save, ml_saveC)

survival_predT, gap_predT, arrival_predT, theta_predT, phi_predT, lambda_predT, eta_predT = predict(Sigma_e_save, theta_save, phi_save, nl_saveT, lambda_save, eta_save, ml_saveT)


@rput theta_predC phi_predC lambda_predC eta_predC
@rput theta_predT phi_predT lambda_predT eta_predT
R"""
png(paste0(fig_path, "theta_predC.png"), height=480, width=480)
hist(unlist(theta_predC), main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "phi_predC.png"), height=480, width=480)
hist(unlist(phi_predC), main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "lambda_predC.png"), height=480, width=480)
hist(unlist(lambda_predC), main="", xlab="", cex.axis=2)
dev.off() 


png(paste0(fig_path, "eta_predC.png"), height=480, width=480)
hist(unlist(eta_predC), main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "theta_predT.png"), height=480, width=480)
hist(unlist(theta_predT), main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "phi_predT.png"), height=480, width=480)
hist(unlist(phi_predT), main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "lambda_predT.png"), height=480, width=480)
hist(unlist(lambda_predT), main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "eta_predT.png"), height=480, width=480)
hist(unlist(eta_predT), main="", xlab="", cex.axis=2)
dev.off() 
"""


@rput theta_save phi_save
@rput nl_saveC nl_saveT 
R""" 
iter = 500
png(paste0(fig_path, "theta_iter500.png"))
hmax = max(max(nl_saveC[[iter]], nl_saveT[[iter]]))
plot(nl_saveC[[iter]] ~ theta_save[[iter]], type="h", col="red", ylim=c(0, hmax), xlab="theta", ylab="nl", cex.axis=2)
lines(nl_saveT[[iter]] ~ theta_save[[iter]], type="h", col="blue")
dev.off() 
png(paste0(fig_path, "phi_iter500.png"))
hmax = max(max(nl_saveC[[iter]], nl_saveT[[iter]]))
plot(nl_saveC[[iter]] ~ phi_save[[iter]], type="h", col="red", ylim=c(0, hmax), xlab="phi", ylab="nl", cex.axis=2)
lines(nl_saveT[[iter]] ~ phi_save[[iter]], type="h", col="blue")
dev.off() 

iter = 1000
png(paste0(fig_path, "theta_iter1000.png"))
hmax = max(max(nl_saveC[[iter]], nl_saveT[[iter]]))
plot(nl_saveC[[iter]] ~ theta_save[[iter]], type="h", col="red", ylim=c(0, hmax), xlab="theta", ylab="nl", cex.axis=2)
lines(nl_saveT[[iter]] ~ theta_save[[iter]], type="h", col="blue")
dev.off() 
png(paste0(fig_path, "phi_iter1000.png"))
hmax = max(max(nl_saveC[[iter]], nl_saveT[[iter]]))
plot(nl_saveC[[iter]] ~ phi_save[[iter]], type="h", col="red", ylim=c(0, hmax), xlab="phi", ylab="nl", cex.axis=2)
lines(nl_saveT[[iter]] ~ phi_save[[iter]], type="h", col="blue")
dev.off() 

iter = 1500
png(paste0(fig_path, "theta_iter1500.png"))
hmax = max(max(nl_saveC[[iter]], nl_saveT[[iter]]))
plot(nl_saveC[[iter]] ~ theta_save[[iter]], type="h", col="red", ylim=c(0, hmax), xlab="theta", ylab="nl", cex.axis=2)
lines(nl_saveT[[iter]] ~ theta_save[[iter]], type="h", col="blue")
dev.off() 
png(paste0(fig_path, "phi_iter1500.png"))
hmax = max(max(nl_saveC[[iter]], nl_saveT[[iter]]))
plot(nl_saveC[[iter]] ~ phi_save[[iter]], type="h", col="red", ylim=c(0, hmax), xlab="phi", ylab="nl", cex.axis=2)
lines(nl_saveT[[iter]] ~ phi_save[[iter]], type="h", col="blue")
dev.off() 

iter = 2000
png(paste0(fig_path, "theta_iter2000.png"))
hmax = max(max(nl_saveC[[iter]], nl_saveT[[iter]]))
plot(nl_saveC[[iter]] ~ theta_save[[iter]], type="h", col="red", ylim=c(0, hmax), xlab="theta", ylab="nl", cex.axis=2)
lines(nl_saveT[[iter]] ~ theta_save[[iter]], type="h", col="blue")
dev.off() 
png(paste0(fig_path, "phi_iter2000.png"))
hmax = max(max(nl_saveC[[iter]], nl_saveT[[iter]]))
plot(nl_saveC[[iter]] ~ phi_save[[iter]], type="h", col="red", ylim=c(0, hmax), xlab="phi", ylab="nl", cex.axis=2)
lines(nl_saveT[[iter]] ~ phi_save[[iter]], type="h", col="blue")
dev.off() 
"""


surv_grids = range(0.001, 13, length=100)

survival_posC = survival_functional_estimation(
    Sigma_e_save,
    phi_save, 
    theta_save, 
    alpha_saveC, 
    nl_saveC, 
    n_clusters, 
    a_phi, 
    b_phi_save, 
    mu_theta_save, 
    sigma2_theta, 
    surv_grids
    )
   
survival_densC = survival_posC["d"]
survival_survC = survival_posC["s"]
survival_hazaC = survival_densC ./ survival_survC 

 
survival_posT = survival_functional_estimation(
    Sigma_e_save,
    phi_save, 
    theta_save, 
    alpha_saveT, 
    nl_saveT, 
    n_clusters, 
    a_phi, 
    b_phi_save, 
    mu_theta_save, 
    sigma2_theta, 
    surv_grids
    )
   
survival_densT = survival_posT["d"]
survival_survT = survival_posT["s"]
survival_hazaT = survival_densT ./ survival_survT 

gap_grids = range(0.001, 10, length=100)
gap_res_posC = survival_functional_estimation(
    Sigma_e_save,
    eta_save, 
    lambda_save, 
    zeta_saveC, 
    ml_saveC, 
    m_clusters, 
    a_eta,
    b_eta_save, 
    mu_lambda_save, 
    sigma2_lambda, 
    gap_grids
    )

gap_densC = gap_res_posC["d"]
gap_survC = gap_res_posC["s"]
gap_hazaC = gap_densC ./ gap_survC

gap_res_posT = survival_functional_estimation(
    Sigma_e_save,
    eta_save, 
    lambda_save, 
    zeta_saveT, 
    ml_saveT, 
    m_clusters, 
    a_eta,
    b_eta_save, 
    mu_lambda_save, 
    sigma2_lambda, 
    gap_grids
    )

gap_densT = gap_res_posT["d"]
gap_survT = gap_res_posT["s"]
gap_hazaT = gap_densT ./ gap_survT

@rput gap_grids 
@rput gap_densC gap_survC gap_hazaC
@rput gap_densT gap_survT gap_hazaT
R"""
gap_dens_meanC = apply(gap_densC, 2, mean, na.rm=TRUE) 
gap_dens_quanC = apply(gap_densC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
gap_dens_meanT = apply(gap_densT, 2, mean, na.rm=TRUE) 
gap_dens_quanT = apply(gap_densT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

gap_df_densC = data.frame(
    x=gap_grids, 
    m=gap_dens_meanC, 
    l=gap_dens_quanC[1,], 
    h=gap_dens_quanC[2,]
)
gap_df_densT = data.frame(
    x=gap_grids, 
    m=gap_dens_meanT, 
    l=gap_dens_quanT[1,], 
    h=gap_dens_quanT[2,]
)

gap_p_dens = ggplot(gap_df_densC) 
gap_p_dens = gap_p_dens + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
gap_p_dens = gap_p_dens + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
gap_p_dens = gap_p_dens + geom_line(data=gap_df_densT, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
gap_p_dens = gap_p_dens + geom_ribbon(data=gap_df_densT, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
gap_p_dens = gap_p_dens + ylab("Density") + xlab("t") + ylim(0,2)
ggsave(paste0(fig_path, "gap_density.png"), gap_p_dens)
"""
    


@rput surv_grids
@rput survival_densC survival_survC survival_hazaC
@rput survival_densT survival_survT survival_hazaT
R"""
library(ggplot2)
survival_dens_meanC = apply(survival_densC, 2, mean, na.rm=TRUE) 
survival_dens_quanC = apply(survival_densC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_dens_meanT = apply(survival_densT, 2, mean, na.rm=TRUE) 
survival_dens_quanT = apply(survival_densT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_densC = data.frame(
    x=surv_grids, 
    m=survival_dens_meanC, 
    l=survival_dens_quanC[1,], 
    h=survival_dens_quanC[2,] 
    )
survival_df_densT = data.frame(
    x=surv_grids, 
    m=survival_dens_meanT, 
    l=survival_dens_quanT[1,], 
    h=survival_dens_quanT[2,] 
    )

survival_p_dens = ggplot(survival_df_densC) 
survival_p_dens = survival_p_dens + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_dens = survival_p_dens + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) 
survival_p_dens = survival_p_dens + geom_line(data=survival_df_densT, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
survival_p_dens = survival_p_dens + geom_ribbon(data=survival_df_densT, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
survival_p_dens = survival_p_dens + ylab("Density") + xlab("t")+ theme_bw(base_size=25) 
ggsave(paste0(fig_path, "survival_density.png"), survival_p_dens)
"""

R"""
survival_surv_meanC = apply(survival_survC, 2, mean, na.rm=TRUE) 
survival_surv_quanC = apply(survival_survC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_surv_meanT = apply(survival_survT, 2, mean, na.rm=TRUE) 
survival_surv_quanT = apply(survival_survT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_survC = data.frame(
    x=surv_grids, 
    m=survival_surv_meanC, 
    l=survival_surv_quanC[1,], 
    h=survival_surv_quanC[2,]
    )

survival_df_survT = data.frame(
    x=surv_grids, 
    m=survival_surv_meanT, 
    l=survival_surv_quanT[1,], 
    h=survival_surv_quanT[2,]
    )
survival_p_surv = ggplot(survival_df_survC) 
survival_p_surv = survival_p_surv + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_surv = survival_p_surv + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
survival_p_surv = survival_p_surv + geom_line(data=survival_df_survT, aes(x=x,y=m), color="blue",  linetype="dashed", size=1.5)
survival_p_surv = survival_p_surv + geom_ribbon(data=survival_df_survT, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
survival_p_surv = survival_p_surv + ylab("Survival") + xlab("t")
ggsave(paste0(fig_path, "survival_survival.png"), survival_p_surv)
"""



# @rput gap_grids 
# @rput gap_densC gap_survC gap_hazaC
# @rput gap_densT gap_survT gap_hazaT
# R"""
# gap_dens_meanC = apply(gap_densC, 2, mean, na.rm=TRUE) 
# gap_dens_quanC = apply(gap_densC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# gap_dens_meanT = apply(gap_densT, 2, mean, na.rm=TRUE) 
# gap_dens_quanT = apply(gap_densT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

# gap_df_densC = data.frame(
#     x=gap_grids, 
#     m=gap_dens_meanC, 
#     l=gap_dens_quanC[1,], 
#     h=gap_dens_quanC[2,]
# )
# gap_df_densT = data.frame(
#     x=gap_grids, 
#     m=gap_dens_meanT, 
#     l=gap_dens_quanT[1,], 
#     h=gap_dens_quanT[2,]
# )

# gap_p_dens = ggplot(gap_df_densC) 
# gap_p_dens = gap_p_dens + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
# gap_p_dens = gap_p_dens + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
# gap_p_dens = gap_p_dens + geom_line(data=gap_df_densT, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
# gap_p_dens = gap_p_dens + geom_ribbon(data=gap_df_densT, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
# gap_p_dens = gap_p_dens + ylab("Density") + xlab("t") + ylim(0,2)
# ggsave(paste0(fig_path, "gap_density.png"), gap_p_dens)
# """


grids1 = range(0.5, 20, length=100)
grids2 = range(1, 20, length=100)
grids3 = range(2, 20, length=100)
grids4 = range(5, 20, length=100)

cond_probC_1_N0, cond_probC_1_N1 = conditional_survival_probability(
    Sigma_e_save,
    a_eta,
    eta_save,
    lambda_save,
    zeta_saveC,
    ml_saveC,
    b_eta_save,
    mu_lambda_save,
    sigma2_lambda,
    phi_save,
    theta_save,
    alpha_saveC,
    nl_saveC,
    a_phi,
    b_phi_save,
    mu_theta_save,
    sigma2_theta,
    grids1 
)

cond_probC_2_N0, cond_probC_2_N1 = conditional_survival_probability(
    Sigma_e_save,
    a_eta,
    eta_save,
    lambda_save,
    zeta_saveC,
    ml_saveC,
    b_eta_save,
    mu_lambda_save,
    sigma2_lambda,
    phi_save,
    theta_save,
    alpha_saveC,
    nl_saveC,
    a_phi,
    b_phi_save,
    mu_theta_save,
    sigma2_theta,
    grids2 
)

cond_probC_3_N0, cond_probC_3_N1 = conditional_survival_probability(
    Sigma_e_save,
    a_eta,
    eta_save,
    lambda_save,
    zeta_saveC,
    ml_saveC,
    b_eta_save,
    mu_lambda_save,
    sigma2_lambda,
    phi_save,
    theta_save,
    alpha_saveC,
    nl_saveC,
    a_phi,
    b_phi_save,
    mu_theta_save,
    sigma2_theta,
    grids3
)

cond_probC_4_N0, cond_probC_4_N1 = conditional_survival_probability(
    Sigma_e_save,
    a_eta,
    eta_save,
    lambda_save,
    zeta_saveC,
    ml_saveC,
    b_eta_save,
    mu_lambda_save,
    sigma2_lambda,
    phi_save,
    theta_save,
    alpha_saveC,
    nl_saveC,
    a_phi,
    b_phi_save,
    mu_theta_save,
    sigma2_theta,
    grids4 
)



cond_probT_1_N0, cond_probT_1_N1 = conditional_survival_probability(
    Sigma_e_save,
    a_eta,
    eta_save,
    lambda_save,
    zeta_saveT,
    ml_saveT,
    b_eta_save,
    mu_lambda_save,
    sigma2_lambda,
    phi_save,
    theta_save,
    alpha_saveT,
    nl_saveT,
    a_phi,
    b_phi_save,
    mu_theta_save,
    sigma2_theta,
    grids1 
)

cond_probT_2_N0, cond_probT_2_N1 = conditional_survival_probability(
    Sigma_e_save,
    a_eta,
    eta_save,
    lambda_save,
    zeta_saveT,
    ml_saveT,
    b_eta_save,
    mu_lambda_save,
    sigma2_lambda,
    phi_save,
    theta_save,
    alpha_saveT,
    nl_saveT,
    a_phi,
    b_phi_save,
    mu_theta_save,
    sigma2_theta,
    grids2 
)

cond_probT_3_N0, cond_probT_3_N1 = conditional_survival_probability(
    Sigma_e_save,
    a_eta,
    eta_save,
    lambda_save,
    zeta_saveT,
    ml_saveT,
    b_eta_save,
    mu_lambda_save,
    sigma2_lambda,
    phi_save,
    theta_save,
    alpha_saveT,
    nl_saveT,
    a_phi,
    b_phi_save,
    mu_theta_save,
    sigma2_theta,
    grids3
)

cond_probT_4_N0, cond_probT_4_N1 = conditional_survival_probability(
    Sigma_e_save,
    a_eta,
    eta_save,
    lambda_save,
    zeta_saveT,
    ml_saveT,
    b_eta_save,
    mu_lambda_save,
    sigma2_lambda,
    phi_save,
    theta_save,
    alpha_saveT,
    nl_saveT,
    a_phi,
    b_phi_save,
    mu_theta_save,
    sigma2_theta,
    grids4 
)



@rput grids1 cond_probC_1_N0 cond_probT_1_N0
@rput grids2 cond_probC_2_N0 cond_probT_2_N0
@rput grids3 cond_probC_3_N0 cond_probT_3_N0
@rput grids4 cond_probC_4_N0 cond_probT_4_N0
R"""
cond_meanC_1 = apply(cond_probC_1_N0, 2, mean, na.rm=TRUE) 
cond_quanC_1 = apply(cond_probC_1_N0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond_meanT_1 = apply(cond_probT_1_N0, 2, mean, na.rm=TRUE) 
cond_quanT_1 = apply(cond_probT_1_N0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_1 = data.frame(
    x=grids1, 
    m=cond_meanC_1, 
    l=cond_quanC_1[1,], 
    h=cond_quanC_1[2,]
)

df_condT_1 = data.frame(
    x=grids1, 
    m=cond_meanT_1, 
    l=cond_quanT_1[1,], 
    h=cond_quanT_1[2,]
)

p_cond_1 = ggplot(df_condC_1) 
p_cond_1 = p_cond_1 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_1 = p_cond_1 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_1 = p_cond_1 + geom_line(data=df_condT_1, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_1 = p_cond_1 + geom_ribbon(data=df_condT_1, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_1 = p_cond_1 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_1.png"), p_cond_1)
"""

R"""
cond_meanC_2 = apply(cond_probC_2_N0, 2, mean, na.rm=TRUE) 
cond_quanC_2 = apply(cond_probC_2_N0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond_meanT_2 = apply(cond_probT_2_N0, 2, mean, na.rm=TRUE) 
cond_quanT_2 = apply(cond_probT_2_N0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_2 = data.frame(
    x=grids2, 
    m=cond_meanC_2, 
    l=cond_quanC_2[1,], 
    h=cond_quanC_2[2,]
)

df_condT_2 = data.frame(
    x=grids2, 
    m=cond_meanT_2, 
    l=cond_quanT_2[1,], 
    h=cond_quanT_2[2,]
)

p_cond_2 = ggplot(df_condC_2) 
p_cond_2 = p_cond_2 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_2 = p_cond_2 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_2 = p_cond_2 + geom_line(data=df_condT_2, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_2 = p_cond_2 + geom_ribbon(data=df_condT_2, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_2 = p_cond_2 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_2.png"), p_cond_2)
"""

R"""
cond_meanC_3 = apply(cond_probC_3_N0, 2, mean, na.rm=TRUE) 
cond_quanC_3 = apply(cond_probC_3_N0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond_meanT_3 = apply(cond_probT_3_N0, 2, mean, na.rm=TRUE) 
cond_quanT_3 = apply(cond_probT_3_N0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_3 = data.frame(
    x=grids3, 
    m=cond_meanC_3, 
    l=cond_quanC_3[1,], 
    h=cond_quanC_3[2,]
)

df_condT_3 = data.frame(
    x=grids3, 
    m=cond_meanT_3, 
    l=cond_quanT_3[1,], 
    h=cond_quanT_3[2,]
)

p_cond_3 = ggplot(df_condC_3) 
p_cond_3 = p_cond_3 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_3 = p_cond_3 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_3 = p_cond_3 + geom_line(data=df_condT_3, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_3 = p_cond_3 + geom_ribbon(data=df_condT_3, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_3 = p_cond_3 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_3.png"), p_cond_3)
"""

R"""
cond_meanC_4 = apply(cond_probC_4_N0, 2, mean, na.rm=TRUE) 
cond_quanC_4 = apply(cond_probC_4_N0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond_meanT_4 = apply(cond_probT_4_N0, 2, mean, na.rm=TRUE) 
cond_quanT_4 = apply(cond_probT_4_N0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_4 = data.frame(
    x=grids4, 
    m=cond_meanC_4, 
    l=cond_quanC_4[1,], 
    h=cond_quanC_4[2,]
)

df_condT_4 = data.frame(
    x=grids4, 
    m=cond_meanT_4, 
    l=cond_quanT_4[1,], 
    h=cond_quanT_4[2,]
)

p_cond_4 = ggplot(df_condC_4) 
p_cond_4 = p_cond_4 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_4 = p_cond_4 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_4 = p_cond_4 + geom_line(data=df_condT_4, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_4 = p_cond_4 + geom_ribbon(data=df_condT_4, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_4 = p_cond_4 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_4.png"), p_cond_4)
"""
