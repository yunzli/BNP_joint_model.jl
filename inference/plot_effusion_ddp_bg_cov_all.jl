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

# config = TOML.parsefile("../configs/effusion_ddp_cov_all_BG_s1.TOML")
config = TOML.parsefile("../configs/effusion_ddp_cov_all_BG.TOML")

fig_path = config["fig_path"]
if !isdir(fig_path)
    print(fig_path,"\n")
	mkdir(fig_path)
end
@rput fig_path 


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

BG, BH = hyper["BG"], hyper["BH"]

nsam = length(pos["alpha"])
nburn = div(nsam, 2)
nthin = div(nsam-nburn,2000)
keep_index = [nburn+1:nthin:nsam;]
nkeep = length(keep_index)

alpha_save = pos["alpha"][keep_index]
alpha0_save = pos["alpha0"][keep_index]
zeta_save = pos["zeta"][keep_index]
zeta0_save = pos["zeta0"][keep_index]

a0, b0 = hyper["a0"], hyper["b0"]
c0 = hyper["c0"]
alpha0_prior = rand(Beta(a0, b0), 2000)
alpha_prior = zeros(2000)
for i in 1:2000
    alpha_prior[i] = rand(Pareto(c0, alpha0_prior[i]), 1)[1]
end

G_corr_save = G_corr_calc(alpha_save, alpha0_save)
@rput G_corr_save
R"""
png(paste0(fig_path, "G_corr.png"))
hist(G_corr_save, cex.axis=2)
dev.off()
"""

a1, b1 = hyper["a1"], hyper["b1"]
c1 = hyper["c1"]
zeta0_prior = rand(Beta(a1, b1), 2000)
zeta_prior = zeros(2000)
for i in 1:2000
    zeta_prior[i] = rand(Pareto(c1, zeta0_prior[i]), 1)[1]
end

H_corr_save = G_corr_calc(zeta_save, zeta0_save)
@rput H_corr_save
R"""
png(paste0(fig_path, "H_corr.png"))
hist(H_corr_save, cex.axis=2)
dev.off()
"""

@rput alpha0_prior alpha_prior
@rput zeta0_prior zeta_prior
@rput alpha0_save alpha_save zeta0_save zeta_save
R"""
library(ggplot2)
png(paste0(fig_path, "alpha0_alpha.png"))
plot(alpha0_save ~ alpha_save, cex.axis=2, ylim=c(0,1), xlim=c(0,max(alpha_save)))
points(alpha0_prior ~ alpha_prior, col=rgb(red=0, green=0, blue=1, alpha=0.5))
dev.off()

png(paste0(fig_path, "alpha0_trace.png"))
plot(alpha0_save, type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path, "alpha0_hist.png"))
hist(alpha0_save, main="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "alpha_trace.png"))
plot(alpha_save, type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path, "zeta0_zeta.png"))
plot(zeta0_save ~ zeta_save, cex.axis=2, ylim=c(0,1), xlim=c(0,max(zeta_save)))
points(zeta0_prior ~ zeta_prior, col=rgb(red=0, green=0, blue=1, alpha=0.5))
dev.off()

png(paste0(fig_path, "zeta0_trace.png"))
plot(zeta0_save, type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path, "zeta0_hist.png"))
hist(zeta0_save, main="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "zeta_trace.png"))
plot(zeta_save, type='l', cex.axis=2)
dev.off( )
"""

LC_save = pos["LC"][keep_index,:]
LT_save = pos["LT"][keep_index,:]

logpC_save = pos["logpC"][keep_index,:]
logpT_save = pos["logpT"][keep_index,:]

mu_theta_save = pos["mu_theta"][keep_index]
mu_beta_save = pos["mu_beta"][keep_index,:]
b_phi_save = pos["b_phi"][keep_index]

s_theta, S_theta = hyper["s_theta"], hyper["S_theta"]
s_beta, S_beta = hyper["s_beta"], hyper["S_beta"]
r_phi, R_phi = hyper["r_phi"], hyper["R_phi"]
@rput s_theta S_theta
@rput s_beta S_beta
@rput r_phi R_phi
@rput mu_theta_save mu_beta_save b_phi_save
R"""
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
"""

R"""
for(i in 1:6){
    png(paste0(fig_path, "mu_beta_", i, "_trace.png"))
    plot(mu_beta_save[,i], type='l', cex.axis=2)
    dev.off() 

    mu_beta_i_grids = seq(min(mu_beta_save[,i]), max(mu_beta_save[,i]), length=200)
    mu_beta_i_dens = dnorm(mu_beta_i_grids, s_beta[i], sqrt(S_beta[i,i]))

    png(paste0(fig_path,"mu_beta_", i, "_hist.png"))
    hist(mu_beta_save[,i], main="", cex.axis=2, freq=FALSE)
    lines(mu_beta_i_dens ~ mu_beta_i_grids, type="l", col="red")
    dev.off() 
}
"""

@rput logpC_save logpT_save BG
R"""
png(paste0(fig_path, "logpC_BG.png"))
hist(exp(logpC_save[, BG]), cex.axis=2, ylab="pC_BG")
dev.off()

png(paste0(fig_path, "logpT_BG.png"))
hist(exp(logpT_save[, BG]), cex.axis=2, ylab="pT_BG")
dev.off()
"""

tUC_save = pos["tUC"][keep_index,:]
tUT_save = pos["tUT"][keep_index,:]

logomegaC_save = pos["logomegaC"][keep_index,:]
logomegaT_save = pos["logomegaT"][keep_index,:]

mu_lambda_save = pos["mu_lambda"][keep_index]
mu_gamma_save = pos["mu_gamma"][keep_index,:]
b_eta_save = pos["b_eta"][keep_index]

s_lambda, S_lambda = hyper["s_lambda"], hyper["S_lambda"]
s_gamma, S_gamma = hyper["s_gamma"], hyper["S_gamma"]
r_eta, R_eta = hyper["r_eta"], hyper["R_eta"]
@rput s_lambda S_lambda
@rput s_gamma S_gamma
@rput r_eta R_eta
@rput mu_lambda_save mu_gamma_save b_eta_save
R"""
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

R"""
for(i in 1:6){
    png(paste0(fig_path, "mu_gamma_", i, "_trace.png"))
    plot(mu_gamma_save[,i], type='l', cex.axis=2)
    dev.off() 

    mu_gamma_i_grids = seq(min(mu_gamma_save[,i]), max(mu_gamma_save[,i]), length=200)
    mu_gamma_i_dens = dnorm(mu_gamma_i_grids, s_gamma[i], sqrt(S_gamma[i,i]))

    png(paste0(fig_path,"mu_gamma_", i, "_hist.png"))
    hist(mu_gamma_save[,i], main="", cex.axis=2, freq=FALSE)
    lines(mu_gamma_i_dens ~ mu_gamma_i_grids, type="l", col="red")
    dev.off() 
}
"""

@rput logomegaC_save logomegaT_save BG
R"""
png(paste0(fig_path, "logomegaC_BG.png"))
hist(exp(logomegaC_save[, BG]), cex.axis=2, ylab="omegaC_BG")
dev.off()

png(paste0(fig_path, "logomegaT_BG.png"))
hist(exp(logomegaT_save[, BG]), cex.axis=2, ylab="omegaT_BG")
dev.off()
"""

Sigma_e_1_save = pos["Sigma_e_1"][keep_index,:,:]
Sigma_e_2_save = pos["Sigma_e_2"][keep_index,:,:]

@rput Sigma_e_1_save Sigma_e_2_save
R"""
png(paste0(fig_path, "Sigma_e_trace.png"))
par(mfrow=c(2,3))
plot(Sigma_e_1_save[,1,1], type='l')
plot(Sigma_e_1_save[,1,2], type='l')
plot(Sigma_e_1_save[,2,2], type='l')
plot(Sigma_e_2_save[,1,1], type='l')
plot(Sigma_e_2_save[,1,2], type='l')
plot(Sigma_e_2_save[,2,2], type='l')
dev.off() 

png(paste0(fig_path, "Sigma_e_hist.png"))
par(mfrow=c(2,3))
hist(Sigma_e_1_save[,1,1])
hist(Sigma_e_1_save[,1,2])
hist(Sigma_e_1_save[,2,2])
hist(Sigma_e_2_save[,1,1])
hist(Sigma_e_2_save[,1,2])
hist(Sigma_e_2_save[,2,2])
dev.off()
"""


epsilon_predC = zeros(nkeep)
xi_predC = zeros(nkeep)
for i in 1:nkeep
    re = rand(MvLogNormal(zeros(2), Sigma_e_1_save[i,:,:]), 1)
    epsilon_predC[i] = re[1]
    xi_predC[i] = re[2]
end

epsilon_predT = zeros(nkeep)
xi_predT = zeros(nkeep)
for i in 1:nkeep
    re = rand(MvLogNormal(zeros(2), Sigma_e_2_save[i,:,:]), 1)
    epsilon_predT[i] = re[1]
    xi_predT[i] = re[2]
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


@rput epsilon_prior xi_prior
@rput epsilon_predC xi_predC
@rput epsilon_predT xi_predT
R"""
png(paste0(fig_path, "random_effectsC.png"))
plot(epsilon_predC ~ xi_predC, cex.axis=2, xlim=c(0,5), ylim=c(0,15))
points(epsilon_prior ~ xi_prior, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
dev.off()

png(paste0(fig_path, "random_effectsT.png"))
plot(epsilon_predT ~ xi_predT, cex.axis=2, xlim=c(0,5), ylim=c(0,15))
points(epsilon_prior ~ xi_prior, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
dev.off()
"""



theta_save = pos["theta"][keep_index,:]
beta_save = pos["beta"][keep_index,:,:]
phi_save = pos["phi"][keep_index,:]

surv_grids = range(0.001, 13, length=100)

x0_C = [0, 1, 1, 1, 1, 1]
x0_T = [1, 1, 1, 1, 1, 1]
x1_C = [0, 0, 1, 1, 1, 1]
x1_T = [1, 0, 1, 1, 1, 1]
x2_C = [0, 1, 0, 1, 1, 1]
x2_T = [1, 1, 0, 1, 1, 1]
x3_C = [0, 1, 1, 0, 1, 1]
x3_T = [1, 1, 1, 0, 1, 1]
x4_C = [0, 1, 1, 1, 1, 0]
x4_T = [1, 1, 1, 1, 1, 0]

surv_res_0 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,1,1], 
    Sigma_e_2_save[:,1,1],
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x0_C,
    x0_T,
    surv_grids,
    BG
    )

survival_densC_0 = surv_res_0["densC"]
survival_survC_0 = surv_res_0["survC"]
survival_densT_0 = surv_res_0["densT"]
survival_survT_0 = surv_res_0["survT"]

surv_res_1 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,1,1], 
    Sigma_e_2_save[:,1,1],
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x1_C,
    x1_T,
    surv_grids,
    BG
    )

survival_densC_1 = surv_res_1["densC"]
survival_survC_1 = surv_res_1["survC"]
survival_densT_1 = surv_res_1["densT"]
survival_survT_1 = surv_res_1["survT"]

surv_res_2 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,1,1], 
    Sigma_e_2_save[:,1,1],
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x2_C,
    x2_T,
    surv_grids,
    BG
    )

survival_densC_2 = surv_res_2["densC"]
survival_survC_2 = surv_res_2["survC"]
survival_densT_2 = surv_res_2["densT"]
survival_survT_2 = surv_res_2["survT"]

surv_res_3 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,1,1], 
    Sigma_e_2_save[:,1,1],
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x3_C,
    x3_T,
    surv_grids,
    BG
    )

survival_densC_3 = surv_res_3["densC"]
survival_survC_3 = surv_res_3["survC"]
survival_densT_3 = surv_res_3["densT"]
survival_survT_3 = surv_res_3["survT"]

surv_res_4 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,1,1], 
    Sigma_e_2_save[:,1,1],
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x4_C,
    x4_T,
    surv_grids,
    BG
    )

survival_densC_4 = surv_res_4["densC"]
survival_survC_4 = surv_res_4["survC"]
survival_densT_4 = surv_res_4["densT"]
survival_survT_4 = surv_res_4["survT"]

gap_grids = range(0.001, 10, length=100)

lambda_save = pos["lambda"][keep_index,:]
gamma_save = pos["gamma"][keep_index,:,:]
eta_save = pos["eta"][keep_index,:]

z0_C = [0, 1, 1, 1, 1, 1]
z0_T = [1, 1, 1, 1, 1, 1]
z1_C = [0, 0, 1, 1, 1, 1]
z1_T = [1, 0, 1, 1, 1, 1]
z2_C = [0, 1, 0, 1, 1, 1]
z2_T = [1, 1, 0, 1, 1, 1]
z3_C = [0, 1, 1, 0, 1, 1]
z3_T = [1, 1, 1, 0, 1, 1]
z4_C = [0, 1, 1, 1, 1, 0]
z4_T = [1, 1, 1, 1, 1, 0]

gap_res_0 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,2,2], 
    Sigma_e_2_save[:,2,2],
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z0_C,
    z0_T,
    gap_grids,
    BH
    )

gap_densC_0 = gap_res_0["densC"]
gap_survC_0 = gap_res_0["survC"]
gap_densT_0 = gap_res_0["densT"]
gap_survT_0 = gap_res_0["survT"]

gap_res_1 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,2,2], 
    Sigma_e_2_save[:,2,2],
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z1_C,
    z1_T,
    gap_grids,
    BH
    )

gap_densC_1 = gap_res_1["densC"]
gap_survC_1 = gap_res_1["survC"]
gap_densT_1 = gap_res_1["densT"]
gap_survT_1 = gap_res_1["survT"]

gap_res_2 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,2,2], 
    Sigma_e_2_save[:,2,2],
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z2_C,
    z2_T,
    gap_grids,
    BH
    )

gap_densC_2 = gap_res_2["densC"]
gap_survC_2 = gap_res_2["survC"]
gap_densT_2 = gap_res_2["densT"]
gap_survT_2 = gap_res_2["survT"]

gap_res_3 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,2,2], 
    Sigma_e_2_save[:,2,2],
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z3_C,
    z3_T,
    gap_grids,
    BH
    )

gap_densC_3 = gap_res_3["densC"]
gap_survC_3 = gap_res_3["survC"]
gap_densT_3 = gap_res_3["densT"]
gap_survT_3 = gap_res_3["survT"]

gap_res_4 = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,2,2], 
    Sigma_e_2_save[:,2,2],
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z4_C,
    z4_T,
    gap_grids,
    BH
    )

gap_densC_4 = gap_res_4["densC"]
gap_survC_4 = gap_res_4["survC"]
gap_densT_4 = gap_res_4["densT"]
gap_survT_4 = gap_res_4["survT"]

@rput surv_grids
@rput survival_densC_0 survival_survC_0
@rput survival_densT_0 survival_survT_0
R"""
library(ggplot2)
survival_dens_meanC_0 = apply(survival_densC_0, 2, mean, na.rm=TRUE) 
survival_dens_quanC_0 = apply(survival_densC_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_dens_meanT_0 = apply(survival_densT_0, 2, mean, na.rm=TRUE) 
survival_dens_quanT_0 = apply(survival_densT_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_densC_0 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanC_0, 
    l=survival_dens_quanC_0[1,], 
    h=survival_dens_quanC_0[2,] 
    )
survival_df_densT_0 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanT_0, 
    l=survival_dens_quanT_0[1,], 
    h=survival_dens_quanT_0[2,] 
    )

survival_p_dens_0 = ggplot(survival_df_densC_0) 
survival_p_dens_0 = survival_p_dens_0 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_dens_0 = survival_p_dens_0 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) 
survival_p_dens_0 = survival_p_dens_0 + geom_line(data=survival_df_densT_0, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
survival_p_dens_0 = survival_p_dens_0 + geom_ribbon(data=survival_df_densT_0, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
survival_p_dens_0 = survival_p_dens_0 + ylab("Density") + xlab("t")+ theme_bw(base_size=25) 
ggsave(paste0(fig_path, "survival_density_0.png"), survival_p_dens_0)
"""

R"""
survival_surv_meanC_0 = apply(survival_survC_0, 2, mean, na.rm=TRUE) 
survival_surv_quanC_0 = apply(survival_survC_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_surv_meanT_0 = apply(survival_survT_0, 2, mean, na.rm=TRUE) 
survival_surv_quanT_0 = apply(survival_survT_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_survC_0 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanC_0, 
    l=survival_surv_quanC_0[1,], 
    h=survival_surv_quanC_0[2,]
    )

survival_df_survT_0 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanT_0, 
    l=survival_surv_quanT_0[1,], 
    h=survival_surv_quanT_0[2,]
    )
survival_p_surv_0 = ggplot(survival_df_survC_0) 
survival_p_surv_0 = survival_p_surv_0 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_surv_0 = survival_p_surv_0 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
survival_p_surv_0 = survival_p_surv_0 + geom_line(data=survival_df_survT_0, aes(x=x,y=m), color="blue",  linetype="dashed", size=1.5)
survival_p_surv_0 = survival_p_surv_0 + geom_ribbon(data=survival_df_survT_0, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
survival_p_surv_0 = survival_p_surv_0 + ylab("Survival") + xlab("t") 
ggsave(paste0(fig_path, "survival_survival_0.png"), survival_p_surv_0)
"""


@rput survival_densC_1 survival_survC_1
@rput survival_densT_1 survival_survT_1
R"""
library(ggplot2)
survival_dens_meanC_1 = apply(survival_densC_1, 2, mean, na.rm=TRUE) 
survival_dens_quanC_1 = apply(survival_densC_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_dens_meanT_1 = apply(survival_densT_1, 2, mean, na.rm=TRUE) 
survival_dens_quanT_1 = apply(survival_densT_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_densC_1 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanC_1, 
    l=survival_dens_quanC_1[1,], 
    h=survival_dens_quanC_1[2,] 
    )
survival_df_densT_1 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanT_1, 
    l=survival_dens_quanT_1[1,], 
    h=survival_dens_quanT_1[2,] 
    )

survival_p_dens_1 = ggplot(survival_df_densC_1) 
survival_p_dens_1 = survival_p_dens_1 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_dens_1 = survival_p_dens_1 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) 
survival_p_dens_1 = survival_p_dens_1 + geom_line(data=survival_df_densT_1, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
survival_p_dens_1 = survival_p_dens_1 + geom_ribbon(data=survival_df_densT_1, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
survival_p_dens_1 = survival_p_dens_1 + ylab("Density") + xlab("t")+ theme_bw(base_size=25) 
ggsave(paste0(fig_path, "survival_density_1.png"), survival_p_dens_1)
"""

R"""
survival_surv_meanC_1 = apply(survival_survC_1, 2, mean, na.rm=TRUE) 
survival_surv_quanC_1 = apply(survival_survC_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_surv_meanT_1 = apply(survival_survT_1, 2, mean, na.rm=TRUE) 
survival_surv_quanT_1 = apply(survival_survT_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_survC_1 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanC_1, 
    l=survival_surv_quanC_1[1,], 
    h=survival_surv_quanC_1[2,]
    )

survival_df_survT_1 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanT_1, 
    l=survival_surv_quanT_1[1,], 
    h=survival_surv_quanT_1[2,]
    )
survival_p_surv_1 = ggplot(survival_df_survC_1) 
survival_p_surv_1 = survival_p_surv_1 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_surv_1 = survival_p_surv_1 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
survival_p_surv_1 = survival_p_surv_1 + geom_line(data=survival_df_survT_1, aes(x=x,y=m), color="blue",  linetype="dashed", size=1.5)
survival_p_surv_1 = survival_p_surv_1 + geom_ribbon(data=survival_df_survT_1, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
survival_p_surv_1 = survival_p_surv_1 + ylab("Survival") + xlab("t") 
ggsave(paste0(fig_path, "survival_survival_1.png"), survival_p_surv_1)
"""

@rput survival_densC_2 survival_survC_2
@rput survival_densT_2 survival_survT_2
R"""
library(ggplot2)
survival_dens_meanC_2 = apply(survival_densC_2, 2, mean, na.rm=TRUE) 
survival_dens_quanC_2 = apply(survival_densC_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_dens_meanT_2 = apply(survival_densT_2, 2, mean, na.rm=TRUE) 
survival_dens_quanT_2 = apply(survival_densT_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_densC_2 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanC_2, 
    l=survival_dens_quanC_2[1,], 
    h=survival_dens_quanC_2[2,] 
    )
survival_df_densT_2 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanT_2, 
    l=survival_dens_quanT_2[1,], 
    h=survival_dens_quanT_2[2,] 
    )

survival_p_dens_2 = ggplot(survival_df_densC_2) 
survival_p_dens_2 = survival_p_dens_2 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_dens_2 = survival_p_dens_2 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) 
survival_p_dens_2 = survival_p_dens_2 + geom_line(data=survival_df_densT_2, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
survival_p_dens_2 = survival_p_dens_2 + geom_ribbon(data=survival_df_densT_2, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
survival_p_dens_2 = survival_p_dens_2 + ylab("Density") + xlab("t")+ theme_bw(base_size=25) 
ggsave(paste0(fig_path, "survival_density_2.png"), survival_p_dens_2)
"""

R"""
survival_surv_meanC_2 = apply(survival_survC_2, 2, mean, na.rm=TRUE) 
survival_surv_quanC_2 = apply(survival_survC_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_surv_meanT_2 = apply(survival_survT_2, 2, mean, na.rm=TRUE) 
survival_surv_quanT_2 = apply(survival_survT_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_survC_2 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanC_2, 
    l=survival_surv_quanC_2[1,], 
    h=survival_surv_quanC_2[2,]
    )

survival_df_survT_2 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanT_2, 
    l=survival_surv_quanT_2[1,], 
    h=survival_surv_quanT_2[2,]
    )
survival_p_surv_2 = ggplot(survival_df_survC_2) 
survival_p_surv_2 = survival_p_surv_2 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_surv_2 = survival_p_surv_2 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
survival_p_surv_2 = survival_p_surv_2 + geom_line(data=survival_df_survT_2, aes(x=x,y=m), color="blue",  linetype="dashed", size=1.5)
survival_p_surv_2 = survival_p_surv_2 + geom_ribbon(data=survival_df_survT_2, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
survival_p_surv_2 = survival_p_surv_2 + ylab("Survival") + xlab("t") 
ggsave(paste0(fig_path, "survival_survival_2.png"), survival_p_surv_2)
"""

@rput survival_densC_3 survival_survC_3
@rput survival_densT_3 survival_survT_3
R"""
library(ggplot2)
survival_dens_meanC_3 = apply(survival_densC_3, 2, mean, na.rm=TRUE) 
survival_dens_quanC_3 = apply(survival_densC_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_dens_meanT_3 = apply(survival_densT_3, 2, mean, na.rm=TRUE) 
survival_dens_quanT_3 = apply(survival_densT_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_densC_3 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanC_3, 
    l=survival_dens_quanC_3[1,], 
    h=survival_dens_quanC_3[2,] 
    )
survival_df_densT_3 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanT_3, 
    l=survival_dens_quanT_3[1,], 
    h=survival_dens_quanT_3[2,] 
    )

survival_p_dens_3 = ggplot(survival_df_densC_3) 
survival_p_dens_3 = survival_p_dens_3 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_dens_3 = survival_p_dens_3 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) 
survival_p_dens_3 = survival_p_dens_3 + geom_line(data=survival_df_densT_3, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
survival_p_dens_3 = survival_p_dens_3 + geom_ribbon(data=survival_df_densT_3, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
survival_p_dens_3 = survival_p_dens_3 + ylab("Density") + xlab("t")+ theme_bw(base_size=25) 
ggsave(paste0(fig_path, "survival_density_3.png"), survival_p_dens_3)
"""

R"""
survival_surv_meanC_3 = apply(survival_survC_3, 2, mean, na.rm=TRUE) 
survival_surv_quanC_3 = apply(survival_survC_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_surv_meanT_3 = apply(survival_survT_3, 2, mean, na.rm=TRUE) 
survival_surv_quanT_3 = apply(survival_survT_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_survC_3 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanC_3, 
    l=survival_surv_quanC_3[1,], 
    h=survival_surv_quanC_3[2,]
    )

survival_df_survT_3 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanT_3, 
    l=survival_surv_quanT_3[1,], 
    h=survival_surv_quanT_3[2,]
    )
survival_p_surv_3 = ggplot(survival_df_survC_3) 
survival_p_surv_3 = survival_p_surv_3 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_surv_3 = survival_p_surv_3 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
survival_p_surv_3 = survival_p_surv_3 + geom_line(data=survival_df_survT_3, aes(x=x,y=m), color="blue",  linetype="dashed", size=1.5)
survival_p_surv_3 = survival_p_surv_3 + geom_ribbon(data=survival_df_survT_3, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
survival_p_surv_3 = survival_p_surv_3 + ylab("Survival") + xlab("t") 
ggsave(paste0(fig_path, "survival_survival_3.png"), survival_p_surv_3)
"""

@rput survival_densC_4 survival_survC_4
@rput survival_densT_4 survival_survT_4
R"""
library(ggplot2)
survival_dens_meanC_4 = apply(survival_densC_4, 2, mean, na.rm=TRUE) 
survival_dens_quanC_4 = apply(survival_densC_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_dens_meanT_4 = apply(survival_densT_4, 2, mean, na.rm=TRUE) 
survival_dens_quanT_4 = apply(survival_densT_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_densC_4 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanC_4, 
    l=survival_dens_quanC_4[1,], 
    h=survival_dens_quanC_4[2,] 
    )
survival_df_densT_4 = data.frame(
    x=surv_grids, 
    m=survival_dens_meanT_4, 
    l=survival_dens_quanT_4[1,], 
    h=survival_dens_quanT_4[2,] 
    )

survival_p_dens_4 = ggplot(survival_df_densC_4) 
survival_p_dens_4 = survival_p_dens_4 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_dens_4 = survival_p_dens_4 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) 
survival_p_dens_4 = survival_p_dens_4 + geom_line(data=survival_df_densT_4, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
survival_p_dens_4 = survival_p_dens_4 + geom_ribbon(data=survival_df_densT_4, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
survival_p_dens_4 = survival_p_dens_4 + ylab("Density") + xlab("t")+ theme_bw(base_size=25) 
ggsave(paste0(fig_path, "survival_density_4.png"), survival_p_dens_4)
"""

R"""
survival_surv_meanC_4 = apply(survival_survC_4, 2, mean, na.rm=TRUE) 
survival_surv_quanC_4 = apply(survival_survC_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
survival_surv_meanT_4 = apply(survival_survT_4, 2, mean, na.rm=TRUE) 
survival_surv_quanT_4 = apply(survival_survT_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

survival_df_survC_4 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanC_4, 
    l=survival_surv_quanC_4[1,], 
    h=survival_surv_quanC_4[2,]
    )

survival_df_survT_4 = data.frame(
    x=surv_grids, 
    m=survival_surv_meanT_4, 
    l=survival_surv_quanT_4[1,], 
    h=survival_surv_quanT_4[2,]
    )
survival_p_surv_4 = ggplot(survival_df_survC_4) 
survival_p_surv_4 = survival_p_surv_4 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
survival_p_surv_4 = survival_p_surv_4 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
survival_p_surv_4 = survival_p_surv_4 + geom_line(data=survival_df_survT_4, aes(x=x,y=m), color="blue",  linetype="dashed", size=1.5)
survival_p_surv_4 = survival_p_surv_4 + geom_ribbon(data=survival_df_survT_4, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
survival_p_surv_4 = survival_p_surv_4 + ylab("Survival") + xlab("t") 
ggsave(paste0(fig_path, "survival_survival_4.png"), survival_p_surv_4)
"""



@rput gap_grids
@rput gap_densC_0 gap_survC_0
@rput gap_densT_0 gap_survT_0
R"""
gap_dens_meanC_0 = apply(gap_densC_0, 2, mean, na.rm=TRUE) 
gap_dens_quanC_0 = apply(gap_densC_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
gap_dens_meanT_0 = apply(gap_densT_0, 2, mean, na.rm=TRUE) 
gap_dens_quanT_0 = apply(gap_densT_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

gap_df_densC_0 = data.frame(
    x=gap_grids, 
    m=gap_dens_meanC_0, 
    l=gap_dens_quanC_0[1,], 
    h=gap_dens_quanC_0[2,] 
    )
gap_df_densT_0 = data.frame(
    x=gap_grids, 
    m=gap_dens_meanT_0, 
    l=gap_dens_quanT_0[1,], 
    h=gap_dens_quanT_0[2,] 
    )

gap_p_dens_0 = ggplot(gap_df_densC_0) 
gap_p_dens_0 = gap_p_dens_0 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
gap_p_dens_0 = gap_p_dens_0 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) 
gap_p_dens_0 = gap_p_dens_0 + geom_line(data=gap_df_densT_0, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
gap_p_dens_0 = gap_p_dens_0 + geom_ribbon(data=gap_df_densT_0, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
gap_p_dens_0 = gap_p_dens_0 + ylab("Density") + xlab("t")+ theme_bw(base_size=25) 
ggsave(paste0(fig_path, "gap_density_0.png"), gap_p_dens_0)
"""

R"""
gap_surv_meanC_0 = apply(gap_survC_0, 2, mean, na.rm=TRUE) 
gap_surv_quanC_0 = apply(gap_survC_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
gap_surv_meanT_0 = apply(gap_survT_0, 2, mean, na.rm=TRUE) 
gap_surv_quanT_0 = apply(gap_survT_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

gap_df_survC_0 = data.frame(
    x=surv_grids, 
    m=gap_surv_meanC_0, 
    l=gap_surv_quanC_0[1,], 
    h=gap_surv_quanC_0[2,]
    )

gap_df_survT_0 = data.frame(
    x=surv_grids, 
    m=gap_surv_meanT_0, 
    l=gap_surv_quanT_0[1,], 
    h=gap_surv_quanT_0[2,]
    )
gap_p_surv_0 = ggplot(gap_df_survC_0) 
gap_p_surv_0 = gap_p_surv_0 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
gap_p_surv_0 = gap_p_surv_0 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
gap_p_surv_0 = gap_p_surv_0 + geom_line(data=gap_df_survT_0, aes(x=x,y=m), color="blue",  linetype="dashed", size=1.5)
gap_p_surv_0 = gap_p_surv_0 + geom_ribbon(data=gap_df_survT_0, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
gap_p_surv_0 = gap_p_surv_0 + ylab("gap") + xlab("t") + ylim(0, 1)
ggsave(paste0(fig_path, "gap_survival_0.png"), gap_p_surv_0)
"""

grids1 = range(0.5, 13, length=100)
grids2 = range(1,   13, length=100)
grids3 = range(2,   13, length=100)
grids4 = range(5,   13, length=100)

######## cov 0 ########
cond_res1_0 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x0_C,
    x0_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z0_C,
    z0_T,
    BH,
    grids1
)

condC_dens1_0 = cond_res1_0["condC"]
condT_dens1_0 = cond_res1_0["condT"]

@rput condC_dens1_0 condT_dens1_0 grids1
R"""
condC_mean_1_0 = apply(condC_dens1_0, 2, mean, na.rm=TRUE) 
condC_quan_1_0 = apply(condC_dens1_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_1_0 = apply(condT_dens1_0, 2, mean, na.rm=TRUE) 
condT_quan_1_0 = apply(condT_dens1_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_1_0 = data.frame(
    x=grids1, 
    m=condC_mean_1_0, 
    l=condC_quan_1_0[1,], 
    h=condC_quan_1_0[2,]
)

df_condT_1_0 = data.frame(
    x=grids1, 
    m=condT_mean_1_0, 
    l=condT_quan_1_0[1,], 
    h=condT_quan_1_0[2,]
)

p_cond_1_0 = ggplot(df_condC_1_0) 
p_cond_1_0 = p_cond_1_0 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_1_0 = p_cond_1_0 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_1_0 = p_cond_1_0 + geom_line(data=df_condT_1_0, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_1_0 = p_cond_1_0 + geom_ribbon(data=df_condT_1_0, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_1_0 = p_cond_1_0 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_1_0.png"), p_cond_1_0)
"""

cond_res2_0 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x0_C,
    x0_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z0_C,
    z0_T,
    BH,
    grids2
)

condC_dens2_0 = cond_res2_0["condC"]
condT_dens2_0 = cond_res2_0["condT"]

@rput condC_dens2_0 condT_dens2_0 grids2
R"""
condC_mean_2_0 = apply(condC_dens2_0, 2, mean, na.rm=TRUE) 
condC_quan_2_0 = apply(condC_dens2_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_2_0 = apply(condT_dens2_0, 2, mean, na.rm=TRUE) 
condT_quan_2_0 = apply(condT_dens2_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_2_0 = data.frame(
    x=grids2, 
    m=condC_mean_2_0, 
    l=condC_quan_2_0[1,], 
    h=condC_quan_2_0[2,]
)

df_condT_2_0 = data.frame(
    x=grids2, 
    m=condT_mean_2_0, 
    l=condT_quan_2_0[1,], 
    h=condT_quan_2_0[2,]
)

p_cond_2_0 = ggplot(df_condC_2_0) 
p_cond_2_0 = p_cond_2_0 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_2_0 = p_cond_2_0 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_2_0 = p_cond_2_0 + geom_line(data=df_condT_2_0, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_2_0 = p_cond_2_0 + geom_ribbon(data=df_condT_2_0, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_2_0 = p_cond_2_0 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_2_0.png"), p_cond_2_0)
"""

cond_res3_0 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x0_C,
    x0_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z0_C,
    z0_T,
    BH,
    grids3
)

condC_dens3_0 = cond_res3_0["condC"]
condT_dens3_0 = cond_res3_0["condT"]

@rput condC_dens3_0 condT_dens3_0 grids3
R"""
condC_mean_3_0 = apply(condC_dens3_0, 2, mean, na.rm=TRUE) 
condC_quan_3_0 = apply(condC_dens3_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_3_0 = apply(condT_dens3_0, 2, mean, na.rm=TRUE) 
condT_quan_3_0 = apply(condT_dens3_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_3_0 = data.frame(
    x=grids3, 
    m=condC_mean_3_0, 
    l=condC_quan_3_0[1,], 
    h=condC_quan_3_0[2,]
)

df_condT_3_0 = data.frame(
    x=grids3, 
    m=condT_mean_3_0, 
    l=condT_quan_3_0[1,], 
    h=condT_quan_3_0[2,]
)

p_cond_3_0 = ggplot(df_condC_3_0) 
p_cond_3_0 = p_cond_3_0 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_3_0 = p_cond_3_0 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_0 = p_cond_3_0 + geom_line(data=df_condT_3_0, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_3_0 = p_cond_3_0 + geom_ribbon(data=df_condT_3_0, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_0 = p_cond_3_0 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_3_0.png"), p_cond_3_0)
"""

cond_res4_0 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x0_C,
    x0_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z0_C,
    z0_T,
    BH,
    grids4
)

condC_dens4_0 = cond_res4_0["condC"]
condT_dens4_0 = cond_res4_0["condT"]

@rput condC_dens4_0 condT_dens4_0 grids4
R"""
condC_mean_4_0 = apply(condC_dens4_0, 2, mean, na.rm=TRUE) 
condC_quan_4_0 = apply(condC_dens4_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_4_0 = apply(condT_dens4_0, 2, mean, na.rm=TRUE) 
condT_quan_4_0 = apply(condT_dens4_0, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_4_0 = data.frame(
    x=grids4, 
    m=condC_mean_4_0, 
    l=condC_quan_4_0[1,], 
    h=condC_quan_4_0[2,]
)

df_condT_4_0 = data.frame(
    x=grids4, 
    m=condT_mean_4_0, 
    l=condT_quan_4_0[1,], 
    h=condT_quan_4_0[2,]
)

p_cond_4_0 = ggplot(df_condC_4_0) 
p_cond_4_0 = p_cond_4_0 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_4_0 = p_cond_4_0 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_0 = p_cond_4_0 + geom_line(data=df_condT_4_0, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_4_0 = p_cond_4_0 + geom_ribbon(data=df_condT_4_0, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_0 = p_cond_4_0 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_4_0.png"), p_cond_4_0)
"""



######## cov 1 ########
cond_res1_1 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x1_C,
    x1_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z1_C,
    z1_T,
    BH,
    grids1
)

condC_dens1_1 = cond_res1_1["condC"]
condT_dens1_1 = cond_res1_1["condT"]

@rput condC_dens1_1 condT_dens1_1 grids1
R"""
condC_mean_1_1 = apply(condC_dens1_1, 2, mean, na.rm=TRUE) 
condC_quan_1_1 = apply(condC_dens1_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_1_1 = apply(condT_dens1_1, 2, mean, na.rm=TRUE) 
condT_quan_1_1 = apply(condT_dens1_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_1_1 = data.frame(
    x=grids1, 
    m=condC_mean_1_1, 
    l=condC_quan_1_1[1,], 
    h=condC_quan_1_1[2,]
)

df_condT_1_1 = data.frame(
    x=grids1, 
    m=condT_mean_1_1, 
    l=condT_quan_1_1[1,], 
    h=condT_quan_1_1[2,]
)

p_cond_1_1 = ggplot(df_condC_1_1) 
p_cond_1_1 = p_cond_1_1 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_1_1 = p_cond_1_1 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_1_1 = p_cond_1_1 + geom_line(data=df_condT_1_1, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_1_1 = p_cond_1_1 + geom_ribbon(data=df_condT_1_1, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_1_1 = p_cond_1_1 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_1_1.png"), p_cond_1_1)
"""

cond_res2_1 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x1_C,
    x1_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z1_C,
    z1_T,
    BH,
    grids2
)

condC_dens2_1 = cond_res2_1["condC"]
condT_dens2_1 = cond_res2_1["condT"]

@rput condC_dens2_1 condT_dens2_1 grids2
R"""
condC_mean_2_1 = apply(condC_dens2_1, 2, mean, na.rm=TRUE) 
condC_quan_2_1 = apply(condC_dens2_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_2_1 = apply(condT_dens2_1, 2, mean, na.rm=TRUE) 
condT_quan_2_1 = apply(condT_dens2_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_2_1 = data.frame(
    x=grids2, 
    m=condC_mean_2_1, 
    l=condC_quan_2_1[1,], 
    h=condC_quan_2_1[2,]
)

df_condT_2_1 = data.frame(
    x=grids2, 
    m=condT_mean_2_1, 
    l=condT_quan_2_1[1,], 
    h=condT_quan_2_1[2,]
)

p_cond_2_1 = ggplot(df_condC_2_1) 
p_cond_2_1 = p_cond_2_1 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_2_1 = p_cond_2_1 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_2_1 = p_cond_2_1 + geom_line(data=df_condT_2_1, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_2_1 = p_cond_2_1 + geom_ribbon(data=df_condT_2_1, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_2_1 = p_cond_2_1 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_2_1.png"), p_cond_2_1)
"""

cond_res3_1 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x1_C,
    x1_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z1_C,
    z1_T,
    BH,
    grids3
)

condC_dens3_1 = cond_res3_1["condC"]
condT_dens3_1 = cond_res3_1["condT"]

@rput condC_dens3_1 condT_dens3_1 grids3
R"""
condC_mean_3_1 = apply(condC_dens3_1, 2, mean, na.rm=TRUE) 
condC_quan_3_1 = apply(condC_dens3_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_3_1 = apply(condT_dens3_1, 2, mean, na.rm=TRUE) 
condT_quan_3_1 = apply(condT_dens3_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_3_1 = data.frame(
    x=grids3, 
    m=condC_mean_3_1, 
    l=condC_quan_3_1[1,], 
    h=condC_quan_3_1[2,]
)

df_condT_3_1 = data.frame(
    x=grids3, 
    m=condT_mean_3_1, 
    l=condT_quan_3_1[1,], 
    h=condT_quan_3_1[2,]
)

p_cond_3_1 = ggplot(df_condC_3_1) 
p_cond_3_1 = p_cond_3_1 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_3_1 = p_cond_3_1 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_1 = p_cond_3_1 + geom_line(data=df_condT_3_1, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_3_1 = p_cond_3_1 + geom_ribbon(data=df_condT_3_1, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_1 = p_cond_3_1 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_3_1.png"), p_cond_3_1)
"""

cond_res4_1 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x1_C,
    x1_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z1_C,
    z1_T,
    BH,
    grids4
)

condC_dens4_1 = cond_res4_1["condC"]
condT_dens4_1 = cond_res4_1["condT"]

@rput condC_dens4_1 condT_dens4_1 grids4
R"""
condC_mean_4_1 = apply(condC_dens4_1, 2, mean, na.rm=TRUE) 
condC_quan_4_1 = apply(condC_dens4_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_4_1 = apply(condT_dens4_1, 2, mean, na.rm=TRUE) 
condT_quan_4_1 = apply(condT_dens4_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_4_1 = data.frame(
    x=grids4, 
    m=condC_mean_4_1, 
    l=condC_quan_4_1[1,], 
    h=condC_quan_4_1[2,]
)

df_condT_4_1 = data.frame(
    x=grids4, 
    m=condT_mean_4_1, 
    l=condT_quan_4_1[1,], 
    h=condT_quan_4_1[2,]
)

p_cond_4_1 = ggplot(df_condC_4_1) 
p_cond_4_1 = p_cond_4_1 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_4_1 = p_cond_4_1 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_1 = p_cond_4_1 + geom_line(data=df_condT_4_1, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_4_1 = p_cond_4_1 + geom_ribbon(data=df_condT_4_1, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_1 = p_cond_4_1 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_4_1.png"), p_cond_4_1)
"""


######## cov 2 ########
cond_res1_2 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x2_C,
    x2_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z2_C,
    z2_T,
    BH,
    grids1
)

condC_dens1_2 = cond_res1_2["condC"]
condT_dens1_2 = cond_res1_2["condT"]

@rput condC_dens1_2 condT_dens1_2 grids1
R"""
condC_mean_1_2 = apply(condC_dens1_2, 2, mean, na.rm=TRUE) 
condC_quan_1_2 = apply(condC_dens1_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_1_2 = apply(condT_dens1_2, 2, mean, na.rm=TRUE) 
condT_quan_1_2 = apply(condT_dens1_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_1_2 = data.frame(
    x=grids1, 
    m=condC_mean_1_2, 
    l=condC_quan_1_2[1,], 
    h=condC_quan_1_2[2,]
)

df_condT_1_2 = data.frame(
    x=grids1, 
    m=condT_mean_1_2, 
    l=condT_quan_1_2[1,], 
    h=condT_quan_1_2[2,]
)

p_cond_1_2 = ggplot(df_condC_1_2) 
p_cond_1_2 = p_cond_1_2 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_1_2 = p_cond_1_2 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_1_2 = p_cond_1_2 + geom_line(data=df_condT_1_2, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_1_2 = p_cond_1_2 + geom_ribbon(data=df_condT_1_2, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_1_2 = p_cond_1_2 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_1_2.png"), p_cond_1_2)
"""

cond_res2_2 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x2_C,
    x2_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z2_C,
    z2_T,
    BH,
    grids2
)

condC_dens2_2 = cond_res2_2["condC"]
condT_dens2_2 = cond_res2_2["condT"]

@rput condC_dens2_2 condT_dens2_2 grids2
R"""
condC_mean_2_2 = apply(condC_dens2_2, 2, mean, na.rm=TRUE) 
condC_quan_2_2 = apply(condC_dens2_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_2_2 = apply(condT_dens2_2, 2, mean, na.rm=TRUE) 
condT_quan_2_2 = apply(condT_dens2_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_2_2 = data.frame(
    x=grids2, 
    m=condC_mean_2_2, 
    l=condC_quan_2_2[1,], 
    h=condC_quan_2_2[2,]
)

df_condT_2_2 = data.frame(
    x=grids2, 
    m=condT_mean_2_2, 
    l=condT_quan_2_2[1,], 
    h=condT_quan_2_2[2,]
)

p_cond_2_2 = ggplot(df_condC_2_2) 
p_cond_2_2 = p_cond_2_2 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_2_2 = p_cond_2_2 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_2_2 = p_cond_2_2 + geom_line(data=df_condT_2_1, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_2_2 = p_cond_2_2 + geom_ribbon(data=df_condT_2_2, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_2_2 = p_cond_2_2 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_2_2.png"), p_cond_2_2)
"""

cond_res3_2 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x2_C,
    x2_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z2_C,
    z2_T,
    BH,
    grids3
)

condC_dens3_2 = cond_res3_2["condC"]
condT_dens3_2 = cond_res3_2["condT"]

@rput condC_dens3_2 condT_dens3_2 grids3
R"""
condC_mean_3_2 = apply(condC_dens3_2, 2, mean, na.rm=TRUE) 
condC_quan_3_2 = apply(condC_dens3_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_3_2 = apply(condT_dens3_2, 2, mean, na.rm=TRUE) 
condT_quan_3_2 = apply(condT_dens3_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_3_2 = data.frame(
    x=grids3, 
    m=condC_mean_3_2, 
    l=condC_quan_3_2[1,], 
    h=condC_quan_3_2[2,]
)

df_condT_3_2 = data.frame(
    x=grids3, 
    m=condT_mean_3_2, 
    l=condT_quan_3_2[1,], 
    h=condT_quan_3_2[2,]
)

p_cond_3_2 = ggplot(df_condC_3_2) 
p_cond_3_2 = p_cond_3_2 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_3_2 = p_cond_3_2 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_2 = p_cond_3_2 + geom_line(data=df_condT_3_2, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_3_2 = p_cond_3_2 + geom_ribbon(data=df_condT_3_2, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_2 = p_cond_3_2 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_3_2.png"), p_cond_3_2)
"""

cond_res4_2 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x2_C,
    x2_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z2_C,
    z2_T,
    BH,
    grids4
)

condC_dens4_2 = cond_res4_2["condC"]
condT_dens4_2 = cond_res4_2["condT"]

@rput condC_dens4_2 condT_dens4_2 grids4
R"""
condC_mean_4_2 = apply(condC_dens4_2, 2, mean, na.rm=TRUE) 
condC_quan_4_2 = apply(condC_dens4_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_4_2 = apply(condT_dens4_2, 2, mean, na.rm=TRUE) 
condT_quan_4_2 = apply(condT_dens4_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_4_2 = data.frame(
    x=grids4, 
    m=condC_mean_4_2, 
    l=condC_quan_4_2[1,], 
    h=condC_quan_4_2[2,]
)

df_condT_4_2 = data.frame(
    x=grids4, 
    m=condT_mean_4_2, 
    l=condT_quan_4_2[1,], 
    h=condT_quan_4_2[2,]
)

p_cond_4_2 = ggplot(df_condC_4_2) 
p_cond_4_2 = p_cond_4_2 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_4_2 = p_cond_4_2 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_2 = p_cond_4_2 + geom_line(data=df_condT_4_2, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_4_2 = p_cond_4_2 + geom_ribbon(data=df_condT_4_2, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_2 = p_cond_4_2 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_4_2.png"), p_cond_4_2)
"""



######################

cond_res3_3 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x3_C,
    x3_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z3_C,
    z3_T,
    BH,
    grids3
)

condC_dens3_3 = cond_res3_3["condC"]
condT_dens3_3 = cond_res3_3["condT"]

@rput condC_dens3_3 condT_dens3_3 grids3
R"""
condC_mean_3_3 = apply(condC_dens3_3, 2, mean, na.rm=TRUE) 
condC_quan_3_3 = apply(condC_dens3_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_3_3 = apply(condT_dens3_3, 2, mean, na.rm=TRUE) 
condT_quan_3_3 = apply(condT_dens3_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_3_3 = data.frame(
    x=grids3, 
    m=condC_mean_3_3, 
    l=condC_quan_3_3[1,], 
    h=condC_quan_3_3[2,]
)

df_condT_3_3 = data.frame(
    x=grids3, 
    m=condT_mean_3_3, 
    l=condT_quan_3_3[1,], 
    h=condT_quan_3_3[2,]
)

p_cond_3_3 = ggplot(df_condC_3_3) 
p_cond_3_3 = p_cond_3_3 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_3_3 = p_cond_3_3 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_3 = p_cond_3_3 + geom_line(data=df_condT_3_3, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_3_3 = p_cond_3_3 + geom_ribbon(data=df_condT_3_3, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_3 = p_cond_3_3 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_3_3.png"), p_cond_3_3)
"""

cond_res4_3 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x3_C,
    x3_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z3_C,
    z3_T,
    BH,
    grids4
)

condC_dens4_3 = cond_res4_3["condC"]
condT_dens4_3 = cond_res4_3["condT"]

@rput condC_dens4_3 condT_dens4_3 grids4
R"""
condC_mean_4_3 = apply(condC_dens4_3, 2, mean, na.rm=TRUE) 
condC_quan_4_3 = apply(condC_dens4_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_4_3 = apply(condT_dens4_3, 2, mean, na.rm=TRUE) 
condT_quan_4_3 = apply(condT_dens4_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_4_3 = data.frame(
    x=grids4, 
    m=condC_mean_4_3, 
    l=condC_quan_4_3[1,], 
    h=condC_quan_4_3[2,]
)

df_condT_4_3 = data.frame(
    x=grids4, 
    m=condT_mean_4_3, 
    l=condT_quan_4_3[1,], 
    h=condT_quan_4_3[2,]
)

p_cond_4_3 = ggplot(df_condC_4_3) 
p_cond_4_3 = p_cond_4_3 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_4_3 = p_cond_4_3 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_3 = p_cond_4_3 + geom_line(data=df_condT_4_3, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_4_3 = p_cond_4_3 + geom_ribbon(data=df_condT_4_3, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_3 = p_cond_4_3 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_4_3.png"), p_cond_4_3)
"""


######################

cond_res3_4 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x4_C,
    x4_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z4_C,
    z4_T,
    BH,
    grids3
)

condC_dens3_4 = cond_res3_4["condC"]
condT_dens3_4 = cond_res3_4["condT"]

@rput condC_dens3_4 condT_dens3_4 grids3
R"""
condC_mean_3_4 = apply(condC_dens3_4, 2, mean, na.rm=TRUE) 
condC_quan_3_4 = apply(condC_dens3_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_3_4 = apply(condT_dens3_4, 2, mean, na.rm=TRUE) 
condT_quan_3_4 = apply(condT_dens3_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_3_4 = data.frame(
    x=grids3, 
    m=condC_mean_3_4, 
    l=condC_quan_3_4[1,], 
    h=condC_quan_3_4[2,]
)

df_condT_3_4 = data.frame(
    x=grids3, 
    m=condT_mean_3_4, 
    l=condT_quan_3_4[1,], 
    h=condT_quan_3_4[2,]
)

p_cond_3_4 = ggplot(df_condC_3_4) 
p_cond_3_4 = p_cond_3_4 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_3_4 = p_cond_3_4 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_4 = p_cond_3_4 + geom_line(data=df_condT_3_4, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_3_4 = p_cond_3_4 + geom_ribbon(data=df_condT_3_4, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_3_4 = p_cond_3_4 + xlab("t") + ylab("Conditional probability") + xlim(0,13)
ggsave(paste0(fig_path, "conditinal_probability_3_4.png"), p_cond_3_4)
"""

cond_res4_4 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x4_C,
    x4_T,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z4_C,
    z4_T,
    BH,
    grids4
)

condC_dens4_4 = cond_res4_4["condC"]
condT_dens4_4 = cond_res4_4["condT"]

@rput condC_dens4_4 condT_dens4_4 grids4
R"""
condC_mean_4_4 = apply(condC_dens4_4, 2, mean, na.rm=TRUE) 
condC_quan_4_4 = apply(condC_dens4_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_4_4 = apply(condT_dens4_4, 2, mean, na.rm=TRUE) 
condT_quan_4_4 = apply(condT_dens4_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_4_4 = data.frame(
    x=grids4, 
    m=condC_mean_4_4, 
    l=condC_quan_4_4[1,], 
    h=condC_quan_4_4[2,]
)

df_condT_4_4 = data.frame(
    x=grids4, 
    m=condT_mean_4_4, 
    l=condT_quan_4_4[1,], 
    h=condT_quan_4_4[2,]
)

p_cond_4_4 = ggplot(df_condC_4_4) 
p_cond_4_4 = p_cond_4_4 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_4_4 = p_cond_4_4 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_4 = p_cond_4_4 + geom_line(data=df_condT_4_4, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_4_4 = p_cond_4_4 + geom_ribbon(data=df_condT_4_4, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_4_4 = p_cond_4_4 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_4_4.png"), p_cond_4_4)
"""






# pred_tmp = predict_blocked_gibbs_cov(logpC_save, logpT_save, theta_save, beta_save, phi_save, BG, logomegaC_save, logomegaT_save, lambda_save, gamma_save, eta_save, BH)

# theta_predC = pred_tmp["thetaC"] 
# theta_predT = pred_tmp["thetaT"] 
# beta_predC = pred_tmp["betaC"] 
# beta_predT = pred_tmp["betaT"] 
# phi_predC = pred_tmp["phiC"] 
# phi_predT = pred_tmp["phiT"] 
# lambda_predC = pred_tmp["lambdaC"] 
# lambda_predT = pred_tmp["lambdaT"] 
# gamma_predC = pred_tmp["gammaC"] 
# gamma_predT = pred_tmp["gammaT"] 
# eta_predC = pred_tmp["etaC"] 
# eta_predT = pred_tmp["etaT"]

# @rput x0 x1 z0 z1
# @rput theta_predC beta_predC phi_predC lambda_predC gamma_predC eta_predC
# @rput theta_predT beta_predT phi_predT lambda_predT gamma_predT eta_predT
# R"""
# # png(paste0(fig_path, "theta_predC.png"), height=480, width=480)
# # hist(unlist(theta_predC), main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,120,by=3))
# # dev.off() 

# # png(paste0(fig_path, "theta_predT.png"), height=480, width=480)
# # hist(unlist(theta_predT), main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,120,by=3))
# # dev.off() 

# surv_scaleC = unlist(theta_predC) * exp(beta_predC %*% x0)
# png(paste0(fig_path, "surv_scale_predC.png"), height=480, width=480)
# hist(surv_scaleC, main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,max(surv_scaleC)+3,by=3))
# dev.off() 

# surv_scaleT = unlist(theta_predT) * exp(beta_predT %*% x1)
# png(paste0(fig_path, "surv_scale_predT.png"), height=480, width=480)
# hist(surv_scaleT, main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,max(surv_scaleT)+3,by=3))
# dev.off() 

# png(paste0(fig_path, "phi_predC.png"), height=480, width=480)
# hist(unlist(phi_predC), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
# dev.off() 

# png(paste0(fig_path, "phi_predT.png"), height=480, width=480)
# hist(unlist(phi_predT), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
# dev.off() 
# """

# R"""
# gap_scaleT = unlist(lambda_predT) * exp(gamma_predT %*% z1)
# png(paste0(fig_path, "gap_scale_predT.png"), height=480, width=480)
# hist(gap_scaleT, main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,max(gap_scaleT)+3,by=3))
# dev.off() 

# gap_scaleT = unlist(lambda_predT) * exp(gamma_predT %*% z1)
# png(paste0(fig_path, "gap_scale_predT.png"), height=480, width=480)
# hist(gap_scaleT, main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,max(gap_scaleT)+3,by=3))
# dev.off() 

# png(paste0(fig_path, "eta_predC.png"), height=480, width=480)
# hist(unlist(eta_predC), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
# dev.off() 

# png(paste0(fig_path, "eta_predT.png"), height=480, width=480)
# hist(unlist(eta_predT), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
# dev.off() 
# """


# # prior simulation
# a_phi, a_eta = hyper["a_phi"], hyper["a_eta"]

# b_phi_prior = rand(InverseGamma(r_phi, R_phi), 2000)
# phi_prior = zeros(2000, BG)
# for i in 1:2000
#     for l in 1:BG
#         phi_prior[i,l] = sqrt(rand(Gamma(a_phi, b_phi_prior[i]), 1)[1])
#     end
# end

# sigma2_theta = hyper["sigma2_theta"]
# mu_theta_prior = rand(Normal(s_theta, sqrt(S_theta)), 2000)
# theta_prior = zeros(2000, BG)
# for i in 1:2000
#     for l in 1:BG
#         theta_prior[i,l] = exp(rand(Normal(mu_theta_prior[i], sigma2_theta), 1)[1])
#     end
# end

# Sigma_beta = hyper["Sigma_beta"]
# mu_beta_prior = rand(MvNormal(s_beta, S_beta), 2000)
# beta_prior = zeros(2000, BG, 1)
# for i in 1:2000
#     for l in 1:BG
#         beta_prior[i,l,:] = rand(MvNormal(vec(mu_beta_prior[:,i]), Sigma_beta), 1)
#     end
# end

# VC_prior = zeros(2000, BG-1)
# VT_prior = zeros(2000, BG-1)
# logpC_prior = zeros(2000, BG)
# logpT_prior = zeros(2000, BG)
# for i in 1:2000
#     for l in 1:(BG-1)
#         u_prior = rand(Dirichlet([1 - alpha0_prior[i], alpha0_prior[i], alpha0_prior[i], alpha_prior[i] - alpha0_prior[i]]), 1)

#         VC_prior[i,l] = u_prior[1] + u_prior[2]
#         VT_prior[i,l] = u_prior[1] + u_prior[3]
        
#         if VC_prior[i,l] >= 1
#             VC_prior[i,l] = 1 - eps(0.0)
#         end

#         if VT_prior[i,l] >= 1
#             VT_prior[i,l] = 1 - eps(0.0)
#         end

#         if VC_prior[i,l] == 0
#             VC_prior[i,l] = eps(0.0)
#         end

#         if VT_prior[i,l] >= 0
#             VT_prior[i,l] = eps(0.0)
#         end

#         @assert VC_prior[i,l] <= 1 println(u_prior)
#         @assert VT_prior[i,l] <= 1 println(u_prior)
#     end
# end

# for i in 1:2000
#     logpC_prior[i,1] = log(VC_prior[i,1])
#     logpT_prior[i,1] = log(VT_prior[i,1])

#     for l in 2:(BG-1)
#         logpC_prior[i,l] = log(VC_prior[i,l]) + sum(log.(1 .- VC_prior[i,1:(l-1)]))
#         logpT_prior[i,l] = log(VT_prior[i,l]) + sum(log.(1 .- VT_prior[i,1:(l-1)]))
#     end

#     logpC_prior[i,end] = sum(log.(1 .- VC_prior[i,:]))
#     logpT_prior[i,end] = sum(log.(1 .- VT_prior[i,:]))

#     for l in 1:BG
#         if logpC_prior[i,l] == -Inf
#             # print(logp[l], " ", l, "\n")
#             logpC_prior[i,l] = log(eps(0.0))
#         end 
#         if logpT_prior[i,l] == -Inf
#             # print(logp[l], " ", l, "\n")
#             logpT_prior[i,l] = log(eps(0.0))
#         end 
#     end

#     @assert sum(exp.(logpC_prior[i,:])) ≈ 1.0 println(i)
#     @assert sum(exp.(logpT_prior[i,:])) ≈ 1.0 println(i)
# end

# b_eta_prior = rand(InverseGamma(r_eta, R_eta), 2000)
# eta_prior = zeros(2000, BH)
# for i in 1:2000
#     for l in 1:BH
#         eta_prior[i,l] = sqrt(rand(Gamma(a_eta, b_eta_prior[i]), 1)[1])
#     end
# end

# sigma2_lambda = hyper["sigma2_lambda"]
# mu_lambda_prior = rand(Normal(s_lambda, sqrt(S_lambda)), 2000)
# lambda_prior = zeros(2000, BH)
# for i in 1:2000
#     for l in 1:BH
#         lambda_prior[i,l] = exp(rand(Normal(mu_lambda_prior[i], sigma2_lambda), 1)[1])
#     end
# end

# Sigma_gamma = hyper["Sigma_gamma"]
# mu_gamma_prior = rand(MvNormal(s_gamma, S_gamma), 2000)
# gamma_prior = zeros(2000, BH, 1)
# for i in 1:2000
#     for l in 1:BH
#         gamma_prior[i,l,:] = rand(MvNormal(vec(mu_gamma_prior[:,i]), Sigma_gamma), 1)
#     end
# end

# piC_prior = zeros(2000, BG-1)
# piT_prior = zeros(2000, BG-1)
# logomegaC_prior = zeros(2000, BG)
# logomegaT_prior = zeros(2000, BG)
# for i in 1:2000
#     for l in 1:(BG-1)
#         u_prior = rand(Dirichlet([1 - zeta0_prior[i], zeta0_prior[i], zeta0_prior[i], zeta_prior[i] - zeta0_prior[i]]), 1)

#         piC_prior[i,l] = u_prior[1] + u_prior[2]
#         piT_prior[i,l] = u_prior[1] + u_prior[3]
        
#         if piC_prior[i,l] >= 1
#             piC_prior[i,l] = 1 - eps(0.0)
#         end

#         if piT_prior[i,l] >= 1
#             piT_prior[i,l] = 1 - eps(0.0)
#         end

#         if piC_prior[i,l] == 0
#             piC_prior[i,l] = eps(0.0)
#         end

#         if piT_prior[i,l] >= 0
#             piT_prior[i,l] = eps(0.0)
#         end

#         @assert piC_prior[i,l] <= 1 println(u_prior, " ", piC_prior[i,l])
#         @assert piT_prior[i,l] <= 1 println(u_prior, " ", piT_prior[i,l])
#     end
# end

# for i in 1:2000
#     logomegaC_prior[i,1] = log(piC_prior[i,1])
#     logomegaT_prior[i,1] = log(piT_prior[i,1])

#     for l in 2:(BG-1)
#         logomegaC_prior[i,l] = log(piC_prior[i,l]) + sum(log.(1 .- piC_prior[i,1:(l-1)]))
#         logomegaT_prior[i,l] = log(piT_prior[i,l]) + sum(log.(1 .- piT_prior[i,1:(l-1)]))
#     end

#     logomegaC_prior[i,end] = sum(log.(1 .- piC_prior[i,:]))
#     logomegaT_prior[i,end] = sum(log.(1 .- piT_prior[i,:]))
# end

# c_e, C_e = hyper["c_e"], hyper["C_e"]
# Sigma_e_1_prior = zeros(2000, 2, 2)
# Sigma_e_2_prior = zeros(2000, 2, 2)
# for i in 1:2000
#     Sigma_e_1_prior[i,:,:] = rand(InverseWishart(c_e, C_e), 1)[1]
#     Sigma_e_2_prior[i,:,:] = rand(InverseWishart(c_e, C_e), 1)[1]
# end

# surv_prior = functional_estimation_blocked_gibbs_cov(
#     Sigma_e_1_prior[:,1,1], 
#     Sigma_e_2_prior[:,1,1],
#     theta_prior,
#     beta_prior,
#     phi_prior,
#     logpC_prior,
#     logpT_prior,
#     x0,
#     x1,
#     surv_grids,
#     BG
# )

# surv_densC_prior = surv_prior["densC"]
# surv_survC_prior = surv_prior["survC"]
# surv_densT_prior = surv_prior["densT"]
# surv_survT_prior = surv_prior["survT"]

# gap_prior = functional_estimation_blocked_gibbs_cov(
#     Sigma_e_1_prior[:,2,2], 
#     Sigma_e_2_prior[:,2,2],
#     lambda_prior,
#     gamma_prior,
#     eta_prior,
#     logomegaC_prior,
#     logomegaT_prior,
#     z0,
#     z1,
#     gap_grids,
#     BH
#     )

# gap_densC_prior = gap_prior["densC"]
# gap_survC_prior = gap_prior["survC"]
# gap_densT_prior = gap_prior["densT"]
# gap_survT_prior = gap_prior["survT"]

# cond_prior1 = conditional_functional_estimation_blocked_gibbs_cov(
#     Sigma_e_1_prior,
#     Sigma_e_2_prior,
#     theta_prior,
#     beta_prior,
#     phi_prior,
#     logpC_prior,
#     logpT_prior,
#     x0,
#     x1,
#     BG,
#     lambda_prior,
#     gamma_prior,
#     eta_prior,
#     logomegaC_prior,
#     logomegaT_prior,
#     z0,
#     z1,
#     BH,
#     grids1
# )

# cond1_priorC = cond_prior1["condC"]
# cond1_priorT = cond_prior1["condT"]

# cond_prior2 = conditional_functional_estimation_blocked_gibbs_cov(
#     Sigma_e_1_prior,
#     Sigma_e_2_prior,
#     theta_prior,
#     beta_prior,
#     phi_prior,
#     logpC_prior,
#     logpT_prior,
#     x0,
#     x1,
#     BG,
#     lambda_prior,
#     gamma_prior,
#     eta_prior,
#     logomegaC_prior,
#     logomegaT_prior,
#     z0,
#     z1,
#     BH,
#     grids2
# )

# cond2_priorC = cond_prior2["condC"]
# cond2_priorT = cond_prior2["condT"]

# cond_prior3 = conditional_functional_estimation_blocked_gibbs_cov(
#     Sigma_e_1_prior,
#     Sigma_e_2_prior,
#     theta_prior,
#     beta_prior,
#     phi_prior,
#     logpC_prior,
#     logpT_prior,
#     x0,
#     x1,
#     BG,
#     lambda_prior,
#     gamma_prior,
#     eta_prior,
#     logomegaC_prior,
#     logomegaT_prior,
#     z0,
#     z1,
#     BH,
#     grids3
# )

# cond3_priorC = cond_prior3["condC"]
# cond3_priorT = cond_prior3["condT"]

# cond_prior4 = conditional_functional_estimation_blocked_gibbs_cov(
#     Sigma_e_1_prior,
#     Sigma_e_2_prior,
#     theta_prior,
#     beta_prior,
#     phi_prior,
#     logpC_prior,
#     logpT_prior,
#     x0,
#     x1,
#     BG,
#     lambda_prior,
#     gamma_prior,
#     eta_prior,
#     logomegaC_prior,
#     logomegaT_prior,
#     z0,
#     z1,
#     BH,
#     grids4
# )

# cond4_priorC = cond_prior4["condC"]
# cond4_priorT = cond_prior4["condT"]

# @rput surv_densC_prior surv_densT_prior
# @rput surv_survC_prior surv_survT_prior
# @rput gap_densC_prior gap_densT_prior
# @rput gap_survC_prior gap_survT_prior
# @rput cond1_priorC cond1_priorT
# @rput cond2_priorC cond2_priorT
# @rput cond3_priorC cond3_priorT
# @rput cond4_priorC cond4_priorT
# R"""
# surv_densC_prior_quan = apply(surv_densC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# surv_densC_prior_df = data.frame(
#     x = surv_grids, 
#     l = surv_densC_prior_quan[1,],
#     h = surv_densC_prior_quan[2,]
# )
# surv_densT_prior_quan = apply(surv_densT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# surv_densT_prior_df = data.frame(
#     x = surv_grids, 
#     l = surv_densT_prior_quan[1,],
#     h = surv_densT_prior_quan[2,]
# )

# surv_survC_prior_quan = apply(surv_survC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# surv_survC_prior_df = data.frame(
#     x = surv_grids, 
#     l = surv_survC_prior_quan[1,],
#     h = surv_survC_prior_quan[2,]
# )
# surv_survT_prior_quan = apply(surv_survT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# surv_survT_prior_df = data.frame(
#     x = surv_grids, 
#     l = surv_survT_prior_quan[1,],
#     h = surv_survT_prior_quan[2,]
# )

# gap_densC_prior_quan = apply(gap_densC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# gap_densC_prior_df = data.frame(
#     x = gap_grids, 
#     l = gap_densC_prior_quan[1,],
#     h = gap_densC_prior_quan[2,]
# )
# gap_densT_prior_quan = apply(gap_densT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# gap_densT_prior_df = data.frame(
#     x = gap_grids, 
#     l = gap_densT_prior_quan[1,],
#     h = gap_densT_prior_quan[2,]
# )

# gap_survC_prior_quan = apply(gap_densC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# gap_survC_prior_df = data.frame(
#     x = gap_grids, 
#     l = gap_survC_prior_quan[1,],
#     h = gap_survC_prior_quan[2,]
# )
# gap_survT_prior_quan = apply(gap_densT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# gap_survT_prior_df = data.frame(
#     x = gap_grids, 
#     l = gap_survT_prior_quan[1,],
#     h = gap_survT_prior_quan[2,]
# )

# cond1_priorC_quan = apply(cond1_priorC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# cond1_priorC_df = data.frame(
#     x = grids1, 
#     l = cond1_priorC_quan[1,],
#     h = cond1_priorC_quan[2,]
# )
# cond1_priorT_quan = apply(cond1_priorT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# cond1_priorT_df = data.frame(
#     x = grids1, 
#     l = cond1_priorT_quan[1,],
#     h = cond1_priorT_quan[2,]
# )

# cond2_priorC_quan = apply(cond2_priorC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# cond2_priorC_df = data.frame(
#     x = grids2, 
#     l = cond2_priorC_quan[1,],
#     h = cond2_priorC_quan[2,]
# )
# cond2_priorT_quan = apply(cond2_priorT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# cond2_priorT_df = data.frame(
#     x = grids2,
#     l = cond2_priorT_quan[1,],
#     h = cond2_priorT_quan[2,]
# )

# cond3_priorC_quan = apply(cond3_priorC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# cond3_priorC_df = data.frame(
#     x = grids3, 
#     l = cond3_priorC_quan[1,],
#     h = cond3_priorC_quan[2,]
# )
# cond3_priorT_quan = apply(cond3_priorT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# cond3_priorT_df = data.frame(
#     x = grids3,
#     l = cond3_priorT_quan[1,],
#     h = cond3_priorT_quan[2,]
# )

# cond4_priorC_quan = apply(cond4_priorC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# cond4_priorC_df = data.frame(
#     x = grids4, 
#     l = cond4_priorC_quan[1,],
#     h = cond4_priorC_quan[2,]
# )
# cond4_priorT_quan = apply(cond4_priorT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# cond4_priorT_df = data.frame(
#     x = grids4,
#     l = cond4_priorT_quan[1,],
#     h = cond4_priorT_quan[2,]
# )
# """

# R"""
# survival_p_dens_prior = survival_p_dens
# survival_p_dens_prior = survival_p_dens_prior + geom_ribbon(data=surv_densC_prior_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
# survival_p_dens_prior = survival_p_dens_prior + geom_ribbon(data=surv_densT_prior_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) + ylim(0,1)
# ggsave(paste0(fig_path, "survival_density_w_prior.png"), survival_p_dens_prior)
# """

# R"""
# survival_p_surv_prior = survival_p_surv
# survival_p_surv_prior = survival_p_surv_prior + geom_ribbon(data=surv_survC_prior_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
# survival_p_surv_prior = survival_p_surv_prior + geom_ribbon(data=surv_survT_prior_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
# ggsave(paste0(fig_path, "survival_survival_w_prior.png"), survival_p_surv_prior)
# """

# R"""
# gap_p_dens_prior = gap_p_dens
# gap_p_dens_prior = gap_p_dens_prior + geom_ribbon(data=gap_densC_prior_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
# gap_p_dens_prior = gap_p_dens_prior + geom_ribbon(data=gap_densT_prior_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
# ggsave(paste0(fig_path, "gap_density_w_prior.png"), gap_p_dens_prior) + ylim(0,1)
# """

# R"""
# p_cond1_prior = p_cond_1
# p_cond1_prior = p_cond1_prior + geom_ribbon(data=cond1_priorC_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
# p_cond1_prior = p_cond1_prior + geom_ribbon(data=cond1_priorT_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
# ggsave(paste0(fig_path, "conditinal_probability_1_w_prior.png"), p_cond1_prior)
# """

# R"""
# p_cond2_prior = p_cond_2
# p_cond2_prior = p_cond2_prior + geom_ribbon(data=cond2_priorC_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
# p_cond2_prior = p_cond2_prior + geom_ribbon(data=cond2_priorT_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
# ggsave(paste0(fig_path, "conditinal_probability_2_w_prior.png"), p_cond2_prior)
# """

# R"""
# p_cond3_prior = p_cond_3
# p_cond3_prior = p_cond3_prior + geom_ribbon(data=cond3_priorC_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
# p_cond3_prior = p_cond3_prior + geom_ribbon(data=cond3_priorT_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
# ggsave(paste0(fig_path, "conditinal_probability_3_w_prior.png"), p_cond3_prior)
# """

# R"""
# p_cond4_prior = p_cond_4
# p_cond4_prior = p_cond4_prior + geom_ribbon(data=cond4_priorC_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
# p_cond4_prior = p_cond4_prior + geom_ribbon(data=cond4_priorT_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
# ggsave(paste0(fig_path, "conditinal_probability_4_w_prior.png"), p_cond4_prior)
# """