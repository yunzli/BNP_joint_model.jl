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

config = TOML.parsefile("../configs/effusion_cov_common_atoms_blocked_gibbs.TOML")

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

nsam = length(pos["alpha"])
nburn = div(nsam, 4)
nthin = div(nsam-nburn,2000)
keep_index = [nburn+1:nthin:nsam;]
nkeep = length(keep_index)

alpha_save = pos["alpha"][keep_index]
alpha0_save = pos["alpha0"][keep_index]
zeta_save = pos["zeta"][keep_index]
zeta0_save = pos["zeta0"][keep_index]

a0, b0, c0 = hyper["a0"], hyper["b0"], hyper["c0"]
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

a1, b1, c1 = hyper["a1"], hyper["b1"], hyper["c1"]
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

png(paste0(fig_path, "zeta_trace.png"))
plot(zeta_save, type='l', cex.axis=2)
dev.off( )
"""

LC_save = pos["LC"][keep_index,:]
LT_save = pos["LT"][keep_index,:]

logpC_save = pos["logpC"][keep_index,:]
logpT_save = pos["logpT"][keep_index,:]

mu_theta_save = pos["mu_theta"][keep_index]
b_phi_save = pos["b_phi"][keep_index]

s_theta, S_theta = hyper["s_theta"], hyper["S_theta"]
r_phi, R_phi = hyper["r_phi"], hyper["R_phi"]
@rput s_theta S_theta
@rput r_phi R_phi
@rput mu_theta_save b_phi_save
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

BG, BH = hyper["BG"], hyper["BH"]

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
b_eta_save = pos["b_eta"][keep_index]

s_lambda, S_lambda = hyper["s_lambda"], hyper["S_lambda"]
r_eta, R_eta = hyper["r_eta"], hyper["R_eta"]
@rput s_lambda S_lambda
@rput r_eta R_eta
@rput mu_lambda_save b_eta_save
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
phi_save = pos["phi"][keep_index,:]

surv_grids = range(0.001, 13, length=100)

surv_res = functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_save[:,1,1], 
    Sigma_e_2_save[:,1,1],
    theta_save,
    phi_save,
    logpC_save,
    logpT_save,
    surv_grids,
    BG
    )

survival_densC = surv_res["densC"]
survival_survC = surv_res["survC"]
survival_densT = surv_res["densT"]
survival_survT = surv_res["survT"]

lambda_save = pos["lambda"][keep_index,:]
eta_save = pos["eta"][keep_index,:]

gap_grids = range(0.001, 13, length=100)

gap_res = functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_save[:,2,2], 
    Sigma_e_2_save[:,2,2],
    lambda_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    gap_grids,
    BH
    )

gap_densC = gap_res["densC"]
gap_survC = gap_res["survC"]
gap_densT = gap_res["densT"]
gap_survT = gap_res["survT"]

@rput surv_grids
@rput survival_densC survival_survC
@rput survival_densT survival_survT
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
survival_p_surv = survival_p_surv + ylab("Survival") + xlab("t") + ylim(0, 1)
ggsave(paste0(fig_path, "survival_survival.png"), survival_p_surv)
"""



@rput gap_grids
@rput gap_densC gap_survC
@rput gap_densT gap_survT
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
gap_p_dens = gap_p_dens + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) 
gap_p_dens = gap_p_dens + geom_line(data=gap_df_densT, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
gap_p_dens = gap_p_dens + geom_ribbon(data=gap_df_densT, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
gap_p_dens = gap_p_dens + ylab("Density") + xlab("t")+ theme_bw(base_size=25) 
ggsave(paste0(fig_path, "gap_density.png"), gap_p_dens)
"""

R"""
gap_surv_meanC = apply(gap_survC, 2, mean, na.rm=TRUE) 
gap_surv_quanC = apply(gap_survC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
gap_surv_meanT = apply(gap_survT, 2, mean, na.rm=TRUE) 
gap_surv_quanT = apply(gap_survT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

gap_df_survC = data.frame(
    x=surv_grids, 
    m=gap_surv_meanC, 
    l=gap_surv_quanC[1,], 
    h=gap_surv_quanC[2,]
    )

gap_df_survT = data.frame(
    x=surv_grids, 
    m=gap_surv_meanT, 
    l=gap_surv_quanT[1,], 
    h=gap_surv_quanT[2,]
    )
gap_p_surv = ggplot(gap_df_survC) 
gap_p_surv = gap_p_surv + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
gap_p_surv = gap_p_surv + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
gap_p_surv = gap_p_surv + geom_line(data=gap_df_survT, aes(x=x,y=m), color="blue",  linetype="dashed", size=1.5)
gap_p_surv = gap_p_surv + geom_ribbon(data=gap_df_survT, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
gap_p_surv = gap_p_surv + ylab("gap") + xlab("t") + ylim(0, 1)
ggsave(paste0(fig_path, "gap_survival.png"), gap_p_surv)
"""

grids1 = range(0.5, 13, length=100)
grids2 = range(1,   13, length=100)
grids3 = range(2,   13, length=100)
grids4 = range(5,   13, length=100)

cond_res1 = conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    phi_save,
    logpC_save,
    logpT_save,
    BG,
    lambda_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    BH,
    grids1
)

condC_surv1 = cond_res1["condC"]
condT_surv1 = cond_res1["condT"]

@rput condC_surv1 condT_surv1 grids1
R"""
condC_mean_1 = apply(condC_surv1, 2, mean, na.rm=TRUE) 
condC_quan_1 = apply(condC_surv1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_1 = apply(condT_surv1, 2, mean, na.rm=TRUE) 
condT_quan_1 = apply(condT_surv1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_1 = data.frame(
    x=grids1, 
    m=condC_mean_1, 
    l=condC_quan_1[1,], 
    h=condC_quan_1[2,]
)

df_condT_1 = data.frame(
    x=grids1, 
    m=condT_mean_1, 
    l=condT_quan_1[1,], 
    h=condT_quan_1[2,]
)

p_cond_1 = ggplot(df_condC_1) 
p_cond_1 = p_cond_1 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_1 = p_cond_1 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_1 = p_cond_1 + geom_line(data=df_condT_1, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_1 = p_cond_1 + geom_ribbon(data=df_condT_1, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_1 = p_cond_1 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_1.png"), p_cond_1)
"""

cond_res2 = conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    phi_save,
    logpC_save,
    logpT_save,
    BG,
    lambda_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    BH,
    grids2
)

condC_surv2 = cond_res2["condC"]
condT_surv2 = cond_res2["condT"]

@rput condC_surv2 condT_surv2 grids2
R"""
condC_mean_2 = apply(condC_surv2, 2, mean, na.rm=TRUE) 
condC_quan_2 = apply(condC_surv2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_2 = apply(condT_surv2, 2, mean, na.rm=TRUE) 
condT_quan_2 = apply(condT_surv2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_2 = data.frame(
    x=grids2, 
    m=condC_mean_2, 
    l=condC_quan_2[1,], 
    h=condC_quan_2[2,]
)

df_condT_2 = data.frame(
    x=grids2, 
    m=condT_mean_2, 
    l=condT_quan_2[1,], 
    h=condT_quan_2[2,]
)

p_cond_2 = ggplot(df_condC_2) 
p_cond_2 = p_cond_2 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_2 = p_cond_2 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_2 = p_cond_2 + geom_line(data=df_condT_2, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_2 = p_cond_2 + geom_ribbon(data=df_condT_2, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_2 = p_cond_2 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_2.png"), p_cond_2)
"""

cond_res3 = conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    phi_save,
    logpC_save,
    logpT_save,
    BG,
    lambda_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    BH,
    grids3
)

condC_surv3 = cond_res3["condC"]
condT_surv3 = cond_res3["condT"]

@rput condC_surv3 condT_surv3 grids3
R"""
condC_mean_3 = apply(condC_surv3, 2, mean, na.rm=TRUE) 
condC_quan_3 = apply(condC_surv3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_3 = apply(condT_surv3, 2, mean, na.rm=TRUE) 
condT_quan_3 = apply(condT_surv3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_3 = data.frame(
    x=grids3, 
    m=condC_mean_3, 
    l=condC_quan_3[1,], 
    h=condC_quan_3[2,]
)

df_condT_3 = data.frame(
    x=grids3, 
    m=condT_mean_3, 
    l=condT_quan_3[1,], 
    h=condT_quan_3[2,]
)

p_cond_3 = ggplot(df_condC_3) 
p_cond_3 = p_cond_3 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_3 = p_cond_3 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_3 = p_cond_3 + geom_line(data=df_condT_3, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_3 = p_cond_3 + geom_ribbon(data=df_condT_3, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_3 = p_cond_3 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_3.png"), p_cond_3)
"""

cond_res4 = conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    phi_save,
    logpC_save,
    logpT_save,
    BG,
    lambda_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    BH,
    grids4
)

condC_surv4 = cond_res4["condC"]
condT_surv4 = cond_res4["condT"]

@rput condC_surv4 condT_surv4 grids4
R"""
condC_mean_4 = apply(condC_surv4, 2, mean, na.rm=TRUE) 
condC_quan_4 = apply(condC_surv4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_4 = apply(condT_surv4, 2, mean, na.rm=TRUE) 
condT_quan_4 = apply(condT_surv4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_4 = data.frame(
    x=grids4, 
    m=condC_mean_4, 
    l=condC_quan_4[1,], 
    h=condC_quan_4[2,]
)

df_condT_4 = data.frame(
    x=grids4, 
    m=condT_mean_4, 
    l=condT_quan_4[1,], 
    h=condT_quan_4[2,]
)

p_cond_4 = ggplot(df_condC_4) 
p_cond_4 = p_cond_4 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_4 = p_cond_4 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_4 = p_cond_4 + geom_line(data=df_condT_4, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_4 = p_cond_4 + geom_ribbon(data=df_condT_4, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_4 = p_cond_4 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_4.png"), p_cond_4)
"""


pred_tmp = predict_blocked_gibbs_common_atoms(logpC_save, logpT_save, theta_save, phi_save, BG, logomegaC_save, logomegaT_save, lambda_save, eta_save, BH)

theta_predC = pred_tmp["thetaC"] 
theta_predT = pred_tmp["thetaT"] 
phi_predC = pred_tmp["phiC"] 
phi_predT = pred_tmp["phiT"] 
lambda_predC = pred_tmp["lambdaC"] 
lambda_predT = pred_tmp["lambdaT"] 
eta_predC = pred_tmp["etaC"] 
eta_predT = pred_tmp["etaT"]

@rput theta_predC phi_predC lambda_predC eta_predC
@rput theta_predT phi_predT lambda_predT eta_predT
R"""
png(paste0(fig_path, "theta_predC.png"), height=480, width=480)
hist(unlist(theta_predC), main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,120,by=3))
dev.off() 

png(paste0(fig_path, "phi_predC.png"), height=480, width=480)
hist(unlist(phi_predC), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
dev.off() 

png(paste0(fig_path, "theta_predT.png"), height=480, width=480)
hist(unlist(theta_predT), main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,120,by=3))
dev.off() 

png(paste0(fig_path, "phi_predT.png"), height=480, width=480)
hist(unlist(phi_predT), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
dev.off() 
"""

R"""
png(paste0(fig_path, "lambda_predC.png"), height=480, width=480)
hist(unlist(lambda_predC), main="", xlab="", cex.axis=2, xlim=c(0,60), breaks=seq(0,330,by=3))
dev.off() 

png(paste0(fig_path, "eta_predC.png"), height=480, width=480)
hist(unlist(eta_predC), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
dev.off() 

png(paste0(fig_path, "lambda_predT.png"), height=480, width=480)
hist(unlist(lambda_predT), main="", xlab="", cex.axis=2, xlim=c(0,60), breaks=seq(0,330,by=3))
dev.off() 

png(paste0(fig_path, "eta_predT.png"), height=480, width=480)
hist(unlist(eta_predT), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
dev.off() 
"""


# prior simulation
a_phi, a_eta = hyper["a_phi"], hyper["a_eta"]

b_phi_prior = rand(InverseGamma(r_phi, R_phi), 2000)
phi_prior = zeros(2000, BG)
for i in 1:2000
    for l in 1:BG
        phi_prior[i,l] = sqrt(rand(Gamma(a_phi, b_phi_prior[i]), 1)[1])
    end
end

sigma2_theta = hyper["sigma2_theta"]
mu_theta_prior = rand(Normal(s_theta, sqrt(S_theta)), 2000)
theta_prior = zeros(2000, BG)
for i in 1:2000
    for l in 1:BG
        theta_prior[i,l] = exp(rand(Normal(mu_theta_prior[i], sigma2_theta), 1)[1])
    end
end

VC_prior = zeros(2000, BG-1)
VT_prior = zeros(2000, BG-1)
logpC_prior = zeros(2000, BG)
logpT_prior = zeros(2000, BG)
for i in 1:2000
    for l in 1:(BG-1)
        u_prior = rand(Dirichlet([1 - alpha0_prior[i], alpha0_prior[i], alpha0_prior[i], alpha_prior[i] - alpha0_prior[i]]), 1)

        VC_prior[i,l] = u_prior[1] + u_prior[2]
        VT_prior[i,l] = u_prior[1] + u_prior[3]
        
        if VC_prior[i,l] >= 1
            VC_prior[i,l] = 1 - eps(0.0)
        end

        if VT_prior[i,l] >= 1
            VT_prior[i,l] = 1 - eps(0.0)
        end

        if VC_prior[i,l] == 0
            VC_prior[i,l] = eps(0.0)
        end

        if VT_prior[i,l] >= 0
            VT_prior[i,l] = eps(0.0)
        end

        @assert VC_prior[i,l] <= 1 println(u_prior)
        @assert VT_prior[i,l] <= 1 println(u_prior)
    end
end

for i in 1:2000
    logpC_prior[i,1] = log(VC_prior[i,1])
    logpT_prior[i,1] = log(VT_prior[i,1])

    for l in 2:(BG-1)
        logpC_prior[i,l] = log(VC_prior[i,l]) + sum(log.(1 .- VC_prior[i,1:(l-1)]))
        logpT_prior[i,l] = log(VT_prior[i,l]) + sum(log.(1 .- VT_prior[i,1:(l-1)]))
    end

    logpC_prior[i,end] = sum(log.(1 .- VC_prior[i,:]))
    logpT_prior[i,end] = sum(log.(1 .- VT_prior[i,:]))

    for l in 1:BG
        if logpC_prior[i,l] == -Inf
            # print(logp[l], " ", l, "\n")
            logpC_prior[i,l] = log(eps(0.0))
        end 
        if logpT_prior[i,l] == -Inf
            # print(logp[l], " ", l, "\n")
            logpT_prior[i,l] = log(eps(0.0))
        end 
    end

    @assert sum(exp.(logpC_prior[i,:])) ≈ 1.0 println(i)
    @assert sum(exp.(logpT_prior[i,:])) ≈ 1.0 println(i)
end

b_eta_prior = rand(InverseGamma(r_eta, R_eta), 2000)
eta_prior = zeros(2000, BH)
for i in 1:2000
    for l in 1:BH
        eta_prior[i,l] = sqrt(rand(Gamma(a_eta, b_eta_prior[i]), 1)[1])
    end
end

sigma2_lambda = hyper["sigma2_lambda"]
mu_lambda_prior = rand(Normal(s_lambda, sqrt(S_lambda)), 2000)
lambda_prior = zeros(2000, BH)
for i in 1:2000
    for l in 1:BH
        lambda_prior[i,l] = exp(rand(Normal(mu_lambda_prior[i], sigma2_lambda), 1)[1])
    end
end

piC_prior = zeros(2000, BG-1)
piT_prior = zeros(2000, BG-1)
logomegaC_prior = zeros(2000, BG)
logomegaT_prior = zeros(2000, BG)
for i in 1:2000
    for l in 1:(BG-1)
        u_prior = rand(Dirichlet([1 - zeta0_prior[i], zeta0_prior[i], zeta0_prior[i], zeta_prior[i] - zeta0_prior[i]]), 1)

        piC_prior[i,l] = u_prior[1] + u_prior[2]
        piT_prior[i,l] = u_prior[1] + u_prior[3]
        
        if piC_prior[i,l] >= 1
            piC_prior[i,l] = 1 - eps(0.0)
        end

        if piT_prior[i,l] >= 1
            piT_prior[i,l] = 1 - eps(0.0)
        end

        if piC_prior[i,l] == 0
            piC_prior[i,l] = eps(0.0)
        end

        if piT_prior[i,l] >= 0
            piT_prior[i,l] = eps(0.0)
        end

        @assert piC_prior[i,l] <= 1 println(u_prior, " ", piC_prior[i,l])
        @assert piT_prior[i,l] <= 1 println(u_prior, " ", piT_prior[i,l])
    end
end

for i in 1:2000
    logomegaC_prior[i,1] = log(piC_prior[i,1])
    logomegaT_prior[i,1] = log(piT_prior[i,1])

    for l in 2:(BG-1)
        logomegaC_prior[i,l] = log(piC_prior[i,l]) + sum(log.(1 .- piC_prior[i,1:(l-1)]))
        logomegaT_prior[i,l] = log(piT_prior[i,l]) + sum(log.(1 .- piT_prior[i,1:(l-1)]))
    end

    logomegaC_prior[i,end] = sum(log.(1 .- piC_prior[i,:]))
    logomegaT_prior[i,end] = sum(log.(1 .- piT_prior[i,:]))
end

c_e, C_e = hyper["c_e"], hyper["C_e"]
Sigma_e_1_prior = zeros(2000, 2, 2)
Sigma_e_2_prior = zeros(2000, 2, 2)
for i in 1:2000
    Sigma_e_1_prior[i,:,:] = rand(InverseWishart(c_e, C_e), 1)[1]
    Sigma_e_2_prior[i,:,:] = rand(InverseWishart(c_e, C_e), 1)[1]
end

surv_prior = functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_prior[:,1,1], 
    Sigma_e_2_prior[:,1,1],
    theta_prior,
    phi_prior,
    logpC_prior,
    logpT_prior,
    surv_grids,
    BG
)

surv_densC_prior = surv_prior["densC"]
surv_survC_prior = surv_prior["survC"]
surv_densT_prior = surv_prior["densT"]
surv_survT_prior = surv_prior["survT"]

gap_prior = functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_prior[:,2,2], 
    Sigma_e_2_prior[:,2,2],
    lambda_prior,
    eta_prior,
    logomegaC_prior,
    logomegaT_prior,
    gap_grids,
    BH
    )

gap_densC_prior = gap_prior["densC"]
gap_survC_prior = gap_prior["survC"]
gap_densT_prior = gap_prior["densT"]
gap_survT_prior = gap_prior["survT"]

cond_prior1 = conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_prior,
    Sigma_e_2_prior,
    theta_prior,
    phi_prior,
    logpC_prior,
    logpT_prior,
    BG,
    lambda_prior,
    eta_prior,
    logomegaC_prior,
    logomegaT_prior,
    BH,
    grids1
)

cond1_priorC = cond_prior1["condC"]
cond1_priorT = cond_prior1["condT"]

cond_prior2 = conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_prior,
    Sigma_e_2_prior,
    theta_prior,
    phi_prior,
    logpC_prior,
    logpT_prior,
    BG,
    lambda_prior,
    eta_prior,
    logomegaC_prior,
    logomegaT_prior,
    BH,
    grids2
)

cond2_priorC = cond_prior2["condC"]
cond2_priorT = cond_prior2["condT"]

cond_prior3 = conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_prior,
    Sigma_e_2_prior,
    theta_prior,
    phi_prior,
    logpC_prior,
    logpT_prior,
    BG,
    lambda_prior,
    eta_prior,
    logomegaC_prior,
    logomegaT_prior,
    BH,
    grids3
)

cond3_priorC = cond_prior3["condC"]
cond3_priorT = cond_prior3["condT"]

cond_prior4 = conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_prior,
    Sigma_e_2_prior,
    theta_prior,
    phi_prior,
    logpC_prior,
    logpT_prior,
    BG,
    lambda_prior,
    eta_prior,
    logomegaC_prior,
    logomegaT_prior,
    BH,
    grids4
)

cond4_priorC = cond_prior4["condC"]
cond4_priorT = cond_prior4["condT"]

@rput surv_densC_prior surv_densT_prior
@rput surv_survC_prior surv_survT_prior
@rput gap_densC_prior gap_densT_prior
@rput gap_survC_prior gap_survT_prior
@rput cond1_priorC cond1_priorT
@rput cond2_priorC cond2_priorT
@rput cond3_priorC cond3_priorT
@rput cond4_priorC cond4_priorT
R"""
surv_densC_prior_quan = apply(surv_densC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
surv_densC_prior_df = data.frame(
    x = surv_grids, 
    l = surv_densC_prior_quan[1,],
    h = surv_densC_prior_quan[2,]
)
surv_densT_prior_quan = apply(surv_densT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
surv_densT_prior_df = data.frame(
    x = surv_grids, 
    l = surv_densT_prior_quan[1,],
    h = surv_densT_prior_quan[2,]
)

surv_survC_prior_quan = apply(surv_survC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
surv_survC_prior_df = data.frame(
    x = surv_grids, 
    l = surv_survC_prior_quan[1,],
    h = surv_survC_prior_quan[2,]
)
surv_survT_prior_quan = apply(surv_survT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
surv_survT_prior_df = data.frame(
    x = surv_grids, 
    l = surv_survT_prior_quan[1,],
    h = surv_survT_prior_quan[2,]
)

gap_densC_prior_quan = apply(gap_densC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
gap_densC_prior_df = data.frame(
    x = gap_grids, 
    l = gap_densC_prior_quan[1,],
    h = gap_densC_prior_quan[2,]
)
gap_densT_prior_quan = apply(gap_densT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
gap_densT_prior_df = data.frame(
    x = gap_grids, 
    l = gap_densT_prior_quan[1,],
    h = gap_densT_prior_quan[2,]
)

gap_survC_prior_quan = apply(gap_densC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
gap_survC_prior_df = data.frame(
    x = gap_grids, 
    l = gap_survC_prior_quan[1,],
    h = gap_survC_prior_quan[2,]
)
gap_survT_prior_quan = apply(gap_densT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
gap_survT_prior_df = data.frame(
    x = gap_grids, 
    l = gap_survT_prior_quan[1,],
    h = gap_survT_prior_quan[2,]
)

cond1_priorC_quan = apply(cond1_priorC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond1_priorC_df = data.frame(
    x = grids1, 
    l = cond1_priorC_quan[1,],
    h = cond1_priorC_quan[2,]
)
cond1_priorT_quan = apply(cond1_priorT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond1_priorT_df = data.frame(
    x = grids1, 
    l = cond1_priorT_quan[1,],
    h = cond1_priorT_quan[2,]
)

cond2_priorC_quan = apply(cond2_priorC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond2_priorC_df = data.frame(
    x = grids2, 
    l = cond2_priorC_quan[1,],
    h = cond2_priorC_quan[2,]
)
cond2_priorT_quan = apply(cond2_priorT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond2_priorT_df = data.frame(
    x = grids2,
    l = cond2_priorT_quan[1,],
    h = cond2_priorT_quan[2,]
)

cond3_priorC_quan = apply(cond3_priorC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond3_priorC_df = data.frame(
    x = grids3, 
    l = cond3_priorC_quan[1,],
    h = cond3_priorC_quan[2,]
)
cond3_priorT_quan = apply(cond3_priorT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond3_priorT_df = data.frame(
    x = grids3,
    l = cond3_priorT_quan[1,],
    h = cond3_priorT_quan[2,]
)

cond4_priorC_quan = apply(cond4_priorC, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond4_priorC_df = data.frame(
    x = grids4, 
    l = cond4_priorC_quan[1,],
    h = cond4_priorC_quan[2,]
)
cond4_priorT_quan = apply(cond4_priorT, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond4_priorT_df = data.frame(
    x = grids4,
    l = cond4_priorT_quan[1,],
    h = cond4_priorT_quan[2,]
)
"""

R"""
survival_p_dens_prior = survival_p_dens
survival_p_dens_prior = survival_p_dens_prior + geom_ribbon(data=surv_densC_prior_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
survival_p_dens_prior = survival_p_dens_prior + geom_ribbon(data=surv_densT_prior_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
ggsave(paste0(fig_path, "survival_density_w_prior.png"), survival_p_dens_prior)
"""

R"""
survival_p_surv_prior = survival_p_surv
survival_p_surv_prior = survival_p_surv_prior + geom_ribbon(data=surv_survC_prior_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
survival_p_surv_prior = survival_p_surv_prior + geom_ribbon(data=surv_survT_prior_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
ggsave(paste0(fig_path, "survival_survival_w_prior.png"), survival_p_surv_prior)
"""

R"""
gap_p_dens_prior = gap_p_dens
gap_p_dens_prior = gap_p_dens_prior + geom_ribbon(data=gap_densC_prior_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
gap_p_dens_prior = gap_p_dens_prior + geom_ribbon(data=gap_densT_prior_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
ggsave(paste0(fig_path, "gap_density_w_prior.png"), gap_p_dens_prior)
"""

R"""
p_cond1_prior = p_cond_1
p_cond1_prior = p_cond1_prior + geom_ribbon(data=cond1_priorC_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
p_cond1_prior = p_cond1_prior + geom_ribbon(data=cond1_priorT_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
ggsave(paste0(fig_path, "conditinal_probability_1_w_prior.png"), p_cond1_prior)
"""

R"""
p_cond2_prior = p_cond_2
p_cond2_prior = p_cond2_prior + geom_ribbon(data=cond2_priorC_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
p_cond2_prior = p_cond2_prior + geom_ribbon(data=cond2_priorT_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
ggsave(paste0(fig_path, "conditinal_probability_2_w_prior.png"), p_cond2_prior)
"""

R"""
p_cond3_prior = p_cond_3
p_cond3_prior = p_cond3_prior + geom_ribbon(data=cond3_priorC_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
p_cond3_prior = p_cond3_prior + geom_ribbon(data=cond3_priorT_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
ggsave(paste0(fig_path, "conditinal_probability_3_w_prior.png"), p_cond3_prior)
"""

R"""
p_cond4_prior = p_cond_4
p_cond4_prior = p_cond4_prior + geom_ribbon(data=cond4_priorC_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
p_cond4_prior = p_cond4_prior + geom_ribbon(data=cond4_priorT_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
ggsave(paste0(fig_path, "conditinal_probability_4_w_prior.png"), p_cond4_prior)
"""



grids5 = range(4,   13, length=100)

cond_res5 = conditional_functional_estimation_blocked_gibbs_common_atoms(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    phi_save,
    logpC_save,
    logpT_save,
    BG,
    lambda_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    BH,
    grids5
)

condC_surv5 = cond_res5["condC"]
condT_surv5 = cond_res5["condT"]

@rput condC_surv5 condT_surv5 grids5
R"""
condC_mean_5 = apply(condC_surv5, 2, mean, na.rm=TRUE) 
condC_quan_5 = apply(condC_surv5, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_5 = apply(condT_surv5, 2, mean, na.rm=TRUE) 
condT_quan_5 = apply(condT_surv5, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

df_condC_5 = data.frame(
    x=grids5, 
    m=condC_mean_5, 
    l=condC_quan_5[1,], 
    h=condC_quan_5[2,]
)

df_condT_5 = data.frame(
    x=grids5, 
    m=condT_mean_5, 
    l=condT_quan_5[1,], 
    h=condT_quan_5[2,]
)

p_cond_5 = ggplot(df_condC_5)
p_cond_5 = p_cond_5 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
p_cond_5 = p_cond_5 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
p_cond_5 = p_cond_5 + geom_line(data=df_condT_5, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
p_cond_5 = p_cond_5 + geom_ribbon(data=df_condT_5, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
p_cond_5 = p_cond_5 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1)
ggsave(paste0(fig_path, "conditinal_probability_5.png"), p_cond_5)
"""



NvecC = dataC["Nvec"]
NvecT = dataT["Nvec"]
@rput survivalC survivalT
@rput NvecC NvecT
@rput nuC nuT
R"""
png(paste0(fig_path, "KM_cond5.png"), height=480, width=480)
cond_ori_C_5 = KM_conditional_N0(survivalC, NvecC, nuC, grids5[1])
cond_ori_T_5 = KM_conditional_N0(survivalT, NvecT, nuT, grids5[1])
plot(cond_ori_C_5, lwd=2, conf.int=FALSE, col="red")
lines(cond_ori_T_5, lwd=2, conf.int=FALSE, col="blue")
dev.off()
"""