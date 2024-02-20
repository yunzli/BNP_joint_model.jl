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

config = TOML.parsefile("../configs/effusion_cov_common_atoms_blocked_gibbs_v0.TOML")

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

nsam = length(pos["alphaC"])
nburn = div(nsam, 4)
nthin = div(nsam-nburn,2000)
keep_index = [nburn+1:nthin:nsam;]
nkeep = length(keep_index)

alphaC_save = pos["alphaC"][keep_index]
alphaT_save = pos["alphaT"][keep_index]
zetaC_save = pos["zetaC"][keep_index]
zetaT_save = pos["zetaT"][keep_index]

a_alpha, b_alpha = hyper["a_alpha"], hyper["b_alpha"]
a_zeta, b_zeta = hyper["a_zeta"], hyper["b_zeta"]

@rput a_alpha b_alpha
@rput alphaC_save alphaT_save 
@rput a_zeta b_zeta
@rput zetaC_save zetaT_save
R"""
library(ggplot2)

png(paste0(fig_path, "alphaC_trace.png"))
plot(alphaC_save, type='l')
dev.off()

alphaC_grids = seq(min(alphaC_save), max(alphaC_save), length=200)
alphaC_dens = dgamma(alphaC_grids, a_alpha, scale=b_alpha)
png(paste0(fig_path, "alphaC_hist.png"))
hist(alphaC_save, freq=FALSE, cex.axis=2)
lines(alphaC_dens ~  alphaC_grids, col="red")
dev.off()

png(paste0(fig_path, "alphaT_trace.png"))
plot(alphaT_save, type='l')
dev.off()

alphaT_grids = seq(min(alphaT_save), max(alphaT_save), length=200)
alphaT_dens = dgamma(alphaT_grids, a_alpha, scale=b_alpha)
png(paste0(fig_path, "alphaT_hist.png"))
hist(alphaT_save, freq=FALSE)
lines(alphaT_dens ~  alphaT_grids, col="red")
dev.off()

png(paste0(fig_path, "zetaC_trace.png"))
plot(zetaC_save, type='l')
dev.off()

zetaC_grids = seq(min(zetaC_save), max(zetaC_save), length=200)
zetaC_dens = dgamma(zetaC_grids, a_zeta, scale=b_zeta)
png(paste0(fig_path, "zetaC_hist.png"))
hist(zetaC_save, freq=FALSE)
lines(zetaC_dens ~  zetaC_grids, col="red")
dev.off()

png(paste0(fig_path, "zetaT_trace.png"))
plot(zetaT_save, type='l')
dev.off()

zetaT_grids = seq(min(zetaT_save), max(zetaT_save), length=200)
zetaT_dens = dgamma(zetaT_grids, a_zeta, scale=b_zeta)
png(paste0(fig_path, "zetaT_hist.png"))
hist(zetaT_save, freq=FALSE)
lines(zetaT_dens ~  zetaT_grids, col="red")
dev.off()
"""

LC_save = pos["LC"][keep_index,:]
LT_save = pos["LT"][keep_index,:]

logpC_save = pos["logpC"][keep_index,:]
logpT_save = pos["logpT"][keep_index,:]

mu_theta_save = pos["mu_theta"][keep_index]
b_phi_save = pos["b_phi"][keep_index]

s_theta, S_theta, sigma2_theta = hyper["s_theta"], hyper["S_theta"], hyper["sigma2_theta"]
r_phi, R_phi, a_phi = hyper["r_phi"], hyper["R_phi"], hyper["a_phi"]
@rput s_theta S_theta sigma2_theta
@rput r_phi R_phi a_phi
@rput mu_theta_save b_phi_save
R"""
png(paste0(fig_path, "mu_theta_trace.png"))
plot(mu_theta_save, type='l')
dev.off()

mu_theta_grids = seq(min(mu_theta_save), max(mu_theta_save), length=200)
mu_theta_dens = dnorm(mu_theta_grids, s_theta, sqrt(S_theta))
png(paste0(fig_path, "mu_theta_dens.png"))
hist(mu_theta_save, freq=FALSE)
lines(mu_theta_dens ~ mu_theta_grids, col="red")
dev.off() 

png(paste0(fig_path, "b_phi_trace.png"))
plot(b_phi_save, type='l')
dev.off()

b_phi_grids = seq(0, max(b_phi_save), length=200)
b_phi_dens = invgamma::dinvgamma(b_phi_grids, r_phi, R_phi)
png(paste0(fig_path, "b_phi_dens.png"))
hist(b_phi_save, freq=FALSE)
lines(b_phi_dens ~ b_phi_grids, col="red")
dev.off()
"""

tUC_save = pos["tUC"][keep_index,:]
tUT_save = pos["tUT"][keep_index,:]

logomegaC_save = pos["logomegaC"][keep_index,:]
logomegaT_save = pos["logomegaT"][keep_index,:]

mu_lambda_save = pos["mu_lambda"][keep_index]
b_eta_save = pos["b_eta"][keep_index]

s_lambda, S_lambda, sigma2_lambda = hyper["s_lambda"], hyper["S_lambda"], hyper["sigma2_lambda"]
r_eta, R_eta, a_eta = hyper["r_eta"], hyper["R_eta"], hyper["a_eta"]
@rput s_lambda S_lambda sigma2_lambda
@rput r_eta R_eta a_eta
@rput mu_lambda_save b_eta_save
R"""
png(paste0(fig_path, "mu_lambda_trace.png"))
plot(mu_lambda_save, type="l")
dev.off() 

mu_lambda_grids = seq(min(mu_lambda_save), max(mu_lambda_save), length=200)
mu_lambda_dens = dnorm(mu_lambda_grids, s_lambda, sqrt(S_lambda))
png(paste0(fig_path, "mu_lambda_dens.png"))
hist(mu_lambda_save, main="", cex.axis=2, freq=FALSE)
lines(mu_lambda_dens ~ mu_lambda_grids, type="l", col="red")
dev.off()

png(paste0(fig_path, "b_eta_trace.png"))
plot(b_eta_save, type='l')
dev.off()

b_eta_grids = seq(0, max(b_eta_save), length=200)
b_eta_dens = invgamma::dinvgamma(b_eta_grids, r_eta, R_eta)
png(paste0(fig_path, main="", cex.axis=2, "b_eta_dens.png"))
hist(b_eta_save, freq=FALSE)
lines(b_eta_dens ~ b_eta_grids, col="red")
dev.off()
"""

BG = hyper["BG"]
BH = hyper["BH"]

Sigma_e_1_save = pos["Sigma_e_1"][keep_index,:,:]
Sigma_e_2_save = pos["Sigma_e_2"][keep_index,:,:]
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
survival_p_surv = survival_p_surv + ylab("Survival") + xlab("t")
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
gap_p_surv = gap_p_surv + ylab("gap") + xlab("t") + ylim(0,1)
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


pred_tmp =  predict_blocked_gibbs_common_atoms(
    logpC_save,
    logpT_save,
    theta_save,
    phi_save,
    BG,
    logomegaC_save,
    logomegaT_save,
    lambda_save,
    eta_save,
    BH
    )

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
