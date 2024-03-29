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

# config = TOML.parsefile("../configs/effusion_ddp_cov1_BG.TOML")
config = TOML.parsefile("../configs/effusion_ddp_cov1_BG_s1.TOML")
# config = TOML.parsefile("../configs/effusion_ddp_cov1_BG_s2.TOML")
# config = TOML.parsefile("../configs/effusion_ddp_cov1_BG_s3.TOML")
# config = TOML.parsefile("../configs/effusion_ddp_cov1_BG_s4.TOML")

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
NvecC = dataC["Nvec"]
nC = length(survivalC)

dataT = load(config["data_fileT"])
gapT = dataT["gap"]
survivalT = dataT["survival"]
arrivalT = dataT["arrival"]
nuT = dataT["nu"]
NvecT = dataT["Nvec"]
nT = length(survivalT)


@rput survivalC arrivalC nuC nC
@rput survivalT arrivalT nuT nT
R"""
survivalC = unlist(survivalC)
survivalT = unlist(survivalT)
nuC = unlist(nuC)
nuT = unlist(nuT)
"""

R"""
png("//Users/yunzheli/Research/BNPJoint/data_visualization.png", width=960, height=720)
plot(1:nC ~ survivalC, col="red", pch=-nuC+2, ylim=c(0,nC+nT+1), main="", xlab="survival time", ylab="subject", cex=0.5, cex.axis=2, cex.lab=2)
for(i in 1:nC){
    tmp = c(0)
    if(length(arrivalC[i]) > 0){
        for(j in 1:length(arrivalC[i])){
            points(i ~ arrivalC[[i]][j], cex=1, pch=16)
        }
    }
}

points((nC+1):(nC+nT) ~ survivalT, col="blue", pch=-nuT+2, cex=0.5)
for(i in 1:nT){
    tmp = c(0)
    if(length(arrivalT[i]) > 0){
        for(j in 1:length(arrivalT[i])){
            points(nC+i ~ arrivalT[[i]][j], cex=1, pch=16)
        }
    }
}

legend("topright", title="observation", # << THIS IS THE HACKISH PART
       legend=c("observed","censored","effusion"), 
       pch = c(1, 2, 16),
       ncol=2, cex=2)

legend("topright", inset=c(0, 0.2), title="treatment", # << THIS IS THE HACKISH PART
       legend=c("3DRT","IMRT"), 
       col = c("red", "blue"), pch = 0,
       ncol=2, cex=2)

dev.off()
"""

@rput NvecC NvecT
R"""
NvecC = unlist(NvecC)
NvecT = unlist(NvecT)
"""
R"""
png("//Users/yunzheli/Research/BNPJoint/eff_surv_C.png")
plot(NvecC ~ survivalC, ylab="number of effusions", xlab="survival time")
dev.off()

png("//Users/yunzheli/Research/BNPJoint/eff_surv_T.png")
plot(NvecT ~ survivalT, ylab="number of effusions", xlab="survival time")
dev.off()

png("//Users/yunzheli/Research/BNPJoint/eff_surv.png")
plot(c(NvecC, NvecT) ~ c(survivalC, survivalT), ylab="number of effusions", xlab="survival time")
dev.off()
"""

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


a1, b1 = hyper["a1"], hyper["b1"]
c1 = hyper["c1"]
zeta0_prior = rand(Beta(a1, b1), 2000)
zeta_prior = zeros(2000)
for i in 1:2000
    zeta_prior[i] = rand(Pareto(c1, zeta0_prior[i]), 1)[1]
end

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

png(paste0(fig_path,"mu_beta_trace.png"))
plot(mu_beta_save, type='l', cex.axis=2)
dev.off() 

mu_beta_grids = seq(min(mu_beta_save), max(mu_beta_save), length=200)
mu_beta_dens = dnorm(mu_beta_grids, s_beta, sqrt(S_beta))

png(paste0(fig_path,"mu_beta_hist.png"))
hist(mu_beta_save, main="", cex.axis=2, freq=FALSE)
lines(mu_beta_dens ~ mu_beta_grids, type="l", col="red")
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

png(paste0(fig_path,"mu_gamma_trace.png"))
plot(mu_gamma_save, type='l', cex.axis=2)
dev.off() 

mu_gamma_grids = seq(min(mu_gamma_save), max(mu_gamma_save), length=200)
mu_gamma_dens = dnorm(mu_gamma_grids, s_gamma, sqrt(S_gamma))

png(paste0(fig_path,"mu_gamma_hist.png"))
hist(mu_gamma_save, main="", cex.axis=2, freq=FALSE)
lines(mu_gamma_dens ~ mu_gamma_grids, type="l", col="red")
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
beta_save = pos["beta"][keep_index,:,:]
phi_save = pos["phi"][keep_index,:]

surv_grids = range(0.001, 13, length=100)

x0 = [0]
x1 = [1]

surv_res = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,1,1], 
    Sigma_e_2_save[:,1,1],
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x0,
    x1,
    surv_grids,
    BG
    )

survival_densC = surv_res["densC"]
survival_survC = surv_res["survC"]
survival_densT = surv_res["densT"]
survival_survT = surv_res["survT"]

gap_grids = range(0.001, 10, length=100)

lambda_save = pos["lambda"][keep_index,:]
gamma_save = pos["gamma"][keep_index,:,:]
eta_save = pos["eta"][keep_index,:]

z0 = [0]
z1 = [1]

gap_res = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save[:,2,2], 
    Sigma_e_2_save[:,2,2],
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z0,
    z1,
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
survival_p_dens = survival_p_dens + geom_line(aes(x=x,y=m), color="red", linetype="dashed", linewidth=1.5)
survival_p_dens = survival_p_dens + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) 
survival_p_dens = survival_p_dens + geom_line(data=survival_df_densT, aes(x=x,y=m), color="blue", linetype="dashed", linewidth=1.5)
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
survival_p_surv = survival_p_surv + geom_line(aes(x=x,y=m), color="red", linetype="dashed", linewidth=1.5)
survival_p_surv = survival_p_surv + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5)
survival_p_surv = survival_p_surv + geom_line(data=survival_df_survT, aes(x=x,y=m), color="blue",  linetype="dashed", linewidth=1.5)
survival_p_surv = survival_p_surv + geom_ribbon(data=survival_df_survT, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
survival_p_surv = survival_p_surv + ylab("Survival") + xlab("t") + theme_bw(base_size=25)
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
    x=gap_grids, 
    m=gap_surv_meanC, 
    l=gap_surv_quanC[1,], 
    h=gap_surv_quanC[2,]
    )

gap_df_survT = data.frame(
    x=gap_grids, 
    m=gap_surv_meanT, 
    l=gap_surv_quanT[1,], 
    h=gap_surv_quanT[2,]
    )
gap_p_surv = ggplot(gap_df_survC) 
gap_p_surv = gap_p_surv + geom_line(aes(x=x,y=m), color="red", linetype="dashed", size=1.5)
gap_p_surv = gap_p_surv + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
gap_p_surv = gap_p_surv + geom_line(data=gap_df_survT, aes(x=x,y=m), color="blue",  linetype="dashed", size=1.5)
gap_p_surv = gap_p_surv + geom_ribbon(data=gap_df_survT, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5)
gap_p_surv = gap_p_surv + ylab("gap") + xlab("t") + ylim(0,1.01)
ggsave(paste0(fig_path, "gap_survival.png"), gap_p_surv)
"""

grids1 = range(0.5, 13, length=100)
grids2 = range(2.5, 13, length=100)
grids3 = range(5,   13, length=100)

cond_res1 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x0,
    x1,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z0,
    z1,
    BH,
    grids1
)

condC_dens1 = cond_res1["condC"]
condT_dens1 = cond_res1["condT"]

@rput condC_dens1 condT_dens1 grids1
R"""
condC_mean_1 = apply(condC_dens1, 2, mean, na.rm=TRUE) 
condC_quan_1 = apply(condC_dens1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_1 = apply(condT_dens1, 2, mean, na.rm=TRUE) 
condT_quan_1 = apply(condT_dens1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

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

cond_res2 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x0,
    x1,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z0,
    z1,
    BH,
    grids2
)

condC_dens2 = cond_res2["condC"]
condT_dens2 = cond_res2["condT"]

@rput condC_dens2 condT_dens2 grids2
R"""
condC_mean_2 = apply(condC_dens2, 2, mean, na.rm=TRUE) 
condC_quan_2 = apply(condC_dens2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_2 = apply(condT_dens2, 2, mean, na.rm=TRUE) 
condT_quan_2 = apply(condT_dens2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

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

cond_res3 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_save,
    Sigma_e_2_save,
    theta_save,
    beta_save,
    phi_save,
    logpC_save,
    logpT_save,
    x0,
    x1,
    BG,
    lambda_save,
    gamma_save,
    eta_save,
    logomegaC_save,
    logomegaT_save,
    z0,
    z1,
    BH,
    grids3
)

condC_dens3 = cond_res3["condC"]
condT_dens3 = cond_res3["condT"]

@rput condC_dens3 condT_dens3 grids3
R"""
condC_mean_3 = apply(condC_dens3, 2, mean, na.rm=TRUE) 
condC_quan_3 = apply(condC_dens3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_3 = apply(condT_dens3, 2, mean, na.rm=TRUE) 
condT_quan_3 = apply(condT_dens3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

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

# cond_res4 = conditional_functional_estimation_blocked_gibbs_cov(
#     Sigma_e_1_save,
#     Sigma_e_2_save,
#     theta_save,
#     beta_save,
#     phi_save,
#     logpC_save,
#     logpT_save,
#     x0,
#     x1,
#     BG,
#     lambda_save,
#     gamma_save,
#     eta_save,
#     logomegaC_save,
#     logomegaT_save,
#     z0,
#     z1,
#     BH,
#     grids4
# )

# condC_dens4 = cond_res4["condC"]
# condT_dens4 = cond_res4["condT"]

# @rput condC_dens4 condT_dens4 grids4
# R"""
# condC_mean_4 = apply(condC_dens4, 2, mean, na.rm=TRUE) 
# condC_quan_4 = apply(condC_dens4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
# condT_mean_4 = apply(condT_dens4, 2, mean, na.rm=TRUE) 
# condT_quan_4 = apply(condT_dens4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

# df_condC_4 = data.frame(
#     x=grids4, 
#     m=condC_mean_4, 
#     l=condC_quan_4[1,], 
#     h=condC_quan_4[2,]
# )

# df_condT_4 = data.frame(
#     x=grids4, 
#     m=condT_mean_4, 
#     l=condT_quan_4[1,], 
#     h=condT_quan_4[2,]
# )

# p_cond_4 = ggplot(df_condC_4) 
# p_cond_4 = p_cond_4 + geom_line(aes(x=x,y=m), color="red", linetype="dashed", linewidth=1.5)
# p_cond_4 = p_cond_4 + geom_ribbon(aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.5) + theme_bw(base_size=25)
# p_cond_4 = p_cond_4 + geom_line(data=df_condT_4, aes(x=x,y=m), color="blue", linetype="dashed", linewidth=1.5)
# p_cond_4 = p_cond_4 + geom_ribbon(data=df_condT_4, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) + theme_bw(base_size=25)
# p_cond_4 = p_cond_4 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1) 
# ggsave(paste0(fig_path, "conditinal_probability_4.png"), p_cond_4)
# """


pred_tmp = predict_blocked_gibbs_cov(logpC_save, logpT_save, theta_save, beta_save, phi_save, BG, logomegaC_save, logomegaT_save, lambda_save, gamma_save, eta_save, BH)

theta_predC = pred_tmp["thetaC"] 
theta_predT = pred_tmp["thetaT"] 
beta_predC = pred_tmp["betaC"] 
beta_predT = pred_tmp["betaT"] 
phi_predC = pred_tmp["phiC"] 
phi_predT = pred_tmp["phiT"] 
lambda_predC = pred_tmp["lambdaC"] 
lambda_predT = pred_tmp["lambdaT"] 
gamma_predC = pred_tmp["gammaC"] 
gamma_predT = pred_tmp["gammaT"] 
eta_predC = pred_tmp["etaC"] 
eta_predT = pred_tmp["etaT"]

prediction_tmp = prediction_blocked_gibbs_cov(theta_predC, beta_predC, phi_predC, lambda_predC, gamma_predC, eta_predC, epsilon_predC, xi_predC, x0, z0, theta_predT, beta_predT, phi_predT, lambda_predT, gamma_predT, eta_predT, epsilon_predT, xi_predT, x1, z1)
survival_predC = prediction_tmp["survivalC"]
survival_predT = prediction_tmp["survivalT"]
gap_predC = prediction_tmp["gapC"]
gap_predT = prediction_tmp["gapT"]
Nvec_predC = prediction_tmp["NvecC"]
Nvec_predT = prediction_tmp["NvecT"]

@rput theta_predC beta_predC phi_predC lambda_predC gamma_predC eta_predC
@rput theta_predT beta_predT phi_predT lambda_predT gamma_predT eta_predT
R"""
png(paste0(fig_path, "theta_predC.png"), height=480, width=480)
hist(unlist(theta_predC), main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,max(unlist(theta_predC))+3,by=3))
dev.off() 

png(paste0(fig_path, "phi_predC.png"), height=480, width=480)
hist(unlist(phi_predC), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,max(unlist(phi_predC)+2),by=2))
dev.off() 

surv_scaleT = unlist(theta_predT) * exp(beta_predT[,1])
png(paste0(fig_path, "surv_scale_predT.png"), height=480, width=480)
hist(surv_scaleT, main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,max(surv_scaleT)+3,by=3))
dev.off() 

png(paste0(fig_path, "phi_predT.png"), height=480, width=480)
hist(unlist(phi_predT), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,max(unlist(phi_predT)+2),by=2))
dev.off() 
"""

R"""
png(paste0(fig_path, "lambda_predC.png"), height=480, width=480)
hist(unlist(lambda_predC), main="", xlab="", cex.axis=2, xlim=c(0,60), breaks=seq(0,max(unlist(lambda_predC)+3),by=3))
dev.off() 

png(paste0(fig_path, "eta_predC.png"), height=480, width=480)
hist(unlist(eta_predC), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,max(unlist(eta_predC)+2),by=2))
dev.off() 

gap_scaleT = unlist(lambda_predT) * exp(unlist(gamma_predT[,1]))
png(paste0(fig_path, "gap_scaleT.png"), height=480, width=480)
hist(gap_scaleT, main="", xlab="", cex.axis=2, xlim=c(0,60), breaks=seq(0,max(gap_scaleT)+3,by=3))
dev.off()

png(paste0(fig_path, "eta_predT.png"), height=480, width=480)
hist(unlist(eta_predT), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,max(eta_predT)+2,by=2))
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

Sigma_beta = hyper["Sigma_beta"]
mu_beta_prior = rand(MvNormal(s_beta, S_beta), 2000)
beta_prior = zeros(2000, BG, 1)
for i in 1:2000
    for l in 1:BG
        beta_prior[i,l,:] = rand(MvNormal(vec(mu_beta_prior[:,i]), Sigma_beta), 1)
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

Sigma_gamma = hyper["Sigma_gamma"]
mu_gamma_prior = rand(MvNormal(s_gamma, S_gamma), 2000)
gamma_prior = zeros(2000, BH, 1)
for i in 1:2000
    for l in 1:BH
        gamma_prior[i,l,:] = rand(MvNormal(vec(mu_gamma_prior[:,i]), Sigma_gamma), 1)
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

surv_prior = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_prior[:,1,1], 
    Sigma_e_2_prior[:,1,1],
    theta_prior,
    beta_prior,
    phi_prior,
    logpC_prior,
    logpT_prior,
    x0,
    x1,
    surv_grids,
    BG
)

surv_densC_prior = surv_prior["densC"]
surv_survC_prior = surv_prior["survC"]
surv_densT_prior = surv_prior["densT"]
surv_survT_prior = surv_prior["survT"]

gap_prior = functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_prior[:,2,2], 
    Sigma_e_2_prior[:,2,2],
    lambda_prior,
    gamma_prior,
    eta_prior,
    logomegaC_prior,
    logomegaT_prior,
    z0,
    z1,
    gap_grids,
    BH
    )

gap_densC_prior = gap_prior["densC"]
gap_survC_prior = gap_prior["survC"]
gap_densT_prior = gap_prior["densT"]
gap_survT_prior = gap_prior["survT"]

cond_prior1 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_prior,
    Sigma_e_2_prior,
    theta_prior,
    beta_prior,
    phi_prior,
    logpC_prior,
    logpT_prior,
    x0,
    x1,
    BG,
    lambda_prior,
    gamma_prior,
    eta_prior,
    logomegaC_prior,
    logomegaT_prior,
    z0,
    z1,
    BH,
    grids1
)

cond1_priorC = cond_prior1["condC"]
cond1_priorT = cond_prior1["condT"]

cond_prior2 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_prior,
    Sigma_e_2_prior,
    theta_prior,
    beta_prior,
    phi_prior,
    logpC_prior,
    logpT_prior,
    x0,
    x1,
    BG,
    lambda_prior,
    gamma_prior,
    eta_prior,
    logomegaC_prior,
    logomegaT_prior,
    z0,
    z1,
    BH,
    grids2
)

cond2_priorC = cond_prior2["condC"]
cond2_priorT = cond_prior2["condT"]

cond_prior3 = conditional_functional_estimation_blocked_gibbs_cov(
    Sigma_e_1_prior,
    Sigma_e_2_prior,
    theta_prior,
    beta_prior,
    phi_prior,
    logpC_prior,
    logpT_prior,
    x0,
    x1,
    BG,
    lambda_prior,
    gamma_prior,
    eta_prior,
    logomegaC_prior,
    logomegaT_prior,
    z0,
    z1,
    BH,
    grids3
)

cond3_priorC = cond_prior3["condC"]
cond3_priorT = cond_prior3["condT"]

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

@rput surv_densC_prior surv_densT_prior
@rput surv_survC_prior surv_survT_prior
@rput gap_densC_prior gap_densT_prior
@rput gap_survC_prior gap_survT_prior
@rput cond1_priorC cond1_priorT
@rput cond2_priorC cond2_priorT
@rput cond3_priorC cond3_priorT
# @rput cond4_priorC cond4_priorT
R"""
surv_densC_prior_quan = apply(surv_densC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
surv_densC_prior_quan[which(surv_densC_prior_quan > 1)] = 1
surv_densC_prior_df = data.frame(
    x = surv_grids, 
    l = surv_densC_prior_quan[1,],
    h = surv_densC_prior_quan[2,]
)
surv_densT_prior_quan = apply(surv_densT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
surv_densT_prior_quan[which(surv_densT_prior_quan > 1)] = 1
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

gap_survC_prior_quan = apply(gap_survC_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
gap_survC_prior_df = data.frame(
    x = gap_grids, 
    l = gap_survC_prior_quan[1,],
    h = gap_survC_prior_quan[2,]
)
gap_survT_prior_quan = apply(gap_survT_prior, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
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
"""

R"""
survival_p_dens_prior = survival_p_dens
survival_p_dens_prior = survival_p_dens_prior + geom_ribbon(data=surv_densC_prior_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
survival_p_dens_prior = survival_p_dens_prior + geom_ribbon(data=surv_densT_prior_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) + ylim(0,1)
ggsave(paste0(fig_path, "survival_density_w_prior.png"), survival_p_dens_prior)
"""

R"""
survival_p_surv_prior = survival_p_surv
survival_p_surv_prior = survival_p_surv_prior + geom_ribbon(data=surv_survC_prior_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
survival_p_surv_prior = survival_p_surv_prior + geom_ribbon(data=surv_survT_prior_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
ggsave(paste0(fig_path, "survival_survival_w_prior.png"), survival_p_surv_prior)
"""

R"""
gap_p_surv_prior = gap_p_surv
gap_p_surv_prior = gap_p_surv_prior + geom_ribbon(data=gap_survC_prior_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
gap_p_surv_prior = gap_p_surv_prior + geom_ribbon(data=gap_survT_prior_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
ggsave(paste0(fig_path, "gap_survival_w_prior.png"), gap_p_surv_prior) 
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

# R"""
# p_cond4_prior = p_cond_4
# p_cond4_prior = p_cond4_prior + geom_ribbon(data=cond4_priorC_df, aes(x=x,ymin=l,ymax=h), fill="red", alpha=0.2) 
# p_cond4_prior = p_cond4_prior + geom_ribbon(data=cond4_priorT_df, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.2) 
# ggsave(paste0(fig_path, "conditinal_probability_4_w_prior.png"), p_cond4_prior)
# """


xiC_pos = pos["xiC"][keep_index,:]
Nvec_2surv_predC = zeros(length(survivalC), nkeep)
gap_2surv_predC = []
for i in eachindex(survivalC)
    tmp = predict_recurrent_blocked_gibbs_cov(lambda_save, eta_save, gamma_save, logomegaC_save, xiC_pos[:,i], z0, survivalC[i])
    Nvec_2surv_predC[i,:] = tmp[3]
    push!(gap_2surv_predC, tmp[2])
end

xiT_pos = pos["xiT"][keep_index,:]
Nvec_2surv_predT = zeros(length(survivalT), nkeep)
gap_2surv_predT = []
for i in eachindex(survivalT)
    tmp = predict_recurrent_blocked_gibbs_cov(lambda_save, eta_save, gamma_save, logomegaT_save, xiT_pos[:,i], z1, survivalT[i])
    Nvec_2surv_predT[i,:] = tmp[3]
    push!(gap_2surv_predT, tmp[2])
end

@rput Nvec_2surv_predC NvecC
R"""
NvecC_median = apply(Nvec_2surv_predC, 1, median)
NvecC_mean = apply(Nvec_2surv_predC, 1, mean)
NvecC = unlist(NvecC)
Nvec_y_upperC = max(c(max(table(NvecC)), max(table(NvecC_median))))
Nvec_x_upperC = max(c(max(NvecC), max(NvecC_median)))
"""

@rput Nvec_2surv_predT NvecT
R"""
NvecT_median = apply(Nvec_2surv_predT, 1, median)
NvecT_mean = apply(Nvec_2surv_predT, 1, mean)
NvecT = unlist(NvecT)
Nvec_y_upperT = max(c(max(table(NvecT)), max(table(NvecT_median))))
Nvec_x_upperT = max(c(max(NvecT), max(NvecT_median)))
"""


R"""
png(paste0(fig_path, "numC.png"))
hist(NvecC_median, breaks=seq(-0.5,Nvec_x_upperC+1,by=1), ylim=c(0,Nvec_y_upperC), col=rgb(1,0,0,0.25), main="Number of effusions", xlab="")
hist(NvecC, breaks=seq(-0.5,Nvec_x_upperC+1,by=1), add=TRUE, col=rgb(0,0,1,0.25))
legend("topright", legend=c("predicted", "observed"), fill=c(rgb(1,0,0,0.25), rgb(0,0,1,0.25))) 
dev.off()
"""

R"""
png(paste0(fig_path, "numT.png"))
hist(NvecT_median, breaks=seq(-0.5,Nvec_x_upperT+1,by=1), ylim=c(0,Nvec_y_upperT), col=rgb(1,0,0,0.25), main="Number of effusions", xlab="")
hist(NvecT, breaks=seq(-0.5,Nvec_x_upperT+1,by=1), add=TRUE, col=rgb(0,0,1,0.25))
legend("topright", legend=c("predicted", "observed"), fill=c(rgb(1,0,0,0.25), rgb(0,0,1,0.25))) 
dev.off()
"""

R"""
Nvec_x_upper = max(Nvec_x_upperC, Nvec_x_upperT)
Nvec_y_upper = max(c(max(table(c(NvecC_median, NvecT_median))), max(table(c(NvecC, NvecT)))))
"""

R"""
png(paste0(fig_path, "num.png"))
hist(c(NvecC_median, NvecT_median), breaks=seq(-0.5,Nvec_x_upper+1,by=1), ylim=c(0,Nvec_y_upper), col=rgb(1,0,0,0.25), main="Number of effusions", xlab="")
hist(c(NvecC, NvecT), breaks=seq(-0.5,Nvec_x_upper+1,by=1), add=TRUE, col=rgb(0,0,1,0.25))
legend("topright", legend=c("predicted", "observed"), fill=c(rgb(1,0,0,0.25), rgb(0,0,1,0.25))) 
dev.off()
"""


R"""
png(paste0(fig_path, "prediction.png"))
plot(c(NvecC_median, NvecT_median) ~ c(survivalC, survivalT), main="", xlab="survival", ylab="number of effusions")
dev.off()
"""

R"""
png(paste0(fig_path, "predictionC.png"))
plot(NvecC_median ~ survivalC, main="", xlab="survival", ylab="number of effusions")
dev.off()
"""

R"""
png(paste0(fig_path, "predictionT.png"))
plot(NvecT_median ~ survivalT, main="", xlab="survival", ylab="number of effusions")
dev.off()
"""

NC = dataC["N"]
NT = dataT["N"]
@rput nC nT NC NT gapC gapT
R"""
library(survival)
survival_df_C = data.frame(
    t = survivalC, 
    status = nuC
)

survival_df_T = data.frame(
    t = survivalT,
    status = nuT
)

survival_KM_tmpC = survfit(Surv(t, status) ~ 1, data=survival_df_C)
survival_KM_tmpT = survfit(Surv(t, status) ~ 1, data=survival_df_T)

survival_KM_df_C = data.frame(
    t = survival_KM_tmpC$time,
    S = survival_KM_tmpC$surv
)

survival_KM_df_T = data.frame(
    t = survival_KM_tmpT$time,
    S = survival_KM_tmpT$surv
)


wC = c(unlist(gapC), rep(0, nC))
iotaC = c(rep(1,NC), rep(0,nC)) 
wT = c(unlist(gapT), rep(0, nT))
iotaT = c(rep(1,NT), rep(0,nT))
for(i in 1:nC){
    if(length(gapC[[i]]) == 0){
        wC[NC+i] = survivalC[i]
    }else{
        wC[NC+i] = survivalC[i] - gapC[[i]][[length(gapC[i])]]
    }
}
for(i in 1:nT){
    if(length(gapT[[i]]) == 0){
        wT[NT+i] = survivalT[i]
    }else{
        wT[NT+i] = survivalT[i] - gapT[[i]][[length(gapT[i])]]
    }
}

gap_df_C = data.frame(
    t = wC, 
    status = iotaC
)

gap_df_T = data.frame(
    t = wT,
    status = iotaT
)

gap_KM_tmpC = survfit(Surv(t, status) ~ 1, data=gap_df_C)
gap_KM_tmpT = survfit(Surv(t, status) ~ 1, data=gap_df_T)

gap_KM_df_C = data.frame(
    t = gap_KM_tmpC$time,
    S = gap_KM_tmpC$surv
)

gap_KM_df_T = data.frame(
    t = gap_KM_tmpT$time,
    S = gap_KM_tmpT$surv
)
"""

R"""
survival_p_surv_KM = survival_p_surv + geom_line(data=survival_KM_df_C, aes(x=t, y=S), color="red", linewidth=1.5) 
survival_p_surv_KM = survival_p_surv_KM + geom_line(data=survival_KM_df_T, aes(x=t, y=S), color="blue", linewidth=1.5)
ggsave(paste0(fig_path, "survival_survival_KM.png"), survival_p_surv_KM)


gap_p_surv_KM = gap_p_surv + geom_line(data=gap_KM_df_C, aes(x=t, y=S), color="red", linewidth=1.5) + xlim(0,10)
gap_p_surv_KM = gap_p_surv_KM + geom_line(data=gap_KM_df_T, aes(x=t, y=S), color="blue", linewidth=1.5)
ggsave(paste0(fig_path, "gap_survival_KM.png"), gap_p_surv_KM)
"""

R"""
source("//Users/yunzheli/Packages/BNPJointModel/src/visualizer.R")
cond_ori_C_1 = KM_conditional_N0(survivalC, NvecC, nuC, grids1[1])
cond_ori_T_1 = KM_conditional_N0(survivalT, NvecT, nuT, grids1[1])

cond_ori_C_2 = KM_conditional_N0(survivalC, NvecC, nuC, grids2[1])
cond_ori_T_2 = KM_conditional_N0(survivalT, NvecT, nuT, grids2[1])

cond_ori_C_3 = KM_conditional_N0(survivalC, NvecC, nuC, grids3[1])
cond_ori_T_3 = KM_conditional_N0(survivalT, NvecT, nuT, grids3[1])

# cond_ori_C_4 = KM_conditional_N0(survivalC, NvecC, nuC, grids4[1])
# cond_ori_T_4 = KM_conditional_N0(survivalT, NvecT, nuT, grids4[1])
"""

R"""
plot(cond_ori_C_2, lwd=2, conf.int=FALSE, col="red")
lines(cond_ori_T_2, lwd=2, conf.int=FALSE, col="blue")
"""

R"""
plot(cond_ori_C_3, lwd=2, conf.int=FALSE, col="red")
lines(cond_ori_T_3, lwd=2, conf.int=FALSE, col="blue")
"""

# R"""
# cond_KM_df_C_4 = data.frame(
#     t = cond_ori_C_4$time,
#     S = cond_ori_C_4$surv
# )

# cond_KM_df_T_4 = data.frame(
#     t = cond_ori_T_4$time,
#     S = cond_ori_T_4$surv
# )

# p_cond_4 + geom_line(data=cond_KM_df_C_4, aes(x=t, y=S), col="red") + geom_line(data=cond_KM_df_T_4, aes(x=t, y=S), col="blue")
# """

R"""
# In-sample prediction 
N95_C = apply(Nvec_2surv_predC, 1, quantile, probs=c(0.025, 0.975))
N95_T = apply(Nvec_2surv_predT, 1, quantile, probs=c(0.025, 0.975))
N95 = cbind(rbind(N95_C, NvecC), rbind(N95_T, NvecT))
ifcontain95 = apply(N95, 2, function(x) x[1] <= x[3] && x[2] >= x[3])
print(mean(ifcontain95))
"""

R"""
# In-sample prediction 
N50_C = apply(Nvec_2surv_predC, 1, quantile, probs=c(0.25, 0.75))
N50_T = apply(Nvec_2surv_predT, 1, quantile, probs=c(0.25, 0.75))
N50 = cbind(rbind(N50_C, NvecC), rbind(N50_T, NvecT))
ifcontain50 = apply(N50, 2, function(x) x[1] <= x[3] && x[2] >= x[3])
print(mean(ifcontain50))
"""

xi_posC = pos["xiC"][keep_index,:]
xi_posT = pos["xiT"][keep_index,:]
@rput xi_posC xi_posT
R"""
xi_posC_mean = apply(xi_posC, 1, mean)
xi_posT_mean = apply(xi_posT, 1, mean)

png(paste0(fig_path, "xiC_mean.png"))
plot(density(xi_posC_mean), main="", xlab="xiC")
dev.off()

png(paste0(fig_path, "xiT_mean.png"))
plot(density(xi_posT_mean), main="", xlab="xiT")
dev.off()
"""

R"""
indexC_1 = which.max(survivalC)
indexC_2 = which.max(NvecC)
indexT_1 = which.max(survivalT)
indexT_2 = which.max(NvecT)

survivalC = unlist(survivalC)
survivalT = unlist(survivalT)
nuC = unlist(nuC)
nuT = unlist(nuT)
"""

R"""
png(paste0(fig_path, "xiC_max_surv.png"))
hist(xi_posC[,indexC_1], main="id 1, 3DCRT", xlab="xiC")
dev.off()

png(paste0(fig_path, "xiC_max_Ni.png"))
hist(xi_posC[,indexC_2], main="id 2, 3DCRT", xlab="xiC")
dev.off()

png(paste0(fig_path, "xiT_max_surv.png"))
hist(xi_posT[,indexT_1], main="id 3, IMRT", xlab="xiT")
dev.off()

png(paste0(fig_path, "xiT_max_Ni.png"))
hist(xi_posT[,indexT_2], main="id 4, IMRT", xlab="xiT")
dev.off()
"""

R"""
png(paste0(fig_path, "data_visualization_4xi.png"), width=960, height=720)
plot(1 ~ survivalC[indexC_1], col="red", pch=-nuC[indexC_1]+2, ylim=c(0,5), main="", xlab="survival time", ylab="subject", cex=2, cex.axis=2, xlim=c(0,14))
text(x=survivalC[indexC_1]+1, y=1, "id 1, 3DCRT")
if(length(arrivalC[[indexC_1]]) > 0){
    for(j in 1:length(arrivalC[[indexC_1]])){
        points(y=1, x=arrivalC[[indexC_1]][j], cex=1, pch=16)
    }
}
points(x = survivalC[indexC_2], y = 2, col="blue", pch=-nuC[indexC_2]+2, cex=2)
text(x=survivalC[indexC_2]+1, y=2, "id 2, 3DCRT")
if(length(arrivalC[[indexC_2]]) > 0){
    for(j in 1:length(arrivalC[[indexC_2]])){
        points(y=2, x=arrivalC[[indexC_2]][j], cex=1, pch=16)
    }
}
points(x = survivalT[indexT_1], y = 3, col="green", pch=-nuT[indexT_1]+2, cex=2)
text(x=survivalT[indexT_1]+1, y=3, "id 3, IMRT")
if(length(arrivalT[[indexT_1]]) > 0){
    for(j in 1:length(arrivalT[[indexT_1]])){
        points(y=3, x=arrivalT[[indexT_1]][j], cex=1, pch=16)
    }
}
points(x = survivalT[indexT_2], y = 4, col="black", pch=-nuT[indexT_2]+2, cex=2)
text(x=survivalT[indexT_2]+1, y=4, "id 4, IMRT")
if(length(arrivalT[[indexT_2]]) > 0){
    for(j in 1:length(arrivalT[[indexT_2]])){
        points(y=4, x=arrivalT[[indexT_2]][j], cex=1, pch=16)
    }
}
dev.off()
"""

R"""
N_prob_C = rep(NA, nC)
N_prob_C_fuzzy = rep(NA, nC)
for(i in 1:nC){
    N_prob_C[i] = mean(NvecC[i] == Nvec_2surv_predC[,i])
    N_prob_C_fuzzy[i] = mean((NvecC[i] == Nvec_2surv_predC[,i]) | ((NvecC[i]+1) == Nvec_2surv_predC[,i]) | ((NvecC[i]-1) == Nvec_2surv_predC[,i]))
}
N_prob_T = rep(NA, nT)
N_prob_T_fuzzy = rep(NA, nT)
for(i in 1:nT){
    N_prob_T[i] = mean(NvecT[i] == Nvec_2surv_predT[,i])
    N_prob_T_fuzzy[i] = mean((NvecT[i] == Nvec_2surv_predT[,i]) | ((NvecT[i]+1) == Nvec_2surv_predT[,i]) | ((NvecT[i]-1) == Nvec_2surv_predT[,i]))
}

png(paste0(fig_path, "N_match.png"))
par(mfrow=c(1,2))
hist(N_prob_C, main="")
hist(N_prob_T, main="")
dev.off()
png(paste0(fig_path, "N_match_fuzzy.png"))
par(mfrow=c(1,2))
hist(N_prob_C_fuzzy, main="")
hist(N_prob_T_fuzzy, main="")
dev.off()
"""


function predicted_function(theta, beta, phi, epsilon, grids, x0) 
    nkeep = length(theta)
    ngrids = length(grids)

    dens = zeros(nkeep, ngrids)
    surv = zeros(nkeep, ngrids)

    for i in 1:nkeep
        for g in 1:ngrids
            dens[i,g] = pdf(LogLogistic(theta[i] * exp(beta[i,:]' * x0) / epsilon[i], phi[i]), grids[g])
            surv[i,g] = ccdf(LogLogistic(theta[i] * exp(beta[i,:]' * x0) / epsilon[i], phi[i]), grids[g])
        end
    end

    return Dict("dens" => dens, "surv" => surv)
end

pred_tmpC = predicted_function(theta_predC, beta_predC, phi_predC, epsilon_predC, surv_grids, [0])
pred_tmpT = predicted_function(theta_predT, beta_predT, phi_predT, epsilon_predT, surv_grids, [1])

dens_predC = pred_tmpC["dens"]
surv_predC = pred_tmpC["surv"]
dens_predT = pred_tmpT["dens"]
surv_predT = pred_tmpT["surv"]
@rput dens_predC surv_predC dens_predT surv_predT surv_grids
R"""
dens_predC_mean = apply(dens_predC, 2, mean)
dens_predT_mean = apply(dens_predT, 2, mean)
plot(dens_predC_mean ~ surv_grids, type='l', col="red")
lines(dens_predT_mean ~ surv_grids, col="blue")
"""

R"""
surv_predC_mean = apply(surv_predC, 2, mean)
surv_predC_quan = apply(surv_predC, 2, quantile, prob=c(0.025, 0.975))
surv_predT_mean = apply(surv_predT, 2, mean)
surv_predT_quan = apply(surv_predT, 2, quantile, prob=c(0.025, 0.975))


png(paste0(fig_path, "survival_pred_C.png"))
plot(surv_predC_mean ~ surv_grids, type='l', col="red", ylim=c(0,1), ylab="survival", xlab="t")
lines(surv_predC_quan[1,] ~ surv_grids, col="red")
lines(surv_predC_quan[2,]~  surv_grids, col="red")
lines(survival_KM_df_C$S ~ survival_KM_df_C$t, lty=2, col="red")
dev.off()

png(paste0(fig_path, "survival_pred_T.png"))
plot(surv_predT_mean ~ surv_grids, type='l', col="blue", ylim=c(0,1), ylab="survival", xlab="t")
lines(surv_predT_quan[1,] ~ surv_grids, col="blue")
lines(surv_predT_quan[2,] ~ surv_grids, col="blue")
lines(survival_KM_df_T$S ~ survival_KM_df_T$t, lty=2, col="blue")
dev.off()
"""

Nvec_2_predC = zeros(length(survivalC), nkeep)
gap_2_predC = []
for i in eachindex(survivalC)
    tmp = predict_recurrent_blocked_gibbs_cov(lambda_save, eta_save, gamma_save, logomegaC_save, xiC_pos[:,i], z0, 2)
    Nvec_2_predC[i,:] = tmp[3]
    push!(gap_2_predC, tmp[2])
end

Nvec_2_predT = zeros(length(survivalT), nkeep)
gap_2_predT = []
for i in eachindex(survivalT)
    tmp = predict_recurrent_blocked_gibbs_cov(lambda_save, eta_save, gamma_save, logomegaT_save, xiT_pos[:,i], z1, 2)
    Nvec_2_predT[i,:] = tmp[3]
    push!(gap_2_predT, tmp[2])
end

@rput Nvec_2_predC Nvec_2_predT;
R"""
N_prob2_C = rep(NA, nC)
N_prob2_C_fuzzy = rep(NA, nC)
for(i in 1:nC){
    N_prob2_C[i] = mean(NvecC[i] == Nvec_2_predC[,i])
    N_prob2_C_fuzzy[i] = mean((NvecC[i] == Nvec_2_predC[,i]) | ((NvecC[i]+1) == Nvec_2_predC[,i]) | ((NvecC[i]-1) == Nvec_2_predC[,i]))
}
N_prob2_T = rep(NA, nT)
N_prob2_T_fuzzy = rep(NA, nT)
for(i in 1:nT){
    N_prob2_T[i] = mean(NvecT[i] == Nvec_2_predT[,i])
    N_prob2_T_fuzzy[i] = mean((NvecT[i] == Nvec_2_predT[,i]) | ((NvecT[i]+1) == Nvec_2_predT[,i]) | ((NvecT[i]-1) == Nvec_2_predT[,i]))
}

png(paste0(fig_path, "N_match_2.png"))
par(mfrow=c(1,2))
hist(N_prob2_C, main="")
hist(N_prob2_T, main="")
dev.off()
png(paste0(fig_path, "N_match_fuzzy_2.png"))
par(mfrow=c(1,2))
hist(N_prob2_C_fuzzy, main="")
hist(N_prob2_T_fuzzy, main="")
dev.off()
"""