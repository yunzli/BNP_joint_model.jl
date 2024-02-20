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

config = TOML.parsefile("../configs/effusion_cov_all.TOML")

data = load(config["data_file"])
gap = data["gap"]
survival = data["survival"]
arrival = data["arrival"]
nu = data["nu"]

fitdata = load(config["save_path"])
pos = fitdata["pos"]
hyper = fitdata["hyper"]

nsam = length(pos["alpha"])
nburn = div(nsam, 4)
nthin = div(nsam-nburn,2000)
keep_index = [nburn+1:nthin:nsam;]
nkeep = length(keep_index)

alpha_save = pos["alpha"][keep_index]
L_save = pos["L"][keep_index]
n_clusters = pos["k"][keep_index]
theta_save = pos["theta"][keep_index]
beta_save = pos["beta"][keep_index,:]
phi_save = pos["phi"][keep_index]
mu_theta_save = pos["mu_theta"][keep_index]
mu_beta_save = pos["mu_beta"][keep_index,:]
b_phi_save = pos["b_phi"][keep_index]
nl_save = pos["nl"][keep_index]
sigma2_theta = hyper["sigma2_theta"]
Sigma_beta = hyper["Sigma_beta"]
a_phi = hyper["a_phi"]

zeta_save = pos["zeta"][keep_index]
tU_save = pos["tU"][keep_index]
m_clusters = pos["g"][keep_index]
lambda_save = pos["lambda"][keep_index]
gamma_save = pos["gamma"][keep_index,:]
eta_save = pos["eta"][keep_index]
mu_lambda_save = pos["mu_lambda"][keep_index]
mu_gamma_save = pos["mu_gamma"][keep_index,:]
b_eta_save = pos["b_eta"][keep_index]
ml_save = pos["ml"][keep_index]
sigma2_lambda = hyper["sigma2_lambda"]
Sigma_gamma = hyper["Sigma_gamma"]
a_eta = hyper["a_eta"]

Sigma_e_save = pos["Sigma_e"][keep_index,:,:]

fig_path = "//Users/yunzheli/Research/BNPJoint/figs/effusions_cov1/"
if !isdir(fig_path)
    print(fig_path,"\n")
	mkdir(fig_path)
end
@rput fig_path 


@rput alpha_save n_clusters mu_theta_save mu_beta_save b_phi_save
R"""
png(paste0(fig_path, "alpha_trace.png"), height=480, width=480)
plot(alpha_save, type='l', main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "alpha_hist.png"), height=480, width=480)
hist(alpha_save, main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "n_clusters.png"), height=480, width=480)
hist(n_clusters, main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "mu_theta_trace.png"), height=480, width=480)
plot(mu_theta_save, type='l', main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "mu_theta_hist.png"), height=480, width=480)
hist(mu_theta_save, main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "b_phi_trace.png"), height=480, width=480)
plot(b_phi_save, type="l", main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "b_phi_hist.png"), height=480, width=480)
hist(b_phi_save, main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "mu_beta_trace.png"), height=480, width=480)
plot(mu_beta_save[,1], type="l", main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "mu_beta_hist.png"), height=480, width=480)
hist(mu_beta_save[,1], main="", xlab="", cex.axis=2)
dev.off() 
"""

@rput zeta_save m_clusters mu_lambda_save mu_gamma_save b_eta_save
R"""
png(paste0(fig_path, "zeta_trace.png"), height=480, width=480)
plot(zeta_save, type='l', main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "zeta_hist.png"), height=480, width=480)
hist(zeta_save, main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "m_clusters.png"), height=480, width=480)
hist(m_clusters, main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "mu_lambda_trace.png"), height=480, width=480)
plot(mu_lambda_save, type='l', main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "mu_lambda_hist.png"), height=480, width=480)
hist(mu_lambda_save, main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "b_eta_trace.png"), height=480, width=480)
plot(b_eta_save, type="l", main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "b_eta_hist.png"), height=480, width=480)
hist(b_eta_save, main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "mu_gamma_trace.png"), height=480, width=480)
plot(mu_gamma_save[,1], type="l", main="", xlab="", cex.axis=2)
dev.off() 

png(paste0(fig_path, "mu_gamma_hist.png"), height=480, width=480)
hist(mu_gamma_save[,1], main="", xlab="", cex.axis=2)
dev.off() 
"""

surv_grids = range(0.001, 13, length=100)

xC = [0]
survival_posC = survival_functional_estimation_regression(
    Sigma_e_save,
    phi_save, 
    theta_save, 
    beta_save,
    alpha_save, 
    nl_save, 
    n_clusters, 
    a_phi, 
    b_phi_save, 
    mu_theta_save, 
    sigma2_theta, 
    mu_beta_save,
    Sigma_beta,
    surv_grids,
    xC 
    )
    
survival_densC = survival_posC["d"]
survival_survC = survival_posC["s"]
survival_hazaC = survival_densC ./ survival_survC 


zC = [0]
gap_grids = range(0.001, 10, length=100)
gap_res_posC = survival_functional_estimation_regression(
    Sigma_e_save,
    eta_save, 
    lambda_save, 
    gamma_save,
    zeta_save, 
    ml_save, 
    m_clusters, 
    a_eta,
    b_eta_save, 
    mu_lambda_save, 
    sigma2_lambda, 
    mu_gamma_save,
    Sigma_gamma,
    gap_grids,
    zC
    )
    
gap_densC = gap_res_posC["d"]
gap_survC = gap_res_posC["s"]
gap_hazaC = gap_densC ./ gap_survC

xT = [1]
survival_posT = survival_functional_estimation_regression(
    Sigma_e_save,
    phi_save, 
    theta_save, 
    beta_save,
    alpha_save, 
    nl_save, 
    n_clusters, 
    a_phi, 
    b_phi_save, 
    mu_theta_save, 
    sigma2_theta, 
    mu_beta_save,
    Sigma_beta,
    surv_grids,
    xT 
    )

survival_densT = survival_posT["d"]
survival_survT = survival_posT["s"]
survival_hazaT = survival_densT ./ survival_survT 


zT = [1]
gap_res_posT = survival_functional_estimation_regression(
    Sigma_e_save,
    eta_save, 
    lambda_save, 
    gamma_save,
    zeta_save, 
    ml_save, 
    m_clusters, 
    a_eta,
    b_eta_save, 
    mu_lambda_save, 
    sigma2_lambda, 
    mu_gamma_save,
    Sigma_gamma,
    gap_grids,
    zT
    )
    
gap_densT = gap_res_posT["d"]
gap_survT = gap_res_posT["s"]
gap_hazaT = gap_densT ./ gap_survT

grids1 = range(0.5, 20, length=100)
grids2 = range(1, 20, length=100)
grids3 = range(2, 20, length=100)
grids4 = range(5, 20, length=100)

cond_probC_1_N0, cond_probC_1_N1 = conditional_survival_probability(
    Sigma_e_saveC, 
    a_etaC,
    eta_saveC, 
    lambda_saveC, 
    zeta_saveC, 
    ml_saveC, 
    m_clustersC, 
    b_eta_saveC, 
    mu_lambda_saveC, 
    sigma2_lambdaC, 
    phi_saveC, 
    theta_saveC, 
    alpha_saveC,
    nl_saveC, 
    n_clustersC, 
    a_phiC,
    b_phi_saveC, 
    mu_theta_saveC, 
    sigma2_thetaC, 
    grids1 
)

cond_probC_2_N0, cond_probC_N1 = conditional_survival_probability(
    Sigma_e_saveC, 
    a_etaC,
    eta_saveC, 
    lambda_saveC, 
    zeta_saveC, 
    ml_saveC, 
    m_clustersC, 
    b_eta_saveC, 
    mu_lambda_saveC, 
    sigma2_lambdaC, 
    phi_saveC, 
    theta_saveC, 
    alpha_saveC,
    nl_saveC, 
    n_clustersC, 
    a_phiC,
    b_phi_saveC, 
    mu_theta_saveC, 
    sigma2_thetaC, 
    grids2 
)

cond_probC_3_N0, cond_probC_3_N1 = conditional_survival_probability(
    Sigma_e_saveC, 
    a_etaC,
    eta_saveC, 
    lambda_saveC, 
    zeta_saveC, 
    ml_saveC, 
    m_clustersC, 
    b_eta_saveC, 
    mu_lambda_saveC, 
    sigma2_lambdaC, 
    phi_saveC, 
    theta_saveC, 
    alpha_saveC,
    nl_saveC, 
    n_clustersC, 
    a_phiC,
    b_phi_saveC, 
    mu_theta_saveC, 
    sigma2_thetaC, 
    grids3 
)

cond_probC_4_N0, cond_probC_4_N1 = conditional_survival_probability(
    Sigma_e_saveC, 
    a_etaC,
    eta_saveC, 
    lambda_saveC, 
    zeta_saveC, 
    ml_saveC, 
    m_clustersC, 
    b_eta_saveC, 
    mu_lambda_saveC, 
    sigma2_lambdaC, 
    phi_saveC, 
    theta_saveC, 
    alpha_saveC,
    nl_saveC, 
    n_clustersC, 
    a_phiC,
    b_phi_saveC, 
    mu_theta_saveC, 
    sigma2_thetaC, 
    grids4 
)

survival_predC, gap_predC, arrival_predC = predict(Sigma_e_saveC, theta_saveC, phi_saveC, nl_saveC, lambda_saveC, eta_saveC, ml_saveC)



cond_probT_1_N0, cond_probT_1_N1 = conditional_survival_probability(
    Sigma_e_saveT, 
    a_etaT,
    eta_saveT, 
    lambda_saveT, 
    zeta_saveT, 
    ml_saveT, 
    m_clustersT, 
    b_eta_saveT, 
    mu_lambda_saveT, 
    sigma2_lambdaT, 
    phi_saveT, 
    theta_saveT, 
    alpha_saveT,
    nl_saveT, 
    n_clustersT, 
    a_phiT,
    b_phi_saveT, 
    mu_theta_saveT, 
    sigma2_thetaT, 
    grids1
)

cond_probT_2_N0, cond_probT_2_N1 = conditional_survival_probability(
    Sigma_e_saveT, 
    a_etaT,
    eta_saveT, 
    lambda_saveT, 
    zeta_saveT, 
    ml_saveT, 
    m_clustersT, 
    b_eta_saveT, 
    mu_lambda_saveT, 
    sigma2_lambdaT, 
    phi_saveT, 
    theta_saveT, 
    alpha_saveT,
    nl_saveT, 
    n_clustersT, 
    a_phiT,
    b_phi_saveT, 
    mu_theta_saveT, 
    sigma2_thetaT, 
    grids2 
)

cond_probT_3_N0, cond_probT_3_N1 = conditional_survival_probability(
    Sigma_e_saveT, 
    a_etaT,
    eta_saveT, 
    lambda_saveT, 
    zeta_saveT, 
    ml_saveT, 
    m_clustersT, 
    b_eta_saveT, 
    mu_lambda_saveT, 
    sigma2_lambdaT, 
    phi_saveT, 
    theta_saveT, 
    alpha_saveT,
    nl_saveT, 
    n_clustersT, 
    a_phiT,
    b_phi_saveT, 
    mu_theta_saveT, 
    sigma2_thetaT, 
    grids3 
)

cond_probT_4_N0, cond_probT_4_N1 = conditional_survival_probability(
    Sigma_e_saveT, 
    a_etaT,
    eta_saveT, 
    lambda_saveT, 
    zeta_saveT, 
    ml_saveT, 
    m_clustersT, 
    b_eta_saveT, 
    mu_lambda_saveT, 
    sigma2_lambdaT, 
    phi_saveT, 
    theta_saveT, 
    alpha_saveT,
    nl_saveT, 
    n_clustersT, 
    a_phiT,
    b_phi_saveT, 
    mu_theta_saveT, 
    sigma2_thetaT, 
    grids4 
)

survival_predT, gap_predT, arrival_predT = predict(Sigma_e_saveT, theta_saveT, phi_saveT, nl_saveT, lambda_saveT, eta_saveT, ml_saveT)



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


@rput survival_predC survival_predT 
@rput gap_predC gap_predT  
@rput arrival_predC arrival_predT
R"""
source("//Users/yunzheli/Packages/BNPJointModel/src/visualizer.R")

p_survival_predC = visualize_survival_times(survival_predC)
ggsave(paste0(fig_path, "survival_predictiveC.png"), p_survival_predC)

p_survival_predT = visualize_survival_times(survival_predT)
ggsave(paste0(fig_path, "survival_predictiveT.png"), p_survival_predT)

p_gap_predC = visualize_gap_times(gap_predC)
ggsave(paste0(fig_path, "gap_predictiveC.png"), p_gap_predC)

p_gap_predT = visualize_gap_times(gap_predT)
ggsave(paste0(fig_path, "gap_predictiveT.png"), p_gap_predT)

p_recurrentC = visualize_recurrent_events(arrival_predC, survival_predC)
ggsave(paste0(fig_path, "recurrent_predictiveC.png"), p_recurrentC)

p_recurrentT = visualize_recurrent_events(arrival_predT, survival_predT)
ggsave(paste0(fig_path, "recurrent_predictiveT.png"), p_recurrentT)
"""

Nvec_predC = length.(gap_predC)
Nvec_predT = length.(gap_predT)
@rput Nvec_predC Nvec_predT
R"""
png(paste0(fig_path, "number_of_reccurent_predC.png"))
hist(Nvec_predC, main="", xlab="Ni", cex.axis=2)
dev.off()

png(paste0(fig_path, "number_of_reccurent_predC_cut.png"))
hist(Nvec_predC[which(Nvec_predC<=10)], main="", xlab="Ni", breaks=seq(0,10,by=1), cex.axis=2)
dev.off()

png(paste0(fig_path, "number_of_reccurent_predT.png"))
hist(Nvec_predT, main="", xlab="Ni", cex.axis=2)
dev.off()

png(paste0(fig_path, "number_of_reccurent_predT_cut.png"))
hist(Nvec_predT[which(Nvec_predT<=10)], main="", xlab="Ni", breaks=seq(0,10,by=1), cex.axis=2)
dev.off()
"""


@rput survivalC survivalT 
@rput gapC gapT  
@rput arrivalC arrivalT
@rput nuC nuT 
R"""
p_survivalC = visualize_survival_times(survivalC)
ggsave(paste0(fig_path, "survival_originalC.png"), p_survivalC)

p_survivalT = visualize_survival_times(survivalT)
ggsave(paste0(fig_path, "survival_originalT.png"), p_survivalT)

p_gapC = visualize_gap_times(gapC)
ggsave(paste0(fig_path, "gap_originalC.png"), p_gapC)

p_gapT = visualize_gap_times(gapT)
ggsave(paste0(fig_path, "gap_originalT.png"), p_gapT)

p_recurrentC = visualize_recurrent_events(arrivalC, survivalC)
ggsave(paste0(fig_path, "recurrent_originalC.png"), p_recurrentC)

p_recurrentT = visualize_recurrent_events(arrivalT, survivalT)
ggsave(paste0(fig_path, "recurrent_originalT.png"), p_recurrentT)

survC_KM = survival::survfit(survival::Surv(survivalC, nuC)~1)
survT_KM = survival::survfit(survival::Surv(survivalT, nuT)~1)
png(paste0(fig_path, "KM_surv.png"), height=480, width=480)
plot(survC_KM, lwd=2, conf.int=FALSE, col="red", main="")
lines(survT_KM, lwd=2, conf.int=FALSE, col="blue")
dev.off()
"""

NvecC = dataC["Nvec"]
NvecT = dataT["Nvec"]
@rput NvecC NvecT
R"""
png(paste0(fig_path, "number_of_reccurentC.png"))
hist(NvecC, breaks=seq(0,10,by=1), main="", xlab="", cex.axis=2)
dev.off()

png(paste0(fig_path, "number_of_reccurentT.png"))
hist(NvecT, breaks=seq(0,10,by=1), main="", xlab="", cex.axis=2)
dev.off()
"""


R"""
cond_ori_C_1 = KM_conditional_N0(survivalC, NvecC, nuC, grids1[1])
cond_ori_T_1 = KM_conditional_N0(survivalT, NvecT, nuT, grids1[1])
png(paste0(fig_path, "KM_cond1.png"), height=480, width=480)
plot(cond_ori_C_1, lwd=2, conf.int=FALSE, col="red")
lines(cond_ori_T_1, lwd=2, conf.int=FALSE, col="blue")
dev.off()

png(paste0(fig_path, "KM_cond2.png"), height=480, width=480)
cond_ori_C_2 = KM_conditional_N0(survivalC, NvecC, nuC, grids2[1])
cond_ori_T_2 = KM_conditional_N0(survivalT, NvecT, nuT, grids2[1])
plot(cond_ori_C_2, lwd=2, conf.int=FALSE, col="red")
lines(cond_ori_T_2, lwd=2, conf.int=FALSE, col="blue")
dev.off()

png(paste0(fig_path, "KM_cond3.png"), height=480, width=480)
cond_ori_C_3 = KM_conditional_N0(survivalC, NvecC, nuC, grids3[1])
cond_ori_T_3 = KM_conditional_N0(survivalT, NvecT, nuT, grids3[1])
plot(cond_ori_C_3, lwd=2, conf.int=FALSE, col="red")
lines(cond_ori_T_3, lwd=2, conf.int=FALSE, col="blue")
dev.off()

png(paste0(fig_path, "KM_cond4.png"), height=480, width=480)
cond_ori_C_4 = KM_conditional_N0(survivalC, NvecC, nuC, grids4[1])
cond_ori_T_4 = KM_conditional_N0(survivalT, NvecT, nuT, grids4[1])
plot(cond_ori_C_4, lwd=2, conf.int=FALSE, col="red")
lines(cond_ori_T_4, lwd=2, conf.int=FALSE, col="blue")
dev.off()
"""



R"""
KM_predC = survival::survfit(survival::Surv(unlist(survival_predC))~1)
KM_predT = survival::survfit(survival::Surv(unlist(survival_predT))~1)
png(paste0(fig_path, "KM_pred.png"), height=480, width=480)
plot(survC_KM, lwd=2, conf.int=FALSE, col="red", main="", xlab="t")
lines(survT_KM, lwd=2, conf.int=FALSE, col="blue")
lines(KM_predC, lwd=2, lty=2, conf.int=FALSE, col="red")
lines(KM_predT, lwd=2, lty=2, conf.int=FALSE, col="blue")
legend("topright", col=c("red", "blue", "red", "blue"), lty=c(1,1,2,2), lwd=2, legend=c("3DRT, data", "IMRT, data", "3DRT, predicted data", "IMRT, predicted data"))
dev.off()
"""

# R"""
# gapC_KM = survival::survfit(survival::Surv(unlist(gapC))~)
# gap_KM_predC = survival::survfit(survival::Surv(unlist(gap_predC))~1)
# gap_KM_predT = survival::survfit(survival::Surv(unlist(gap_predT))~1)
# png(paste0(fig_path, "gap_KM_pred.png"), height=480, width=480)
# plot(survC_KM, lwd=2, conf.int=FALSE, col="red", main="", xlab="t")
# lines(survT_KM, lwd=2, conf.int=FALSE, col="blue")
# lines(KM_predC, lwd=2, lty=2, conf.int=FALSE, col="red")
# lines(KM_predT, lwd=2, lty=2, conf.int=FALSE, col="blue")
# legend("topright", col=c("red", "blue", "red", "blue"), lty=c(1,1,2,2), lwd=2, legend=c("3DRT, data", "IMRT, data", "3DRT, predicted data", "IMRT, predicted data"))
# dev.off()
# """


epsilon_predC = zeros(nkeepC)
xi_predC = zeros(nkeepC)
for i in 1:nkeepC
    re = rand(MvLogNormal(zeros(2), Sigma_e_saveC[i,:,:]), 1)
    epsilon_predC[i] = re[1]
    xi_predC[i] = re[2]
end

epsilon_predT = zeros(nkeepT)
xi_predT = zeros(nkeepT)
for i in 1:nkeepT
    re = rand(MvLogNormal(zeros(2), Sigma_e_saveT[i,:,:]), 1)
    epsilon_predT[i] = re[1]
    xi_predT[i] = re[2]
end

@rput epsilon_predC xi_predC
@rput epsilon_predT xi_predT
R"""
png(paste0(fig_path, "random_effects_predC.png"), height=480, width=480)
plot(epsilon_predC ~ xi_predC, main="", xlab="xi", ylab="epsilon")
dev.off() 

png(paste0(fig_path, "random_effects_predT.png"), height=480, width=480)
plot(epsilon_predT ~ xi_predT, main="", xlab="xi", ylab="epsilon")
dev.off() 
"""