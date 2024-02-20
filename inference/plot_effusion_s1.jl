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

configC = TOML.parsefile("../configs/effusionC_s1.TOML")

dataC = load(configC["data_file"])
gapC = dataC["gap"]
survivalC = dataC["survival"]
arrivalC = dataC["arrival"]
nuC = dataC["nu"]

fitdataC = load(configC["save_path"])
posC = fitdataC["pos"]
hyperC = fitdataC["hyper"]

nsamC = length(posC["alpha"])
nburnC = div(nsamC, 4)
nthinC = div(nsamC-nburnC,2000)
keep_indexC = [nburnC+1:nthinC:nsamC;]
nkeepC = length(keep_indexC)

alpha_saveC = posC["alpha"][keep_indexC]
L_saveC = posC["L"][keep_indexC]
n_clustersC = posC["k"][keep_indexC]
theta_saveC = posC["theta"][keep_indexC]
phi_saveC = posC["phi"][keep_indexC]
mu_theta_saveC = posC["mu_theta"][keep_indexC]
b_phi_saveC = posC["b_phi"][keep_indexC]
nl_saveC = posC["nl"][keep_indexC]
sigma2_thetaC = hyperC["sigma2_theta"]
a_phiC = hyperC["a_phi"]

zeta_saveC = posC["zeta"][keep_indexC]
tU_saveC = posC["tU"][keep_indexC]
m_clustersC = posC["g"][keep_indexC]
lambda_saveC = posC["lambda"][keep_indexC]
eta_saveC = posC["eta"][keep_indexC]
mu_lambda_saveC = posC["mu_lambda"][keep_indexC]
b_eta_saveC = posC["b_eta"][keep_indexC]
ml_saveC = posC["ml"][keep_indexC]
sigma2_lambdaC = hyperC["sigma2_lambda"]
a_etaC = hyperC["a_eta"]

Sigma_e_saveC = posC["Sigma_e"][keep_indexC,:,:]

fig_path = "//Users/yunzheli/Research/BNPJoint/figs/effusions_s1/"
if !isdir(fig_path)
    print(fig_path,"\n")
	mkdir(fig_path)
end
@rput fig_path 


surv_grids = range(0.001, 13, length=100)

survival_posC = survival_functional_estimation(
    Sigma_e_saveC,
    phi_saveC, 
    theta_saveC, 
    alpha_saveC, 
    nl_saveC, 
    n_clustersC, 
    a_phiC, 
    b_phi_saveC, 
    mu_theta_saveC, 
    sigma2_thetaC, 
    surv_grids
    )
    
survival_densC = survival_posC["d"]
survival_survC = survival_posC["s"]
survival_hazaC = survival_densC ./ survival_survC 


gap_grids = range(0.001, 10, length=100)
gap_res_posC = survival_functional_estimation(
    Sigma_e_saveC,
    eta_saveC, 
    lambda_saveC, 
    zeta_saveC, 
    ml_saveC, 
    m_clustersC, 
    a_etaC,
    b_eta_saveC, 
    mu_lambda_saveC, 
    sigma2_lambdaC, 
    gap_grids
    )
    
gap_densC = gap_res_posC["d"]
gap_survC = gap_res_posC["s"]
gap_hazaC = gap_densC ./ gap_survC

grids1 = range(0.5, 20, length=100)
grids2 = range(1, 20, length=100)
grids3 = range(2, 20, length=100)
grids4 = range(5, 20, length=100)

cond_probC_1_N0 = conditional_survival_probability(
    Sigma_e_saveC, 
    a_etaC,
    eta_saveC, 
    lambda_saveC, 
    zeta_saveC, 
    ml_saveC, 
    b_eta_saveC, 
    mu_lambda_saveC, 
    sigma2_lambdaC, 
    phi_saveC, 
    theta_saveC, 
    alpha_saveC,
    nl_saveC, 
    a_phiC,
    b_phi_saveC, 
    mu_theta_saveC, 
    sigma2_thetaC, 
    grids1 
)

cond_probC_2_N0 = conditional_survival_probability(
    Sigma_e_saveC, 
    a_etaC,
    eta_saveC, 
    lambda_saveC, 
    zeta_saveC, 
    ml_saveC, 
    b_eta_saveC, 
    mu_lambda_saveC, 
    sigma2_lambdaC, 
    phi_saveC, 
    theta_saveC, 
    alpha_saveC,
    nl_saveC, 
    a_phiC,
    b_phi_saveC, 
    mu_theta_saveC, 
    sigma2_thetaC, 
    grids2 
)

cond_probC_3_N0 = conditional_survival_probability(
    Sigma_e_saveC, 
    a_etaC,
    eta_saveC, 
    lambda_saveC, 
    zeta_saveC, 
    ml_saveC, 
    b_eta_saveC, 
    mu_lambda_saveC, 
    sigma2_lambdaC, 
    phi_saveC, 
    theta_saveC, 
    alpha_saveC,
    nl_saveC, 
    a_phiC,
    b_phi_saveC, 
    mu_theta_saveC, 
    sigma2_thetaC, 
    grids3 
)

cond_probC_4_N0 = conditional_survival_probability(
    Sigma_e_saveC, 
    a_etaC,
    eta_saveC, 
    lambda_saveC, 
    zeta_saveC, 
    ml_saveC, 
    b_eta_saveC, 
    mu_lambda_saveC, 
    sigma2_lambdaC, 
    phi_saveC, 
    theta_saveC, 
    alpha_saveC,
    nl_saveC, 
    a_phiC,
    b_phi_saveC, 
    mu_theta_saveC, 
    sigma2_thetaC, 
    grids4 
)

survival_predC, gap_predC, arrival_predC, theta_predC, phi_predC, lambda_predC, eta_predC = predict(Sigma_e_saveC, theta_saveC, phi_saveC, nl_saveC, lambda_saveC, eta_saveC, ml_saveC)


configT = TOML.parsefile("../configs/effusionT_s1.TOML")

dataT = load(configT["data_file"])
gapT = dataT["gap"]
survivalT = dataT["survival"]
arrivalT = dataT["arrival"]
nuT = dataT["nu"]

fitdataT = load(configT["save_path"])
posT = fitdataT["pos"]
hyperT = fitdataT["hyper"]

nsamT = length(posT["alpha"])
nburnT = div(nsamT, 4)
nthinT = div(nsamT-nburnT,2000)
keep_indexT = [nburnT+1:nthinT:nsamT;]
nkeepT = length(keep_indexT)

alpha_saveT = posT["alpha"][keep_indexT]
L_saveT = posT["L"][keep_indexT]
n_clustersT = posT["k"][keep_indexT]
theta_saveT = posT["theta"][keep_indexT]
phi_saveT = posT["phi"][keep_indexT]
mu_theta_saveT = posT["mu_theta"][keep_indexT]
b_phi_saveT = posT["b_phi"][keep_indexT]
nl_saveT = posT["nl"][keep_indexT]
sigma2_thetaT = hyperT["sigma2_theta"]
a_phiT = hyperT["a_phi"]

zeta_saveT = posT["zeta"][keep_indexT]
tU_saveT = posT["tU"][keep_indexT]
m_clustersT = posT["g"][keep_indexT]
lambda_saveT = posT["lambda"][keep_indexT]
eta_saveT = posT["eta"][keep_indexT]
mu_lambda_saveT = posT["mu_lambda"][keep_indexT]
b_eta_saveT = posT["b_eta"][keep_indexT]
ml_saveT = posT["ml"][keep_indexT]
sigma2_lambdaT = hyperT["sigma2_lambda"]
a_etaT = hyperT["a_eta"]

Sigma_e_saveT = posT["Sigma_e"][keep_indexT,:,:]

survival_posT = survival_functional_estimation(
    Sigma_e_saveT,
    phi_saveT, 
    theta_saveT, 
    alpha_saveT, 
    nl_saveT, 
    n_clustersT, 
    a_phiT, 
    b_phi_saveT, 
    mu_theta_saveT, 
    sigma2_thetaT, 
    surv_grids
    )
    
survival_densT = survival_posT["d"]
survival_survT = survival_posT["s"]
survival_hazaT = survival_densT ./ survival_survT 


gap_res_posT = survival_functional_estimation(
    Sigma_e_saveT,
    eta_saveT, 
    lambda_saveT, 
    zeta_saveT, 
    ml_saveT, 
    m_clustersT, 
    a_etaT, 
    b_eta_saveT, 
    mu_lambda_saveT, 
    sigma2_lambdaT, 
    gap_grids
    )
    
gap_densT = gap_res_posT["d"]
gap_survT = gap_res_posT["s"]
gap_hazaT = gap_densT ./ gap_survT

cond_probT_1_N0 = conditional_survival_probability(
    Sigma_e_saveT, 
    a_etaT,
    eta_saveT, 
    lambda_saveT, 
    zeta_saveT, 
    ml_saveT, 
    b_eta_saveT, 
    mu_lambda_saveT, 
    sigma2_lambdaT, 
    phi_saveT, 
    theta_saveT, 
    alpha_saveT,
    nl_saveT, 
    a_phiT,
    b_phi_saveT, 
    mu_theta_saveT, 
    sigma2_thetaT, 
    grids1
)

cond_probT_2_N0 = conditional_survival_probability(
    Sigma_e_saveT, 
    a_etaT,
    eta_saveT, 
    lambda_saveT, 
    zeta_saveT, 
    ml_saveT, 
    b_eta_saveT, 
    mu_lambda_saveT, 
    sigma2_lambdaT, 
    phi_saveT, 
    theta_saveT, 
    alpha_saveT,
    nl_saveT, 
    a_phiT,
    b_phi_saveT, 
    mu_theta_saveT, 
    sigma2_thetaT, 
    grids2 
)

cond_probT_3_N0 = conditional_survival_probability(
    Sigma_e_saveT, 
    a_etaT,
    eta_saveT, 
    lambda_saveT, 
    zeta_saveT, 
    ml_saveT, 
    b_eta_saveT, 
    mu_lambda_saveT, 
    sigma2_lambdaT, 
    phi_saveT, 
    theta_saveT, 
    alpha_saveT,
    nl_saveT, 
    a_phiT,
    b_phi_saveT, 
    mu_theta_saveT, 
    sigma2_thetaT, 
    grids3 
)

cond_probT_4_N0 = conditional_survival_probability(
    Sigma_e_saveT, 
    a_etaT,
    eta_saveT, 
    lambda_saveT, 
    zeta_saveT, 
    ml_saveT, 
    b_eta_saveT, 
    mu_lambda_saveT, 
    sigma2_lambdaT, 
    phi_saveT, 
    theta_saveT, 
    alpha_saveT,
    nl_saveT, 
    a_phiT,
    b_phi_saveT, 
    mu_theta_saveT, 
    sigma2_thetaT, 
    grids4 
)

survival_predT, gap_predT, arrival_predT, theta_predT, phi_predT, lambda_predT, eta_predT = predict(Sigma_e_saveT, theta_saveT, phi_saveT, nl_saveT, lambda_saveT, eta_saveT, ml_saveT)



@rput Sigma_e_saveC Sigma_e_saveT
R"""
png(paste0(fig_path,"Sigma_e_C_11_trace.png"))
plot(Sigma_e_saveC[,1,1], type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_C_11_hist.png"))
hist(Sigma_e_saveC[,1,1], main="", cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_C_12_trace.png"))
plot(Sigma_e_saveC[,1,2], type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_C_12_hist.png"))
hist(Sigma_e_saveC[,1,2], main="", cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_C_22_trace.png"))
plot(Sigma_e_saveC[,2,2], type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_C_22_hist.png"))
hist(Sigma_e_saveC[,2,2], main="", cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_T_11_trace.png"))
plot(Sigma_e_saveT[,1,1], type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_T_11_hist.png"))
hist(Sigma_e_saveT[,1,1], main="", cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_T_12_trace.png"))
plot(Sigma_e_saveT[,1,2], type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_T_12_hist.png"))
hist(Sigma_e_saveT[,1,2], main="", cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_T_22_trace.png"))
plot(Sigma_e_saveT[,2,2], type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"Sigma_e_T_22_hist.png"))
hist(Sigma_e_saveT[,2,2], main="", cex.axis=2)
dev.off() 
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

c_eC = hyperC["c_e"]
C_eC = hyperC["C_e"]
c_eT = hyperT["c_e"]
C_eT = hyperT["C_e"]

epsilon_priorC = zeros(nkeepC)
xi_priorC = zeros(nkeepC)
for i in 1:nkeepC 
    Sigma_e_priorC = rand(InverseWishart(c_eC, C_eC), 1)[1]
    re = rand(MvLogNormal(zeros(2), Sigma_e_priorC), 1)
    epsilon_priorC[i] = re[1]
    xi_priorC[i] = re[2]
end 

epsilon_priorT = zeros(nkeepT)
xi_priorT = zeros(nkeepT)
for i in 1:nkeepT 
    Sigma_e_priorT = rand(InverseWishart(c_eT, C_eT), 1)[1]
    re = rand(MvLogNormal(zeros(2), Sigma_e_priorT), 1)
    epsilon_priorT[i] = re[1]
    xi_priorT[i] = re[2]
end 

@rput epsilon_predC xi_predC
@rput epsilon_predT xi_predT
@rput epsilon_priorC xi_priorC
@rput epsilon_priorT xi_priorT
R"""
png(paste0(fig_path, "random_effects_predC.png"), height=480, width=480)
plot(epsilon_predC ~ xi_predC, main="", xlab="xi", ylab="epsilon", cex.axis=2)
points(epsilon_priorC ~ xi_priorC, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
dev.off() 

png(paste0(fig_path, "random_effects_predT.png"), height=480, width=480)
plot(epsilon_predT ~ xi_predT, main="", xlab="xi", ylab="epsilon", cex.axis=2)
points(epsilon_priorT ~ xi_priorT, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
dev.off() 

png(paste0(fig_path, "random_effects_predC_recaled.png"), height=480, width=480)
plot(epsilon_predC ~ xi_predC, main="", xlab="xi", ylab="epsilon", cex.axis=2, xlim=c(0,5), ylim=c(0,15))
points(epsilon_priorC ~ xi_priorC, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
dev.off() 

png(paste0(fig_path, "random_effects_predT_recaled.png"), height=480, width=480)
plot(epsilon_predT ~ xi_predT, main="", xlab="xi", ylab="epsilon", cex.axis=2, xlim=c(0,5), ylim=c(0,15))
points(epsilon_priorT ~ xi_priorT, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
dev.off() 
"""