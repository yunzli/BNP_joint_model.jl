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

config = TOML.parsefile("../configs/effusion_ddp_cov1_BG_v0.TOML")

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

@rput alphaC_save alphaT_save zetaC_save zetaT_save
R"""
library(ggplot2)

par(mfrow=c(2,4))
plot(alphaC_save, type='l')
plot(alphaT_save, type='l')
plot(zetaC_save, type='l')
plot(zetaT_save, type='l')

hist(alphaC_save)
hist(alphaT_save)
hist(zetaC_save)
hist(zetaT_save)
par(mfrow=c(1,1))
"""

LC_save = pos["LC"][keep_index,:]
LT_save = pos["LT"][keep_index,:]

logpC_save = pos["logpC"][keep_index,:]
logpT_save = pos["logpT"][keep_index,:]

mu_theta_save = pos["mu_theta"][keep_index]
mu_beta_save = pos["mu_beta"][keep_index,:]
b_phi_save = pos["b_phi"][keep_index]

@rput mu_theta_save mu_beta_save b_phi_save
R"""
par(mfrow = c(3,2))
plot(mu_theta_save, type='l')
hist(mu_theta_save)

plot(mu_beta_save[,1], type='l')
hist(mu_beta_save[,1])

plot(b_phi_save, type='l')
hist(b_phi_save)
par(mfrow = c(1,1))
"""

tUC_save = pos["tUC"][keep_index,:]
tUT_save = pos["tUT"][keep_index,:]

logomegaC_save = pos["logomegaC"][keep_index,:]
logomegaT_save = pos["logomegaT"][keep_index,:]

mu_lambda_save = pos["mu_lambda"][keep_index]
mu_gamma_save = pos["mu_gamma"][keep_index,:]
b_eta_save = pos["b_eta"][keep_index]

@rput mu_lambda_save mu_gamma_save b_eta_save
R"""
par(mfrow = c(3,2))
plot(mu_lambda_save, type="l")
hist(mu_lambda_save)

plot(mu_gamma_save[,1], type='l')
hist(mu_gamma_save[,1])

plot(b_eta_save, type='l')
hist(b_eta_save)
par(mfrow = c(1,1))
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

@rput epsilon_predC xi_predC
@rput epsilon_predT xi_predT
R"""
plot(epsilon_predC ~ xi_predC)
plot(epsilon_predT ~ xi_predT)
"""

@rput Sigma_e_1_save Sigma_e_2_save
R"""
par(mfrow=c(2,3))
plot(Sigma_e_1_save[,1,1], type='l')
plot(Sigma_e_1_save[,1,2], type='l')
plot(Sigma_e_1_save[,2,2], type='l')
plot(Sigma_e_2_save[,1,1], type='l')
plot(Sigma_e_2_save[,1,2], type='l')
plot(Sigma_e_2_save[,2,2], type='l')

hist(Sigma_e_1_save[,1,1])
hist(Sigma_e_1_save[,1,2])
hist(Sigma_e_1_save[,2,2])
hist(Sigma_e_2_save[,1,1])
hist(Sigma_e_2_save[,1,2])
hist(Sigma_e_2_save[,2,2])
par(mfrow=c(1,1))
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

# ngrids = length(surv_grids)
# survival_densC = zeros(nkeep, ngrids)
# survival_survC = zeros(nkeep, ngrids)
# survival_densT = zeros(nkeep, ngrids)
# survival_survT = zeros(nkeep, ngrids)

# theta_predC = zeros(nkeep)
# beta_predC = zeros(nkeep)
# phi_predC = zeros(nkeep)

# theta_predT = zeros(nkeep)
# beta_predT = zeros(nkeep)
# phi_predT = zeros(nkeep)

# z0 = [0]
# z1 = [1]

# for i in 1:nkeep
#     pC = exp.(logpC_save[i,:])
#     pT = exp.(logpT_save[i,:])
#     if i % 100 == 0
#         println(i)
#     end

#     theta = theta_save[i,:]
#     beta = beta_save[i,:,:]
#     phi = phi_save[i,:]

#     theta_predC[i] = sample(theta, Weights(pC))
#     beta_predC[i] = sample(beta[:,1], Weights(pC))
#     phi_predC[i] = sample(phi, Weights(pC))

#     theta_predT[i] = sample(theta, Weights(pT))
#     beta_predT[i] = sample(beta, Weights(pT))
#     phi_predT[i] = sample(phi, Weights(pT))

#     nrep = 10

#     for g in eachindex(surv_grids)
#         epsilon_predC = rand(LogNormal(0, sqrt(Sigma_e_1_save[i,1,1])), nrep)
#         epsilon_predT = rand(LogNormal(0, sqrt(Sigma_e_2_save[i,1,1])), nrep)

#         for i_rep in 1:nrep
#             for l in 1:BG
#                 if pC[l] > 1e-10
#                     survival_densC[i,g] += 1/nrep * pC[l] * pdf(LogLogistic(theta[l]*exp(beta[l,:]'*x0)/epsilon_predC[i_rep], phi[l]), surv_grids[g])
#                     survival_survC[i,g] += 1/nrep * pC[l] * ccdf(LogLogistic(theta[l]*exp(beta[l,:]'*x0)/epsilon_predC[i_rep], phi[l]), surv_grids[g])
#                 end
#                 if pT[l] > 1e-10
#                     survival_densT[i,g] += 1/nrep * pT[l] * pdf(LogLogistic(theta[l]*exp(beta[l,:]'*x1)/epsilon_predT[i_rep], phi[l]), surv_grids[g])
#                     survival_survT[i,g] += 1/nrep * pT[l] * ccdf(LogLogistic(theta[l]*exp(beta[l,:]'*x1)/epsilon_predT[i_rep], phi[l]), surv_grids[g])
#                 end
#             end
#         end
#     end
# end

lambda_save = pos["lambda"][keep_index,:]
gamma_save = pos["gamma"][keep_index,:,:]
eta_save = pos["eta"][keep_index,:]

gap_grids = range(0.001, 6, length=100)

z0 = [0]
z1 = [1]

# gap_densC = zeros(nkeep, ngrids)
# gap_survC = zeros(nkeep, ngrids)
# gap_densT = zeros(nkeep, ngrids)
# gap_survT = zeros(nkeep, ngrids)

# lambda_predC = zeros(nkeep)
# gamma_predC = zeros(nkeep)
# eta_predC = zeros(nkeep)

# lambda_predT = zeros(nkeep)
# gamma_predT = zeros(nkeep)
# eta_predT = zeros(nkeep)


# for i in 1:nkeep
#     omegaC = exp.(logomegaC_save[i,:])
#     omegaT = exp.(logomegaT_save[i,:])
#     if i % 100 == 0
#         println(i)
#     end

#     lambda = lambda_save[i,:]
#     gamma = gamma_save[i,:,:]
#     eta = eta_save[i,:]

#     lambda_predC[i] = sample(lambda, Weights(omegaC))
#     gamma_predC[i] = sample(gamma[:,1], Weights(omegaC))
#     eta_predC[i] = sample(eta, Weights(omegaC))

#     lambda_predT[i] = sample(lambda, Weights(omegaT))
#     gamma_predT[i] = sample(gamma, Weights(omegaT))
#     eta_predT[i] = sample(eta, Weights(omegaT))

#     nrep = 10

#     for g in eachindex(gap_grids)
#         xi_predC = rand(LogNormal(0, sqrt(Sigma_e_1_save[i,2,2])), nrep)
#         xi_predT = rand(LogNormal(0, sqrt(Sigma_e_2_save[i,2,2])), nrep)

#         for i_rep in 1:nrep
#             for l in 1:BH
#                 if omegaC[l] > 1e-10
#                     gap_densC[i,g] += 1/nrep * omegaC[l] * pdf(LogLogistic(lambda[l]*exp(gamma[l,:]'*x0)/xi_predC[i_rep], eta[l]), gap_grids[g])
#                     gap_survC[i,g] += 1/nrep * omegaC[l] * ccdf(LogLogistic(lambda[l]*exp(gamma[l,:]'*x0)/xi_predC[i_rep], eta[l]), gap_grids[g])
#                 end
#                 if omegaT[l] > 1e-10
#                     gap_densT[i,g] += 1/nrep * omegaT[l] * pdf(LogLogistic(lambda[l]*exp(gamma[l,:]'*x1)/xi_predT[i_rep], eta[l]), gap_grids[g])
#                     gap_survT[i,g] += 1/nrep * omegaT[l] * ccdf(LogLogistic(lambda[l]*exp(gamma[l,:]'*x1)/xi_predT[i_rep], eta[l]), gap_grids[g])
#                 end
#             end
#         end
#     end
# end

# @rput gap_grids
# @rput gap_densC gap_densT gap_survC gap_survT
# R"""
# gap_densC_mean = apply(gap_densC, 2, mean)
# gap_survC_mean = apply(gap_survC, 2, mean)
# gap_densT_mean = apply(gap_densT, 2, mean)
# gap_survT_mean = apply(gap_survT, 2, mean)

# gap_densC_quan = apply(gap_densC, 2, quantile, prob=c(0.025, 0.975))
# gap_survC_quan = apply(gap_survC, 2, quantile, prob=c(0.025, 0.975))
# gap_densT_quan = apply(gap_densT, 2, quantile, prob=c(0.025, 0.975))
# gap_survT_quan = apply(gap_survT, 2, quantile, prob=c(0.025, 0.975))
# """

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

@rput theta_predC beta_predC phi_predC
@rput theta_predT beta_predT phi_predT
@rput lambda_predC gamma_predC eta_predC
@rput lambda_predT gamma_predT eta_predT
R"""
png(paste0(fig_path, "theta_predC.png"), height=480, width=480)
hist(unlist(theta_predC), main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,120,by=3))
dev.off() 

png(paste0(fig_path, "phi_predC.png"), height=480, width=480)
hist(unlist(phi_predC), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
dev.off() 

surv_scaleT = unlist(theta_predT) * exp(beta_predT[,1])
png(paste0(fig_path, "surv_scale_predT.png"), height=480, width=480)
hist(surv_scaleT, main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,150,by=3))
dev.off() 

png(paste0(fig_path, "theta_predT.png"), height=480, width=480)
hist(unlist(theta_predT), main="", xlab="", cex.axis=2, xlim=c(0,30), breaks=seq(0,120,by=3))
dev.off() 

png(paste0(fig_path, "phi_predT.png"), height=480, width=480)
hist(unlist(phi_predT), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
dev.off() 

png(paste0(fig_path, "lambda_predC.png"), height=480, width=480)
hist(unlist(lambda_predC), main="", xlab="", cex.axis=2, xlim=c(0,60), breaks=seq(0,330,by=3))
dev.off() 
png(paste0(fig_path, "eta_predC.png"), height=480, width=480)
hist(unlist(eta_predC), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
dev.off() 

gap_scaleT = unlist(lambda_predT) * exp(unlist(gamma_predT[,1]))
png(paste0(fig_path, "gap_scaleT.png"), height=480, width=480)
hist(gap_scaleT, main="", xlab="", cex.axis=2, xlim=c(0,60), breaks=seq(0,1266,by=3))
dev.off()

png(paste0(fig_path, "lambda_predT.png"), height=480, width=480)
hist(unlist(lambda_predT), main="", xlab="", cex.axis=2, xlim=c(0,60), breaks=seq(0,330,by=3))
dev.off() 

png(paste0(fig_path, "eta_predT.png"), height=480, width=480)
hist(unlist(eta_predT), main="", xlab="", cex.axis=2, xlim=c(0,20), breaks=seq(0,120,by=2))
dev.off() 
"""


grids1 = range(0.5, 20, length=100)
grids2 = range(1, 20, length=100)
grids3 = range(2, 20, length=100)
grids4 = range(5, 20, length=100)

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

cond_res4 = conditional_functional_estimation_blocked_gibbs_cov(
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
    grids4
)

condC_dens4 = cond_res4["condC"]
condT_dens4 = cond_res4["condT"]

@rput condC_dens4 condT_dens4 grids4
R"""
condC_mean_4 = apply(condC_dens4, 2, mean, na.rm=TRUE) 
condC_quan_4 = apply(condC_dens4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
condT_mean_4 = apply(condT_dens4, 2, mean, na.rm=TRUE) 
condT_quan_4 = apply(condT_dens4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

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