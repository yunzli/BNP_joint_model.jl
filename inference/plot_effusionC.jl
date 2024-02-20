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

config = TOML.parsefile("../configs/effusionC.TOML")


fitdata = load(config["save_path"])
pos = fitdata["pos"]
hyper = fitdata["hyper"]

simdata = load(config["data_file"])
survival = simdata["survival"] 
gap = simdata["gap"]
nu = simdata["nu"]
iota = simdata["iota"]

nsam = length(pos["alpha"])
nburn = div(nsam, 4)
nthin = div(nsam-nburn,2000)
keep_index = [nburn+1:nthin:nsam;]
nkeep = length(keep_index)

alpha_save = pos["alpha"][keep_index]
L_save = pos["L"][keep_index]
n_clusters = pos["k"][keep_index]
theta_save = pos["theta"][keep_index]
phi_save = pos["phi"][keep_index]
mu_theta_save = pos["mu_theta"][keep_index]
b_phi_save = pos["b_phi"][keep_index]
nl_save = pos["nl"][keep_index]
sigma2_theta = hyper["sigma2_theta"]
a_phi = hyper["a_phi"]

zeta_save = pos["zeta"][keep_index]
tU_save = pos["tU"][keep_index]
m_clusters = pos["g"][keep_index]
lambda_save = pos["lambda"][keep_index]
eta_save = pos["eta"][keep_index]
mu_lambda_save = pos["mu_lambda"][keep_index]
b_eta_save = pos["b_eta"][keep_index]
ml_save = pos["ml"][keep_index]
sigma2_lambda = hyper["sigma2_lambda"]
a_eta = hyper["a_eta"]

Sigma_e_save = pos["Sigma_e"][keep_index,:,:]

fig_path = config["fig_path"]
if !isdir(fig_path)
    print(fig_path,"\n")
	mkdir(fig_path)
end
@rput fig_path 

a_phi = hyper["a_phi"]
a_alpha, b_alpha = hyper["a_alpha"], hyper["b_alpha"]
alpha_grids = range(0.001, Base.maximum(alpha_save), length=1000)
alpha_dens = pdf.(Gamma(a_alpha, b_alpha), alpha_grids)

print("alpha: ", mean(alpha_save), " ", var(alpha_save), "\n")
@rput alpha_save alpha_grids alpha_dens
R"""
png(paste0(fig_path, "alpha_trace.png"))
plot(alpha_save, type='l')
dev.off() 

png(paste0(fig_path, "alpha_hist.png"))
hist(alpha_save, freq=FALSE, main="", xlab="alpha", cex.axis = 2)
lines(alpha_dens ~ alpha_grids)
dev.off() 
"""

a_zeta, b_zeta = hyper["a_zeta"], hyper["b_zeta"]
zeta_grids = range(0.001, Base.maximum(zeta_save), length=1000)
zeta_dens = pdf.(Gamma(a_zeta, b_zeta), zeta_grids)

print("zeta: ", mean(zeta_save), " ", var(zeta_save), "\n")
@rput zeta_save zeta_grids zeta_dens
R"""
png(paste0(fig_path, "zeta_trace.png"))
plot(zeta_save, type='l')
dev.off() 

png(paste0(fig_path, "zeta_hist.png"))
hist(zeta_save, freq=FALSE, main="", xlab="zeta", cex.axis = 2)
lines(zeta_dens ~ zeta_grids)
dev.off() 
"""

@rput survival nu 
R"""
png(paste0(fig_path, "surival.png"))
hist(survival, breaks=seq(0,ceiling(max(survival)+5),by=5))
dev.off()
"""

@rput gap 
R"""
png(paste0(fig_path, "gap.png"))
hist(unlist(gap))
dev.off() 
"""

@rput n_clusters m_clusters
R"""
png(paste0(fig_path, "n_clusters.png"))
hist(n_clusters, main="", xlab="", cex.axis = 2)
dev.off()

png(paste0(fig_path, "n_clusters_trace.png"))
plot(n_clusters, type='l')
dev.off()

png(paste0(fig_path, "m_clusters.png"))
hist(m_clusters, main="", xlab="", cex.axis = 2)
dev.off()

png(paste0(fig_path, "m_clusters_trace.png"))
plot(m_clusters, type='l')
dev.off()
"""

@rput Sigma_e_save
R"""
png(paste0(fig_path, "sigma_e_11_hist.png"))
hist(Sigma_e_save[,1,1], main="", xlab="", cex.axis = 2)
dev.off() 
png(paste0(fig_path, "sigma_e_11_trace.png"))
plot(Sigma_e_save[,1,1], type='l')
dev.off() 

png(paste0(fig_path, "sigma_e_12_hist.png"))
hist(Sigma_e_save[,1,2], main="", xlab="", cex.axis = 2)
dev.off() 
png(paste0(fig_path, "sigma_e_12_trace.png"))
plot(Sigma_e_save[,1,2], type='l')
dev.off() 

png(paste0(fig_path, "sigma_e_22_hist.png"))
hist(Sigma_e_save[,2,2], main="", xlab="", cex.axis = 2)
dev.off() 
png(paste0(fig_path, "sigma_e_22_trace.png"))
plot(Sigma_e_save[,2,2], type='l')
dev.off() 
"""

# surv_grids = range(0.001, 13, length=100)
# survival_pos = survival_functional_estimation(
#     Sigma_e_save,
#     phi_save, 
#     theta_save, 
#     alpha_save, 
#     nl_save, 
#     n_clusters, 
#     b_phi_save, 
#     mu_theta_save, 
#     sigma2_theta, 
#     surv_grids
#     )
    
# survival_dens = survival_pos["d"]
# survival_surv = survival_pos["s"]
# survival_haza = survival_dens ./ survival_surv 

# @rput surv_grids 
# @rput survival_dens survival_surv survival_haza
# R"""
# library(ggplot2)
# survival_dens_mean = apply(survival_dens, 2, mean, na.rm=TRUE) 
# survival_dens_quan = apply(survival_dens, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

# survival_df_t = data.frame(x = survival, status=nu)
# survival_df_dens = data.frame(
#     x=surv_grids, 
#     m=survival_dens_mean, 
#     l=survival_dens_quan[1,], 
#     h=survival_dens_quan[2,] 
#     )
# survival_p_dens = ggplot(survival_df_dens) 
# survival_p_dens = survival_p_dens + geom_line(aes(x=x,y=m), linetype="dashed", size=1.5)
# survival_p_dens = survival_p_dens + geom_ribbon(aes(x=x,ymin=l,ymax=h), alpha=0.5) + theme_bw(base_size=25)
# survival_p_dens = survival_p_dens + ylab("Density") + xlab("t")
# ggsave(paste0(fig_path, "survival_density.png"), survival_p_dens)
# """

# R"""
# survival_surv_mean = apply(survival_surv, 2, mean, na.rm=TRUE) 
# survival_surv_quan = apply(survival_surv, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

# survival_df_surv = data.frame(
#     x=surv_grids, 
#     m=survival_surv_mean, 
#     l=survival_surv_quan[1,], 
#     h=survival_surv_quan[2,]
#     )
# survival_p_surv = ggplot(survival_df_surv) 
# survival_p_surv = survival_p_surv + geom_line(aes(x=x,y=m), linetype="dashed", size=1.5)
# survival_p_surv = survival_p_surv + geom_ribbon(aes(x=x,ymin=l,ymax=h), alpha=0.5) + theme_bw(base_size=25)
# survival_p_surv = survival_p_surv + ylab("Survival") + xlab("t")
# ggsave(paste0(fig_path, "survival_survival.png"), survival_p_surv)
# """

# c_e = hyper["c_e"]
# C_e = hyper["C_e"]
# r_phi = hyper["r_phi"]
# R_phi = hyper["R_phi"]
# s_theta = hyper["s_theta"]
# S_theta = hyper["S_theta"]
# survival_prior = survival_functional_prior(c_e, C_e, a_alpha, b_alpha, a_phi, r_phi, R_phi, s_theta, S_theta, sigma2_theta, surv_grids)
# survival_d_prior = survival_prior["d"]
# survival_s_prior = survival_prior["s"]
# survival_h_prior = survival_d_prior ./ survival_s_prior 
# @rput survival_d_prior survival_s_prior survival_h_prior
# R"""
# survival_d_prior = na.omit(survival_d_prior)
# survival_d_prior_mean = apply(survival_d_prior, 2, mean)
# survival_d_prior_quan = apply(survival_d_prior, 2, quantile, prob=c(0.025,0.5,0.975))

# survival_df_dens_prior = data.frame(x=surv_grids, pl=survival_d_prior_quan[1,], ph=survival_d_prior_quan[3,], pm=survival_d_prior_mean)
# survival_p_dens = survival_p_dens + geom_ribbon(data=survival_df_dens_prior, aes(x=x,ymin=pl,ymax=ph), alpha=0.2)
# ggsave(paste0(fig_path, "survival_density_w_prior.png"), survival_p_dens)
# """ 



# gap_grids = range(0.001, 10, length=100)
# gap_res_pos = survival_functional_estimation(
#     Sigma_e_save,
#     eta_save, 
#     lambda_save, 
#     zeta_save, 
#     ml_save, 
#     m_clusters, 
#     b_eta_save, 
#     mu_lambda_save, 
#     sigma2_lambda, 
#     gap_grids
#     )
    
# gap_dens = gap_res_pos["d"]
# gap_surv = gap_res_pos["s"]
# gap_haza = gap_dens ./ gap_surv 

# @rput gap_grids 
# @rput gap_dens gap_surv gap_haza
# R"""
# library(ggplot2)
# gap_dens_mean = apply(gap_dens, 2, mean, na.rm=TRUE) 
# gap_dens_quan = apply(gap_dens, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

# gaps = unlist(gap)
# gap_df_t = data.frame(x = gaps)
# gap_df_dens = data.frame(
#     x=gap_grids, 
#     m=gap_dens_mean, 
#     l=gap_dens_quan[1,], 
#     h=gap_dens_quan[2,]
# )
# gap_p_dens = ggplot(gap_df_dens) 
# gap_p_dens = gap_p_dens + geom_line(aes(x=x,y=m), linetype="dashed", size=1.5)
# gap_p_dens = gap_p_dens + geom_ribbon(aes(x=x,ymin=l,ymax=h), alpha=0.5) + theme_bw(base_size=25)
# gap_p_dens = gap_p_dens + ylab("Density") + xlab("t")
# ggsave(paste0(fig_path, "gap_density.png"), gap_p_dens)
# """

# r_eta = hyper["r_eta"]
# R_eta = hyper["R_eta"]
# s_lambda = hyper["s_lambda"]
# S_lambda = hyper["S_lambda"]
# gap_prior = gap_functional_prior(c_e, C_e, a_zeta, b_zeta, a_eta, r_eta, R_eta, s_lambda, S_lambda, sigma2_lambda, gap_grids)
# gap_d_prior = gap_prior["d"]
# gap_s_prior = gap_prior["s"]
# gap_h_prior = gap_d_prior ./ gap_s_prior 
# @rput gap_d_prior gap_s_prior gap_h_prior
# R"""
# gap_d_prior = na.omit(gap_d_prior)
# gap_d_prior_mean = apply(gap_d_prior, 2, mean)
# gap_d_prior_quan = apply(gap_d_prior, 2, quantile, prob=c(0.025,0.5,0.975))

# gap_df_dens_prior = data.frame(x=gap_grids, pl=gap_d_prior_quan[1,], ph=gap_d_prior_quan[3,], pm=gap_d_prior_mean)
# gap_p_dens = gap_p_dens + geom_ribbon(data=gap_df_dens_prior, aes(x=x,ymin=pl,ymax=ph), alpha=0.2)
# ggsave(paste0(fig_path, "gap_density_w_prior.png"), gap_p_dens)
# """ 


# grids = range(2, 20, length=100)
# cond_prob = conditional_survival_probability(
#     Sigma_e_save, 
#     a_eta,
#     eta_save, 
#     lambda_save, 
#     zeta_save, 
#     ml_save, 
#     m_clusters, 
#     b_eta_save, 
#     mu_lambda_save, 
#     sigma2_lambda, 
#     phi_save, 
#     theta_save, 
#     alpha_save,
#     nl_save, 
#     n_clusters, 
#     a_phi,
#     b_phi_save, 
#     mu_theta_save, 
#     sigma2_theta, 
#     grids 
# )

# @rput grids cond_prob 
# R"""
# cond_mean = apply(cond_prob, 2, mean, na.rm=TRUE) 
# cond_quan = apply(cond_prob, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

# df_cond = data.frame(
#     x=grids, 
#     m=cond_mean, 
#     l=cond_quan[1,], 
#     h=cond_quan[2,]
# )
# p_cond = ggplot(df_cond) 
# p_cond = p_cond + geom_line(aes(x=x,y=m), linetype="dashed", size=1.5)
# p_cond = p_cond + geom_ribbon(aes(x=x,ymin=l,ymax=h), alpha=0.5) + theme_bw(base_size=25)
# ggsave(paste0(fig_path, "conditinal_probability.png"), p_cond)
# """