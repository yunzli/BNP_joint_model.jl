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
include("prior_analysis_func.jl")

# 5. DPM survival and gap, 
#    bivaraite lognormal random effects fixed Sigma  
fig_path = "//Users/yunzheli/Research/BNPJoint/figs/prior/analysis5/"
if !isdir(fig_path)
    print(fig_path,"\n")
	mkdir(fig_path)
end
@rput fig_path 

alpha = 5
mu_theta = log(5)
sigma2_theta = 1
a_phi = 20 
b_phi = 20


zeta = 5
mu_lambda = log(2)
sigma2_lambda = 1
a_eta = 20
b_eta = 20


res1 = prior_dp(alpha, mu_theta, sigma2_theta, a_phi, b_phi)
w = res1["weights"]
theta = exp.(res1["atoms"][1])
phi = sqrt.(res1["atoms"][2])

res2 = prior_dp(alpha, mu_lambda, sigma2_lambda, a_eta, b_eta)
p = res2["weights"]
lambda = exp.(res2["atoms"][1])
eta = sqrt.(res2["atoms"][2])


upsilon = 5
Sigma_e_1 = 0.5 * [1 0.75; 0.75 1]
Sigma_e_2 = 0.5 * [1 0; 0 1]
Sigma_e_3 = 0.5 * [1 -0.75; -0.75 1]
res3_1 = prior_random_effects_dp(upsilon, Sigma_e_1)
res3_2 = prior_random_effects_dp(upsilon, Sigma_e_2)
res3_3 = prior_random_effects_dp(upsilon, Sigma_e_3)

@rput res3_1 res3_2 res3_3
R"""
index1 = which(res3_1$weights > 0.05)
png(paste0(fig_path, "re1.png"), width=480, height=480)
plot(res3_1$atoms[1,] ~ res3_1$atoms[2,], cex=20*res3_1$weights, xlab="xi", ylab="epsilon")
text(res3_1$atoms[1,index1] ~ res3_1$atoms[2,index1], labels=round(res3_1$weights[index1],digits=2))
dev.off()

index2 = which(res3_2$weights > 0.05)
png(paste0(fig_path, "re2.png"), width=480, height=480)
plot(res3_2$atoms[1,] ~ res3_2$atoms[2,], cex=20*res3_2$weights, xlab="xi", ylab="epsilon")
text(res3_2$atoms[1,index2] ~ res3_2$atoms[2,index2], labels=round(res3_2$weights[index2],digits=2))
dev.off()

index3 = which(res3_3$weights > 0.05)
png(paste0(fig_path, "re3.png"), width=480, height=480)
plot(res3_3$atoms[1,] ~ res3_3$atoms[2,], cex=20*res3_3$weights, xlab="xi", ylab="epsilon")
text(res3_3$atoms[1,index3] ~ res3_3$atoms[2,index3], labels=round(res3_3$weights[index3],digits=2))
dev.off()
"""

grids_1 = range(1, 20, length=100)
grids_2 = range(2, 20, length=100)
grids_3 = range(5, 20, length=100)

cond_N0_S1_grids1, cond_N1_S1_grids1 = dpm_conditional_probability_dpre(res3_1, grids_1, theta, phi, w, lambda, eta, p)
cond_N0_S1_grids2, cond_N1_S1_grids2 = dpm_conditional_probability_dpre(res3_1, grids_2, theta, phi, w, lambda, eta, p)
cond_N0_S1_grids3, cond_N1_S1_grids3 = dpm_conditional_probability_dpre(res3_1, grids_3, theta, phi, w, lambda, eta, p)

cond_N0_S2_grids1, cond_N1_S2_grids1 = dpm_conditional_probability_dpre(res3_2, grids_1, theta, phi, w, lambda, eta, p)
cond_N0_S2_grids2, cond_N1_S2_grids2 = dpm_conditional_probability_dpre(res3_2, grids_2, theta, phi, w, lambda, eta, p)
cond_N0_S2_grids3, cond_N1_S2_grids3 = dpm_conditional_probability_dpre(res3_2, grids_3, theta, phi, w, lambda, eta, p)

cond_N0_S3_grids1, cond_N1_S3_grids1 = dpm_conditional_probability_dpre(res3_3, grids_1, theta, phi, w, lambda, eta, p)
cond_N0_S3_grids2, cond_N1_S3_grids2 = dpm_conditional_probability_dpre(res3_3, grids_2, theta, phi, w, lambda, eta, p)
cond_N0_S3_grids3, cond_N1_S3_grids3 = dpm_conditional_probability_dpre(res3_3, grids_3, theta, phi, w, lambda, eta, p)


@rput grids_1 grids_2 grids_3; 
@rput cond_N0_S1_grids1 cond_N0_S1_grids2 cond_N0_S1_grids3;
@rput cond_N0_S2_grids1 cond_N0_S2_grids2 cond_N0_S2_grids3;
@rput cond_N0_S3_grids1 cond_N0_S3_grids2 cond_N0_S3_grids3;
@rput cond_N1_S1_grids1 cond_N1_S1_grids2 cond_N1_S1_grids3;
@rput cond_N1_S2_grids1 cond_N1_S2_grids2 cond_N1_S2_grids3;
@rput cond_N1_S3_grids1 cond_N1_S3_grids2 cond_N1_S3_grids3;
R"""
png(paste0(fig_path, "N0_grids1.png"), width=480, height=480)
plot(cond_N0_S1_grids1 ~ grids_1, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N0_S2_grids1 ~ grids_1, lwd=2, col="blue")
lines(cond_N0_S3_grids1 ~ grids_1, lwd=2, col="green")
abline(v=1)
legend("topright", lwd=2, col=c("red","blue","green"), legend=c("Positive", "Independent", "Negative"))
dev.off() 

png(paste0(fig_path, "N0_grids2.png"), width=480,  height=480)
plot(cond_N0_S1_grids2 ~ grids_2, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N0_S2_grids2 ~ grids_2, lwd=2, col="blue")
lines(cond_N0_S3_grids2 ~ grids_2, lwd=2, col="green")
abline(v=2)
legend("topright", lwd=2, col=c("red","blue","green"), legend=c("Positive", "Independent", "Negative"))
dev.off() 

png(paste0(fig_path, "N0_grids3.png"), width=480, height=480)
plot(cond_N0_S1_grids3 ~ grids_3, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N0_S2_grids3 ~ grids_3, lwd=2, col="blue")
lines(cond_N0_S3_grids3 ~ grids_3, lwd=2, col="green")
abline(v=5)
legend("topright", lwd=2, col=c("red","blue","green"), legend=c("Positive", "Independent", "Negative"))
dev.off()

png(paste0(fig_path, "N1_grids1.png"), width=480, height=480)
plot(cond_N1_S1_grids1 ~ grids_1, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N1_S2_grids1 ~ grids_1, lwd=2, col="blue")
lines(cond_N1_S3_grids1 ~ grids_1, lwd=2, col="green")
abline(v=1)
legend("topright", lwd=2, col=c("red","blue","green"), legend=c("Positive", "Independent", "Negative"))
dev.off() 

png(paste0(fig_path, "N1_grids2.png"), width=480, height=480)
plot(cond_N1_S1_grids2 ~ grids_2, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N1_S2_grids2 ~ grids_2, lwd=2, col="blue")
lines(cond_N1_S3_grids2 ~ grids_2, lwd=2, col="green")
abline(v=2)
legend("topright", lwd=2, col=c("red","blue","green"), legend=c("Positive", "Independent", "Negative"))
dev.off() 

png(paste0(fig_path, "N1_grids3.png"), width=480, height=480)
plot(cond_N1_S1_grids3 ~ grids_3, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N1_S2_grids3 ~ grids_3, lwd=2, col="blue")
lines(cond_N1_S3_grids3 ~ grids_3, lwd=2, col="green")
abline(v=5)
legend("topright", lwd=2, col=c("red","blue","green"), legend=c("Positive", "Independent", "Negative"))
dev.off()
"""

# N = length(w)
grids = range(0.001, 20, length=200)
surv_dens1, surv_surv1, gap_dens1, gap_surv1 = dpm_density_dpre(res3_1, grids, theta, phi, w, lambda, eta, p) 
surv_dens2, surv_surv2, gap_dens2, gap_surv2 = dpm_density_dpre(res3_2, grids, theta, phi, w, lambda, eta, p) 
surv_dens3, surv_surv3, gap_dens3, gap_surv3 = dpm_density_dpre(res3_3, grids, theta, phi, w, lambda, eta, p) 


@rput grids;
@rput surv_dens1 gap_dens1; 
@rput surv_dens2 gap_dens2; 
@rput surv_dens3 gap_dens3; 
R"""
png(paste0(fig_path, "surv_dens1.png"), width=480, height=480)
plot(surv_dens1 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off()

png(paste0(fig_path, "gap_dens1.png"), width=480, height=480)
plot(gap_dens1 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off() 

png(paste0(fig_path, "surv_dens2.png"), width=480, height=480)
plot(surv_dens2 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off()

png(paste0(fig_path, "gap_dens2.png"), width=480, height=480)
plot(gap_dens2 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off() 

png(paste0(fig_path, "surv_dens3.png"), width=480, height=480)
plot(surv_dens3 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off()

png(paste0(fig_path, "gap_dens3.png"), width=480, height=480)
plot(gap_dens3 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off() 
"""



n = 400
survivals1, gaps1, arrivals1 = dpm_data_generator_dpre(res3_1, theta, phi, w, lambda, eta, p, n)
survivals2, gaps2, arrivals2 = dpm_data_generator_dpre(res3_2, theta, phi, w, lambda, eta, p, n)
survivals3, gaps3, arrivals3 = dpm_data_generator_dpre(res3_3, theta, phi, w, lambda, eta, p, n)

@rput survivals1 survivals2 survivals3 
@rput gaps1 gaps2 gaps3 
@rput arrivals1 arrivals2 arrivals3
R"""
source("//Users/yunzheli/Packages/BNPJointModel/src/visualizer.R")

p_survivals1 = visualize_survival_times(survivals1)
ggsave(paste0(fig_path, "survival_predictive1.png"), p_survivals1)

p_survivals2 = visualize_survival_times(survivals2)
ggsave(paste0(fig_path, "survival_predictive2.png"), p_survivals2)

p_survivals3 = visualize_survival_times(survivals3)
ggsave(paste0(fig_path, "survival_predictive3.png"), p_survivals3)

p_gaps1 = visualize_gap_times(gaps1)
ggsave(paste0(fig_path, "gap_predictive1.png"), p_gaps1)

p_gaps2 = visualize_gap_times(gaps2)
ggsave(paste0(fig_path, "gap_predictive2.png"), p_gaps2)

p_gaps3 = visualize_gap_times(gaps3)
ggsave(paste0(fig_path, "gap_predictive3.png"), p_gaps3)

p_recurrent1 = visualize_recurrent_events(arrivals1, survivals1)
ggsave(paste0(fig_path, "recurrent_predictive1.png"), p_recurrent1)

p_recurrent2 = visualize_recurrent_events(arrivals2, survivals2)
ggsave(paste0(fig_path, "recurrent_predictive2.png"), p_recurrent2)

p_recurrent3 = visualize_recurrent_events(arrivals3, survivals3)
ggsave(paste0(fig_path, "recurrent_predictive3.png"), p_recurrent3)
"""