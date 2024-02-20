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


sigma2_e = 0.5 
power1 = 1
power2 = 0
power3 = -1
power4 = 2
power5 = -2

grids_1 = range(1, 20, length=100)
grids_2 = range(2, 20, length=100)
grids_3 = range(5, 20, length=100)

cond_N0_S1_grids1, cond_N1_S1_grids1 = shared_dpm_conditional_probability_fixed(sigma2_e, power1, grids_1, theta, phi, w, lambda, eta, p)
cond_N0_S1_grids2, cond_N1_S1_grids2 = shared_dpm_conditional_probability_fixed(sigma2_e, power1, grids_2, theta, phi, w, lambda, eta, p)
cond_N0_S1_grids3, cond_N1_S1_grids3 = shared_dpm_conditional_probability_fixed(sigma2_e, power1, grids_3, theta, phi, w, lambda, eta, p)

cond_N0_S2_grids1, cond_N1_S2_grids1 = shared_dpm_conditional_probability_fixed(sigma2_e, power2, grids_1, theta, phi, w, lambda, eta, p)
cond_N0_S2_grids2, cond_N1_S2_grids2 = shared_dpm_conditional_probability_fixed(sigma2_e, power2, grids_2, theta, phi, w, lambda, eta, p)
cond_N0_S2_grids3, cond_N1_S2_grids3 = shared_dpm_conditional_probability_fixed(sigma2_e, power2, grids_3, theta, phi, w, lambda, eta, p)

cond_N0_S3_grids1, cond_N1_S3_grids1 = shared_dpm_conditional_probability_fixed(sigma2_e, power3, grids_1, theta, phi, w, lambda, eta, p)
cond_N0_S3_grids2, cond_N1_S3_grids2 = shared_dpm_conditional_probability_fixed(sigma2_e, power3, grids_2, theta, phi, w, lambda, eta, p)
cond_N0_S3_grids3, cond_N1_S3_grids3 = shared_dpm_conditional_probability_fixed(sigma2_e, power3, grids_3, theta, phi, w, lambda, eta, p)

cond_N0_S4_grids1, cond_N1_S4_grids1 = shared_dpm_conditional_probability_fixed(sigma2_e, power4, grids_1, theta, phi, w, lambda, eta, p)
cond_N0_S4_grids2, cond_N1_S4_grids2 = shared_dpm_conditional_probability_fixed(sigma2_e, power4, grids_2, theta, phi, w, lambda, eta, p)
cond_N0_S4_grids3, cond_N1_S4_grids3 = shared_dpm_conditional_probability_fixed(sigma2_e, power4, grids_3, theta, phi, w, lambda, eta, p)

cond_N0_S5_grids1, cond_N1_S5_grids1 = shared_dpm_conditional_probability_fixed(sigma2_e, power5, grids_1, theta, phi, w, lambda, eta, p)
cond_N0_S5_grids2, cond_N1_S5_grids2 = shared_dpm_conditional_probability_fixed(sigma2_e, power5, grids_2, theta, phi, w, lambda, eta, p)
cond_N0_S5_grids3, cond_N1_S5_grids3 = shared_dpm_conditional_probability_fixed(sigma2_e, power5, grids_3, theta, phi, w, lambda, eta, p)


@rput grids_1 grids_2 grids_3; 
@rput cond_N0_S1_grids1 cond_N0_S1_grids2 cond_N0_S1_grids3;
@rput cond_N0_S2_grids1 cond_N0_S2_grids2 cond_N0_S2_grids3;
@rput cond_N0_S3_grids1 cond_N0_S3_grids2 cond_N0_S3_grids3;
@rput cond_N0_S4_grids1 cond_N0_S4_grids2 cond_N0_S4_grids3;
@rput cond_N0_S5_grids1 cond_N0_S5_grids2 cond_N0_S5_grids3;
@rput cond_N1_S1_grids1 cond_N1_S1_grids2 cond_N1_S1_grids3;
@rput cond_N1_S2_grids1 cond_N1_S2_grids2 cond_N1_S2_grids3;
@rput cond_N1_S3_grids1 cond_N1_S3_grids2 cond_N1_S3_grids3;
@rput cond_N1_S4_grids1 cond_N1_S4_grids2 cond_N1_S4_grids3;
@rput cond_N1_S5_grids1 cond_N1_S5_grids2 cond_N1_S5_grids3;
R"""
fig_path = "//Users/yunzheli/Research/BNPJoint/figs/prior/analysis4/"

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
lines(cond_N0_S4_grids1 ~ grids_1, lwd=2, col="purple")
lines(cond_N0_S5_grids1 ~ grids_1, lwd=2, col="brown")
legend("topright", lwd=2, col=c("red","blue","green","purple","brown"), legend=c("Positive", "Independent", "Negative", "Positive Large", "Negative Large"))
abline(v=1)
dev.off() 

png(paste0(fig_path, "N0_grids2.png"), width=480, height=480)
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
lines(cond_N0_S4_grids2 ~ grids_2, lwd=2, col="purple")
lines(cond_N0_S5_grids2 ~ grids_2, lwd=2, col="brown")
legend("topright", lwd=2, col=c("red","blue","green","purple","brown"), legend=c("Positive", "Independent", "Negative", "Positive Large", "Negative Large"))
abline(v=2)
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
lines(cond_N0_S4_grids3 ~ grids_3, lwd=2, col="purple")
lines(cond_N0_S5_grids3 ~ grids_3, lwd=2, col="brown")
legend("topright", lwd=2, col=c("red","blue","green","purple","brown"), legend=c("Positive", "Independent", "Negative", "Positive Large", "Negative Large"))
abline(v=5)
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
lines(cond_N1_S4_grids1 ~ grids_1, lwd=2, col="purple")
lines(cond_N1_S5_grids1 ~ grids_1, lwd=2, col="brown")
legend("topright", lwd=2, col=c("red","blue","green","purple","brown"), legend=c("Positive", "Independent", "Negative", "Positive Large", "Negative Large"))
abline(v=1)
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
lines(cond_N1_S4_grids2 ~ grids_2, lwd=2, col="purple")
lines(cond_N1_S5_grids2 ~ grids_2, lwd=2, col="brown")
legend("topright", lwd=2, col=c("red","blue","green","purple","brown"), legend=c("Positive", "Independent", "Negative", "Positive Large", "Negative Large"))
abline(v=2)
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
lines(cond_N1_S4_grids3 ~ grids_3, lwd=2, col="purple")
lines(cond_N1_S5_grids3 ~ grids_3, lwd=2, col="brown")
legend("topright", lwd=2, col=c("red","blue","green","purple","brown"), legend=c("Positive", "Independent", "Negative", "Positive Large", "Negative Large"))
abline(v=5)
dev.off()
"""

# N = length(w)
grids = range(0.001, 20, length=200)
# ngrids = length(grids)
surv_dens1, surv_surv1, gap_dens1, gap_surv1 = shared_dpm_density_fixed(sigma2_e, power1, grids, theta, phi, w, lambda, eta, p) 
surv_dens2, surv_surv2, gap_dens2, gap_surv2 = shared_dpm_density_fixed(sigma2_e, power2, grids, theta, phi, w, lambda, eta, p) 
surv_dens3, surv_surv3, gap_dens3, gap_surv3 = shared_dpm_density_fixed(sigma2_e, power3, grids, theta, phi, w, lambda, eta, p) 
surv_dens4, surv_surv4, gap_dens4, gap_surv4 = shared_dpm_density_fixed(sigma2_e, power4, grids, theta, phi, w, lambda, eta, p) 
surv_dens5, surv_surv5, gap_dens5, gap_surv5 = shared_dpm_density_fixed(sigma2_e, power5, grids, theta, phi, w, lambda, eta, p) 


@rput grids; 
@rput surv_dens1 gap_dens1; 
@rput surv_dens2 gap_dens2; 
@rput surv_dens3 gap_dens3; 
@rput surv_dens4 gap_dens4; 
@rput surv_dens5 gap_dens5; 
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

png(paste0(fig_path, "surv_dens4.png"), width=480, height=480)
plot(surv_dens4 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=4)
dev.off()

png(paste0(fig_path, "gap_dens4.png"), width=480, height=480)
plot(gap_dens4 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off() 

png(paste0(fig_path, "surv_dens5.png"), width=480, height=480)
plot(surv_dens5 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off()

png(paste0(fig_path, "gap_dens5.png"), width=480, height=480)
plot(gap_dens5 ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off() 
"""



n = 400
survivals1, gaps1, arrivals1 = shared_dpm_data_generator_fixed(sigma2_e, power1, theta, phi, w, lambda, eta, p, n)
survivals2, gaps2, arrivals2 = shared_dpm_data_generator_fixed(sigma2_e, power2, theta, phi, w, lambda, eta, p, n)
survivals3, gaps3, arrivals3 = shared_dpm_data_generator_fixed(sigma2_e, power3, theta, phi, w, lambda, eta, p, n)
survivals4, gaps4, arrivals4 = shared_dpm_data_generator_fixed(sigma2_e, power4, theta, phi, w, lambda, eta, p, n)
survivals5, gaps5, arrivals5 = shared_dpm_data_generator_fixed(sigma2_e, power5, theta, phi, w, lambda, eta, p, n)

@rput survivals1 survivals2 survivals3 survivals4 survivals5 
@rput gaps1 gaps2 gaps3 gaps4 gaps5 
@rput arrivals1 arrivals2 arrivals3 arrivals4 arrivals5
R"""
source("//Users/yunzheli/Packages/BNPJointModel/src/visualizer.R")

p_survivals1 = visualize_survival_times(survivals1)
ggsave(paste0(fig_path, "survival_predictive1.png"), p_survivals1)

p_survivals2 = visualize_survival_times(survivals2)
ggsave(paste0(fig_path, "survival_predictive2.png"), p_survivals2)

p_survivals3 = visualize_survival_times(survivals3)
ggsave(paste0(fig_path, "survival_predictive3.png"), p_survivals3)

p_survivals4 = visualize_survival_times(survivals4)
ggsave(paste0(fig_path, "survival_predictive4.png"), p_survivals4)

p_survivals5 = visualize_survival_times(survivals5)
ggsave(paste0(fig_path, "survival_predictive5.png"), p_survivals5)

p_gaps1 = visualize_gap_times(gaps1)
ggsave(paste0(fig_path, "gap_predictive1.png"), p_gaps1)

p_gaps2 = visualize_gap_times(gaps2)
ggsave(paste0(fig_path, "gap_predictive2.png"), p_gaps2)

p_gaps3 = visualize_gap_times(gaps3)
ggsave(paste0(fig_path, "gap_predictive3.png"), p_gaps3)

p_gaps4 = visualize_gap_times(gaps4)
ggsave(paste0(fig_path, "gap_predictive4.png"), p_gaps4)

p_gaps5 = visualize_gap_times(gaps5)
ggsave(paste0(fig_path, "gap_predictive5.png"), p_gaps5)

p_recurrent1 = visualize_recurrent_events(arrivals1, survivals1)
ggsave(paste0(fig_path, "recurrent_predictive1.png"), p_recurrent1)

p_recurrent2 = visualize_recurrent_events(arrivals2, survivals2)
ggsave(paste0(fig_path, "recurrent_predictive2.png"), p_recurrent2)

p_recurrent3 = visualize_recurrent_events(arrivals3, survivals3)
ggsave(paste0(fig_path, "recurrent_predictive3.png"), p_recurrent3)

p_recurrent4 = visualize_recurrent_events(arrivals4, survivals4)
ggsave(paste0(fig_path, "recurrent_predictive4.png"), p_recurrent4)

p_recurrent5 = visualize_recurrent_events(arrivals5, survivals5)
ggsave(paste0(fig_path, "recurrent_predictive5.png"), p_recurrent5)
"""