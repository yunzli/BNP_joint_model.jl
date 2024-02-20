using TOML
using RCall 
using JLD2 
using Distributions 
using ProgressMeter
using LinearAlgebra
using StatsBase
using Random 

Random.seed!(200)

include("../src/loglogistic.jl")
include("prior_analysis_func.jl")

# 1. Single loglogistic kernel for survival and gap, 
#    bivariate lognormal random effects fixed Sigma 

theta = 5
phi = 20 

lambda = 2
eta = 20

Sigma_e_1 = 0.5 * [1 0.75; 0.75 1]
Sigma_e_2 = 0.5 * [1 0; 0 1]
Sigma_e_3 = 0.5 * [1 -0.75; -0.75 1]
grids_1 = range(1, 20, length=500)
grids_2 = range(2, 20, length=500)
grids_3 = range(5, 20, length=500)


cond_N0_S1_grids1, cond_N1_S1_grids1 = conditional_probability_fixed(Sigma_e_1, grids_1, theta, phi, lambda, eta)
cond_N0_S1_grids2, cond_N1_S1_grids2 = conditional_probability_fixed(Sigma_e_1, grids_2, theta, phi, lambda, eta)
cond_N0_S1_grids3, cond_N1_S1_grids3 = conditional_probability_fixed(Sigma_e_1, grids_3, theta, phi, lambda, eta)

cond_N0_S2_grids1, cond_N1_S2_grids1 = conditional_probability_fixed(Sigma_e_2, grids_1, theta, phi, lambda, eta)
cond_N0_S2_grids2, cond_N1_S2_grids2 = conditional_probability_fixed(Sigma_e_2, grids_2, theta, phi, lambda, eta)
cond_N0_S2_grids3, cond_N1_S2_grids3 = conditional_probability_fixed(Sigma_e_2, grids_3, theta, phi, lambda, eta)

cond_N0_S3_grids1, cond_N1_S3_grids1 = conditional_probability_fixed(Sigma_e_3, grids_1, theta, phi, lambda, eta)
cond_N0_S3_grids2, cond_N1_S3_grids2 = conditional_probability_fixed(Sigma_e_3, grids_2, theta, phi, lambda, eta)
cond_N0_S3_grids3, cond_N1_S3_grids3 = conditional_probability_fixed(Sigma_e_3, grids_3, theta, phi, lambda, eta)


@rput grids_1 grids_2 grids_3; 
@rput cond_N0_S1_grids1 cond_N0_S1_grids2 cond_N0_S1_grids3;
@rput cond_N0_S2_grids1 cond_N0_S2_grids2 cond_N0_S2_grids3;
@rput cond_N0_S3_grids1 cond_N0_S3_grids2 cond_N0_S3_grids3;
@rput cond_N1_S1_grids1 cond_N1_S1_grids2 cond_N1_S1_grids3;
@rput cond_N1_S2_grids1 cond_N1_S2_grids2 cond_N1_S2_grids3;
@rput cond_N1_S3_grids1 cond_N1_S3_grids2 cond_N1_S3_grids3;
R"""
fig_path = "//Users/yunzheli/Research/BNPJoint/figs/prior/analysis1/"

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



grids = range(0.001, 20, length=500)
surv_dens, surv_surv, gap_dens, gap_surv = density_fixed(Sigma_e_2, grids, theta, phi, lambda, eta) 
@rput grids surv_dens surv_surv gap_dens gap_surv
R"""
png(paste0(fig_path, "surv_dens.png"), width=480, height=480)
plot(surv_dens ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off()

png(paste0(fig_path, "gap_dens.png"), width=480, height=480)
plot(gap_dens ~  grids, type='l', xlab="t", ylab="Density")
abline(v=1)
abline(v=2)
abline(v=5)
dev.off() 
"""

n = 400
survivals1, gaps1, arrivals1 = data_generator_fixed(Sigma_e_1, theta, phi, lambda, eta, n)
survivals2, gaps2, arrivals2 = data_generator_fixed(Sigma_e_2, theta, phi, lambda, eta, n)
survivals3, gaps3, arrivals3 = data_generator_fixed(Sigma_e_3, theta, phi, lambda, eta, n)

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