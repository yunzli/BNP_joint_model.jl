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

# 6. LDDP survival and gap, 
#    bivaraite lognormal random effects fixed Sigma  

alpha = 5
mu_theta = log(5)
sigma2_theta = 1
mu_beta = 0.2*ones(1)
Sigma_beta = Matrix(Diagonal(ones(1)))
a_phi = 20 
b_phi = 20


zeta = 5
mu_lambda = log(2)
sigma2_lambda = 1
mu_gamma = 0.2*ones(1)
Sigma_gamma = Matrix(Diagonal(ones(1)))
a_eta = 20
b_eta = 20


res1 = prior_dp_regression(alpha, mu_theta, sigma2_theta, mu_beta, Sigma_beta, a_phi, b_phi)
w = res1["weights"]
theta = exp.(res1["atoms"][1])
beta = res1["atoms"][2]
phi = sqrt.(res1["atoms"][3])

res2 = prior_dp_regression(alpha, mu_lambda, sigma2_lambda, mu_gamma, Sigma_gamma, a_eta, b_eta)
p = res2["weights"]
lambda = exp.(res2["atoms"][1])
gamma = res2["atoms"][2]
eta = sqrt.(res2["atoms"][3])


Sigma_e_1 = 0.5 * [1 0.75; 0.75 1]
Sigma_e_2 = 0.5 * [1 0; 0 1]
Sigma_e_3 = 0.5 * [1 -0.75; -0.75 1]
grids_1 = range(1, 20, length=100)
grids_2 = range(2, 20, length=100)
grids_3 = range(5, 20, length=100)

cond_N0_S1_grids1C, cond_N1_S1_grids1C = dpm_conditional_probability_fixed_regression(Sigma_e_1, grids_1, [0], [0], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S1_grids2C, cond_N1_S1_grids2C = dpm_conditional_probability_fixed_regression(Sigma_e_1, grids_2, [0], [0], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S1_grids3C, cond_N1_S1_grids3C = dpm_conditional_probability_fixed_regression(Sigma_e_1, grids_3, [0], [0], theta, beta, phi, w, lambda, gamma, eta, p)

cond_N0_S2_grids1C, cond_N1_S2_grids1C = dpm_conditional_probability_fixed_regression(Sigma_e_2, grids_1, [0], [0], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S2_grids2C, cond_N1_S2_grids2C = dpm_conditional_probability_fixed_regression(Sigma_e_2, grids_2, [0], [0], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S2_grids3C, cond_N1_S2_grids3C = dpm_conditional_probability_fixed_regression(Sigma_e_2, grids_3, [0], [0], theta, beta, phi, w, lambda, gamma, eta, p)

cond_N0_S3_grids1C, cond_N1_S3_grids1C = dpm_conditional_probability_fixed_regression(Sigma_e_3, grids_1, [0], [0], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S3_grids2C, cond_N1_S3_grids2C = dpm_conditional_probability_fixed_regression(Sigma_e_3, grids_2, [0], [0], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S3_grids3C, cond_N1_S3_grids3C = dpm_conditional_probability_fixed_regression(Sigma_e_3, grids_3, [0], [0], theta, beta, phi, w, lambda, gamma, eta, p)

cond_N0_S1_grids1T, cond_N1_S1_grids1T = dpm_conditional_probability_fixed_regression(Sigma_e_1, grids_1, [1], [1], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S1_grids2T, cond_N1_S1_grids2T = dpm_conditional_probability_fixed_regression(Sigma_e_1, grids_2, [1], [1], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S1_grids3T, cond_N1_S1_grids3T = dpm_conditional_probability_fixed_regression(Sigma_e_1, grids_3, [1], [1], theta, beta, phi, w, lambda, gamma, eta, p)

cond_N0_S2_grids1T, cond_N1_S2_grids1T = dpm_conditional_probability_fixed_regression(Sigma_e_2, grids_1, [1], [1], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S2_grids2T, cond_N1_S2_grids2T = dpm_conditional_probability_fixed_regression(Sigma_e_2, grids_2, [1], [1], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S2_grids3T, cond_N1_S2_grids3T = dpm_conditional_probability_fixed_regression(Sigma_e_2, grids_3, [1], [1], theta, beta, phi, w, lambda, gamma, eta, p)

cond_N0_S3_grids1T, cond_N1_S3_grids1T = dpm_conditional_probability_fixed_regression(Sigma_e_3, grids_1, [1], [1], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S3_grids2T, cond_N1_S3_grids2T = dpm_conditional_probability_fixed_regression(Sigma_e_3, grids_2, [1], [1], theta, beta, phi, w, lambda, gamma, eta, p)
cond_N0_S3_grids3T, cond_N1_S3_grids3T = dpm_conditional_probability_fixed_regression(Sigma_e_3, grids_3, [1], [1], theta, beta, phi, w, lambda, gamma, eta, p)


@rput grids_1 grids_2 grids_3; 
@rput cond_N0_S1_grids1C cond_N0_S1_grids2C cond_N0_S1_grids3C;
@rput cond_N0_S2_grids1C cond_N0_S2_grids2C cond_N0_S2_grids3C;
@rput cond_N0_S3_grids1C cond_N0_S3_grids2C cond_N0_S3_grids3C;
@rput cond_N1_S1_grids1C cond_N1_S1_grids2C cond_N1_S1_grids3C;
@rput cond_N1_S2_grids1C cond_N1_S2_grids2C cond_N1_S2_grids3C;
@rput cond_N1_S3_grids1C cond_N1_S3_grids2C cond_N1_S3_grids3C;
@rput cond_N0_S1_grids1T cond_N0_S1_grids2T cond_N0_S1_grids3T;
@rput cond_N0_S2_grids1T cond_N0_S2_grids2T cond_N0_S2_grids3T;
@rput cond_N0_S3_grids1T cond_N0_S3_grids2T cond_N0_S3_grids3T;
@rput cond_N1_S1_grids1T cond_N1_S1_grids2T cond_N1_S1_grids3T;
@rput cond_N1_S2_grids1T cond_N1_S2_grids2T cond_N1_S2_grids3T;
@rput cond_N1_S3_grids1T cond_N1_S3_grids2T cond_N1_S3_grids3T;
R"""
fig_path = "//Users/yunzheli/Research/BNPJoint/figs/prior/analysis_regression1/"

png(paste0(fig_path, "N0_grids1.png"), width=480, height=480)
plot(cond_N0_S1_grids1C ~ grids_1, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N0_S2_grids1C ~ grids_1, lwd=2, col="blue")
lines(cond_N0_S3_grids1C ~ grids_1, lwd=2, col="green")
lines(cond_N0_S1_grids1T ~ grids_1, lwd=2, lty=2, col="red")
lines(cond_N0_S2_grids1T ~ grids_1, lwd=2, lty=2, col="blue")
lines(cond_N0_S3_grids1T ~ grids_1, lwd=2, lty=2, col="green")
abline(v=1)
legend("topright", lwd=2, col=c("red","blue","green","red","blue","green"), lty=c(1,1,1,2,2,2), legend=c("Positive Control", "Independent Control", "Negative Control","Positive Treatment","Independent Treatment","Negative Treatment"))
dev.off() 

png(paste0(fig_path, "N0_grids2.png"), width=480,  height=480)
plot(cond_N0_S1_grids2C ~ grids_2, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N0_S2_grids2C ~ grids_2, lwd=2, col="blue")
lines(cond_N0_S3_grids2C ~ grids_2, lwd=2, col="green")
lines(cond_N0_S1_grids2T ~ grids_2, lwd=2, lty=2, col="red")
lines(cond_N0_S2_grids2T ~ grids_2, lwd=2, lty=2, col="blue")
lines(cond_N0_S3_grids2T ~ grids_2, lwd=2, lty=2, col="green")
abline(v=2)
legend("topright", lwd=2, col=c("red","blue","green","red","blue","green"), lty=c(1,1,1,2,2,2), legend=c("Positive Control", "Independent Control", "Negative Control","Positive Treatment","Independent Treatment","Negative Treatment"))
dev.off() 

png(paste0(fig_path, "N0_grids3.png"), width=480, height=480)
plot(cond_N0_S1_grids3C ~ grids_3, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N0_S2_grids3C ~ grids_3, lwd=2, col="blue")
lines(cond_N0_S3_grids3C ~ grids_3, lwd=2, col="green")
lines(cond_N0_S1_grids3T ~ grids_3, lwd=2, lty=2, col="red")
lines(cond_N0_S2_grids3T ~ grids_3, lwd=2, lty=2, col="blue")
lines(cond_N0_S3_grids3T ~ grids_3, lwd=2, lty=2, col="green")
abline(v=5)
legend("topright", lwd=2, col=c("red","blue","green","red","blue","green"), lty=c(1,1,1,2,2,2), legend=c("Positive Control", "Independent Control", "Negative Control","Positive Treatment","Independent Treatment","Negative Treatment"))
dev.off() 

png(paste0(fig_path, "N1_grids1.png"), width=480, height=480)
plot(cond_N1_S1_grids1C ~ grids_1, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N1_S2_grids1C ~ grids_1, lwd=2, col="blue")
lines(cond_N1_S3_grids1C ~ grids_1, lwd=2, col="green")
lines(cond_N1_S1_grids1T ~ grids_1, lwd=2, lty=2, col="red")
lines(cond_N1_S2_grids1T ~ grids_1, lwd=2, lty=2, col="blue")
lines(cond_N1_S3_grids1T ~ grids_1, lwd=2, lty=2, col="green")
abline(v=1)
legend("topright", lwd=2, col=c("red","blue","green","red","blue","green"), lty=c(1,1,1,2,2,2), legend=c("Positive Control", "Independent Control", "Negative Control","Positive Treatment","Independent Treatment","Negative Treatment"))
dev.off() 

png(paste0(fig_path, "N1_grids2.png"), width=480, height=480)
plot(cond_N1_S1_grids2C ~ grids_2, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N1_S2_grids2C ~ grids_2, lwd=2, col="blue")
lines(cond_N1_S3_grids2C ~ grids_2, lwd=2, col="green")
lines(cond_N1_S1_grids2T ~ grids_2, lwd=2, lty=2, col="red")
lines(cond_N1_S2_grids2T ~ grids_2, lwd=2, lty=2, col="blue")
lines(cond_N1_S3_grids2T ~ grids_2, lwd=2, lty=2, col="green")
abline(v=2)
legend("topright", lwd=2, col=c("red","blue","green","red","blue","green"), lty=c(1,1,1,2,2,2), legend=c("Positive Control", "Independent Control", "Negative Control","Positive Treatment","Independent Treatment","Negative Treatment"))
dev.off() 

png(paste0(fig_path, "N1_grids3.png"), width=480, height=480)
plot(cond_N1_S1_grids3C ~ grids_3, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N1_S2_grids3C ~ grids_3, lwd=2, col="blue")
lines(cond_N1_S3_grids3C ~ grids_3, lwd=2, col="green")
lines(cond_N1_S1_grids3T ~ grids_3, lwd=2, lty=2, col="red")
lines(cond_N1_S2_grids3T ~ grids_3, lwd=2, lty=2, col="blue")
lines(cond_N1_S3_grids3T ~ grids_3, lwd=2, lty=2, col="green")
abline(v=5)
legend("topright", lwd=2, col=c("red","blue","green","red","blue","green"), lty=c(1,1,1,2,2,2), legend=c("Positive Control", "Independent Control", "Negative Control","Positive Treatment","Independent Treatment","Negative Treatment"))
dev.off()
"""


# # N = length(w)
# grids = range(0.001, 20, length=200)
# surv_dens, surv_surv, gap_dens, gap_surv = dpm_density_fixed(Sigma_e_2, grids, theta, phi, w, lambda, eta, p) 
# surv_dens2, surv_surv2, gap_dens2, gap_surv2 = dpm_density_fixed(Sigma_e_5, grids, theta, phi, w, lambda, eta, p) 
# # ngrids = length(grids)

# # surv_dens = zeros(ngrids)
# # gap_dens = zeros(ngrids) 
# # for g in 1:ngrids 
# #     for l in 1:N 
# #         surv_dens[g] += w[l] * pdf(LogLogistic(theta[l], phi[l]), grids[g])
# #         gap_dens[g] += p[l] * pdf(LogLogistic(lambda[l], eta[l]), grids[g])
# #     end
# # end

# @rput grids;
# @rput surv_dens gap_dens; 
# @rput surv_dens2 gap_dens2; 
# R"""
# png(paste0(fig_path, "surv_dens.png"), width=480, height=480)
# plot(surv_dens ~  grids, type='l', xlab="t", ylab="Density")
# abline(v=1)
# abline(v=2)
# abline(v=5)
# dev.off()

# png(paste0(fig_path, "gap_dens.png"), width=480, height=480)
# plot(gap_dens ~  grids, type='l', xlab="t", ylab="Density")
# abline(v=1)
# abline(v=2)
# abline(v=5)
# dev.off() 

# png(paste0(fig_path, "surv_dens2.png"), width=480, height=480)
# plot(surv_dens2 ~  grids, type='l', xlab="t", ylab="Density")
# abline(v=1)
# abline(v=2)
# abline(v=5)
# dev.off() 

# png(paste0(fig_path, "gap_dens2.png"), width=480, height=480)
# plot(gap_dens2 ~  grids, type='l', xlab="t", ylab="Density")
# abline(v=1)
# abline(v=2)
# abline(v=5)
# dev.off() 
# """



# n = 400
# survivals1, gaps1, arrivals1 = dpm_data_generator_fixed(Sigma_e_1, theta, phi, w, lambda, eta, p, n)
# survivals2, gaps2, arrivals2 = dpm_data_generator_fixed(Sigma_e_2, theta, phi, w, lambda, eta, p, n)
# survivals3, gaps3, arrivals3 = dpm_data_generator_fixed(Sigma_e_3, theta, phi, w, lambda, eta, p, n)
# survivals4, gaps4, arrivals4 = dpm_data_generator_fixed(Sigma_e_4, theta, phi, w, lambda, eta, p, n)
# survivals5, gaps5, arrivals5 = dpm_data_generator_fixed(Sigma_e_5, theta, phi, w, lambda, eta, p, n)
# survivals6, gaps6, arrivals6 = dpm_data_generator_fixed(Sigma_e_6, theta, phi, w, lambda, eta, p, n)

# @rput survivals1 survivals2 survivals3 survivals4 survivals5 survivals6  
# @rput gaps1 gaps2 gaps3 gaps4 gaps5 gaps6 
# @rput arrivals1 arrivals2 arrivals3 arrivals4 arrivals5 arrivals6
# R"""
# source("//Users/yunzheli/Packages/BNPJointModel/src/visualizer.R")

# p_survivals1 = visualize_survival_times(survivals1)
# ggsave(paste0(fig_path, "survival_predictive1.png"), p_survivals1)

# p_survivals2 = visualize_survival_times(survivals2)
# ggsave(paste0(fig_path, "survival_predictive2.png"), p_survivals2)

# p_survivals3 = visualize_survival_times(survivals3)
# ggsave(paste0(fig_path, "survival_predictive3.png"), p_survivals3)

# p_survivals4 = visualize_survival_times(survivals4)
# ggsave(paste0(fig_path, "survival_predictive4.png"), p_survivals4)

# p_survivals5 = visualize_survival_times(survivals5)
# ggsave(paste0(fig_path, "survival_predictive5.png"), p_survivals5)

# p_survivals6 = visualize_survival_times(survivals6)
# ggsave(paste0(fig_path, "survival_predictive6.png"), p_survivals6)

# p_gaps1 = visualize_gap_times(gaps1)
# ggsave(paste0(fig_path, "gap_predictive1.png"), p_gaps1)

# p_gaps2 = visualize_gap_times(gaps2)
# ggsave(paste0(fig_path, "gap_predictive2.png"), p_gaps2)

# p_gaps3 = visualize_gap_times(gaps3)
# ggsave(paste0(fig_path, "gap_predictive3.png"), p_gaps3)

# p_gaps4 = visualize_gap_times(gaps4)
# ggsave(paste0(fig_path, "gap_predictive4.png"), p_gaps3)

# p_gaps5 = visualize_gap_times(gaps5)
# ggsave(paste0(fig_path, "gap_predictive5.png"), p_gaps3)

# p_gaps6 = visualize_gap_times(gaps6)
# ggsave(paste0(fig_path, "gap_predictive6.png"), p_gaps3)

# p_recurrent1 = visualize_recurrent_events(arrivals1, survivals1)
# ggsave(paste0(fig_path, "recurrent_predictive1.png"), p_recurrent1)

# p_recurrent2 = visualize_recurrent_events(arrivals2, survivals2)
# ggsave(paste0(fig_path, "recurrent_predictive2.png"), p_recurrent2)

# p_recurrent3 = visualize_recurrent_events(arrivals3, survivals3)
# ggsave(paste0(fig_path, "recurrent_predictive3.png"), p_recurrent3)

# p_recurrent4 = visualize_recurrent_events(arrivals4, survivals4)
# ggsave(paste0(fig_path, "recurrent_predictive4.png"), p_recurrent4)

# p_recurrent5 = visualize_recurrent_events(arrivals5, survivals5)
# ggsave(paste0(fig_path, "recurrent_predictive5.png"), p_recurrent5)

# p_recurrent6 = visualize_recurrent_events(arrivals6, survivals6)
# ggsave(paste0(fig_path, "recurrent_predictive6.png"), p_recurrent6
# ggsave(paste0(fig_path, "recurrent_predictive6.png"), p_recurrent6)
# """