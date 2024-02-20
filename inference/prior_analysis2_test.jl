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

# 2. Single loglogistic kernel for survival and gap 
#    bivariate lognormal random effects random Sigma 

theta = 5
phi = 20 

lambda = 2
eta = 20

c_e = 13 

C_e_1 = 0.5 * [1 0.75; 0.75 1]
C_e_2 = 0.5 * [1 0; 0 1]
C_e_3 = 0.5 * [1 -0.75; -0.75 1]

grids_1 = range(1, 20, length=100)
grids_2 = range(2, 20, length=100)
grids_3 = range(5, 20, length=100)

cond_N0_S1_grids1 = zeros(100)
cond_N1_S1_grids1 = zeros(100)
cond_N0_S1_grids2 = zeros(100)
cond_N1_S1_grids2 = zeros(100)
cond_N0_S1_grids3 = zeros(100)
cond_N1_S1_grids3 = zeros(100)
cond_N0_S2_grids1 = zeros(100)
cond_N1_S2_grids1 = zeros(100)
cond_N0_S2_grids2 = zeros(100)
cond_N1_S2_grids2 = zeros(100)
cond_N0_S2_grids3 = zeros(100)
cond_N1_S2_grids3 = zeros(100)
cond_N0_S3_grids1 = zeros(100)
cond_N1_S3_grids1 = zeros(100)
cond_N0_S3_grids2 = zeros(100)
cond_N1_S3_grids2 = zeros(100)
cond_N0_S3_grids3 = zeros(100)
cond_N1_S3_grids3 = zeros(100)
n_rep = 500
for i in 1:n_rep
    Sigma_e_1 = rand(InverseWishart(c_e, C_e_1), 1)[1]
    tmp_N0_S1_grids1, tmp_N1_S1_grids1 = conditional_probability_fixed(Sigma_e_1, grids_1, theta, phi, lambda, eta)
    cond_N0_S1_grids1 .+= 1/n_rep .* tmp_N0_S1_grids1
    cond_N1_S1_grids1 .+= 1/n_rep .* tmp_N1_S1_grids1
    tmp_N0_S1_grids2, tmp_N1_S1_grids2 = conditional_probability_fixed(Sigma_e_1, grids_2, theta, phi, lambda, eta)
    cond_N0_S1_grids2 .+= 1/n_rep .* tmp_N0_S1_grids2
    cond_N1_S1_grids2 .+= 1/n_rep .* tmp_N1_S1_grids2
    tmp_N0_S1_grids3, tmp_N1_S1_grids3 = conditional_probability_fixed(Sigma_e_1, grids_3, theta, phi, lambda, eta)
    cond_N0_S1_grids3 .+= 1/n_rep .* tmp_N0_S1_grids3
    cond_N1_S1_grids3 .+= 1/n_rep .* tmp_N1_S1_grids3

    Sigma_e_2 = rand(InverseWishart(c_e, C_e_2), 1)[1]
    tmp_N0_S2_grids1, tmp_N1_S2_grids1 = conditional_probability_fixed(Sigma_e_2, grids_1, theta, phi, lambda, eta)
    cond_N0_S2_grids1 .+= 1/n_rep .* tmp_N0_S2_grids1
    cond_N1_S2_grids1 .+= 1/n_rep .* tmp_N1_S2_grids1
    tmp_N0_S2_grids2, tmp_N1_S2_grids2 = conditional_probability_fixed(Sigma_e_2, grids_2, theta, phi, lambda, eta)
    cond_N0_S2_grids2 .+= 1/n_rep .* tmp_N0_S2_grids2
    cond_N1_S2_grids2 .+= 1/n_rep .* tmp_N1_S2_grids2
    tmp_N0_S2_grids3, tmp_N1_S2_grids3 = conditional_probability_fixed(Sigma_e_2, grids_3, theta, phi, lambda, eta)
    cond_N0_S2_grids3 .+= 1/n_rep .* tmp_N0_S2_grids3
    cond_N1_S2_grids3 .+= 1/n_rep .* tmp_N1_S2_grids3

    Sigma_e_3 = rand(InverseWishart(c_e, C_e_3), 1)[1]
    tmp_N0_S3_grids1, tmp_N1_S3_grids1 = conditional_probability_fixed(Sigma_e_3, grids_1, theta, phi, lambda, eta)
    cond_N0_S3_grids1 .+= 1/n_rep .* tmp_N0_S3_grids1
    cond_N1_S3_grids1 .+= 1/n_rep .* tmp_N1_S3_grids1
    tmp_N0_S3_grids2, tmp_N1_S3_grids2 = conditional_probability_fixed(Sigma_e_3, grids_2, theta, phi, lambda, eta)
    cond_N0_S3_grids2 .+= 1/n_rep .* tmp_N0_S3_grids2
    cond_N1_S3_grids2 .+= 1/n_rep .* tmp_N1_S3_grids2
    tmp_N0_S3_grids3, tmp_N1_S3_grids3 = conditional_probability_fixed(Sigma_e_3, grids_3, theta, phi, lambda, eta)
    cond_N0_S3_grids3 .+= 1/n_rep .* tmp_N0_S3_grids3
    cond_N1_S3_grids3 .+= 1/n_rep .* tmp_N1_S3_grids3
end


@rput grids_1 grids_2 grids_3; 
@rput cond_N0_S1_grids1 cond_N0_S1_grids2 cond_N0_S1_grids3;
@rput cond_N0_S2_grids1 cond_N0_S2_grids2 cond_N0_S2_grids3;
@rput cond_N0_S3_grids1 cond_N0_S3_grids2 cond_N0_S3_grids3;
@rput cond_N1_S1_grids1 cond_N1_S1_grids2 cond_N1_S1_grids3;
@rput cond_N1_S2_grids1 cond_N1_S2_grids2 cond_N1_S2_grids3;
@rput cond_N1_S3_grids1 cond_N1_S3_grids2 cond_N1_S3_grids3;
R"""
fig_path = "//Users/yunzheli/Research/BNPJoint/figs/prior/analysis2/"

png(paste0(fig_path, "N0_grids1.png"))
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
dev.off() 

png(paste0(fig_path, "N0_grids2.png"))
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
dev.off() 

png(paste0(fig_path, "N0_grids3.png"))
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
dev.off()

png(paste0(fig_path, "N1_grids1.png"))
plot(cond_N1_S1_grids1 ~ grids_1, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N1_S2_grids1 ~ grids_1, lwd=2, lty=2, col="blue")
lines(cond_N1_S3_grids1 ~ grids_1, lwd=2, lty=2, col="green")
abline(v=1)
dev.off() 

png(paste0(fig_path, "N1_grids2.png"))
plot(cond_N1_S1_grids2 ~ grids_2, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N1_S2_grids2 ~ grids_2, lwd=2, lty=2, col="blue")
lines(cond_N1_S3_grids2 ~ grids_2, lwd=2, lty=2, col="green")
abline(v=2)
dev.off() 

png(paste0(fig_path, "N1_grids3.png"))
plot(cond_N1_S1_grids3 ~ grids_3, 
    lwd=2, 
    col="red", 
    type='l', 
    xlim=c(0,20), 
    ylim=c(0,1),
    xlab="t",
    ylab="Conditional probability"
)
lines(cond_N1_S2_grids3 ~ grids_3, lwd=2, lty=2, col="blue")
lines(cond_N1_S3_grids3 ~ grids_3, lwd=2, lty=2, col="green")
abline(v=5)
dev.off()
"""
