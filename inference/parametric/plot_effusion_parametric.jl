using TOML
using RCall 
using JLD2 
using Distributions 
using ProgressMeter
using LinearAlgebra
using StatsBase
using Random 

Random.seed!(100)

include("//Users/yunzheli/Packages/BNPJointModel/src/loglogistic.jl")

configC = TOML.parsefile("//Users/yunzheli/Packages/BNPJointModel/configs/parametric/effusionC.TOML")
configT = TOML.parsefile("//Users/yunzheli/Packages/BNPJointModel/configs/parametric/effusionT.TOML")

fig_path = configC["fig_path"]
if !isdir(fig_path)
    print(fig_path,"\n")
	mkdir(fig_path)
end
@rput fig_path 

dataC = load(configC["data_file"])
gapC = dataC["gap"]
survivalC = dataC["survival"]
arrivalC = dataC["arrival"]
nuC = dataC["nu"]

fitdataC = load(configC["save_path"])
posC = fitdataC["pos"]
hyperC = fitdataC["hyper"]

nsam = length(posC["theta"])
nburn = div(nsam, 4)
nthin = div(nsam-nburn,2000)
keep_index = [nburn+1:nthin:nsam;]
nkeep = length(keep_index)

theta_saveC = posC["theta"][keep_index]
phi_saveC = posC["phi"][keep_index]
lambda_saveC = posC["lambda"][keep_index]
eta_saveC = posC["eta"][keep_index]

Sigma_e_saveC = posC["Sigma_e"][keep_index,:,:]

dataT = load(configT["data_file"])
gapT = dataT["gap"]
survivalT = dataT["survival"]
arrivalT = dataT["arrival"]
nuT = dataT["nu"]

fitdataT = load(configT["save_path"])
posT = fitdataT["pos"]
hyperT = fitdataT["hyper"]

theta_saveT = posT["theta"][keep_index]
phi_saveT = posT["phi"][keep_index]
lambda_saveT = posT["lambda"][keep_index]
eta_saveT = posT["eta"][keep_index]

Sigma_e_saveT = posT["Sigma_e"][keep_index,:,:]

@rput theta_saveC phi_saveC lambda_saveC eta_saveC
@rput theta_saveT phi_saveT lambda_saveT eta_saveT
R"""
png(paste0(fig_path,"thetaC_trace.png"))
plot(theta_saveC, type='l', cex.axis=2)
dev.off() 
png(paste0(fig_path,"phiC_trace.png"))
plot(phi_saveC, type='l', cex.axis=2)
dev.off() 
png(paste0(fig_path,"lambdaC_trace.png"))
plot(lambda_saveC, type='l', cex.axis=2)
dev.off() 
png(paste0(fig_path,"etaC_trace.png"))
plot(eta_saveC, type='l', cex.axis=2)
dev.off() 

png(paste0(fig_path,"thetaT_trace.png"))
plot(theta_saveT, type='l', cex.axis=2)
dev.off() 
png(paste0(fig_path,"phiT_trace.png"))
plot(phi_saveT, type='l', cex.axis=2)
dev.off() 
png(paste0(fig_path,"lambdaT_trace.png"))
plot(lambda_saveT, type='l', cex.axis=2)
dev.off() 
png(paste0(fig_path,"etaT_trace.png"))
plot(eta_saveT, type='l', cex.axis=2)
dev.off() 
"""

s_theta, S_theta = hyperC["s_theta"], hyperC["S_theta"]
a_phi, b_phi = hyperC["a_phi"], hyperC["b_phi"]
s_lambda, S_lambda = hyperC["s_lambda"], hyperC["S_lambda"]
a_eta, b_eta = hyperC["a_eta"], hyperC["b_eta"]
@rput s_theta S_theta
@rput a_phi b_phi
@rput s_lambda S_lambda
@rput a_eta b_eta
R"""
theta_gridsC = seq(min(theta_saveC), max(theta_saveC), length=100)
theta_densC = dnorm(theta_gridsC, s_theta, sqrt(s_theta))
png(paste0(fig_path,"thetaC_hist.png"))
hist(theta_saveC, cex.axis=2, freq=FALSE)
lines(theta_densC ~ theta_gridsC, col="red")
dev.off() 

phi_gridsC = seq(min(phi_saveC), max(phi_saveC), length=100)
phi_densC = dgamma(phi_gridsC, a_phi, scale=b_phi)
png(paste0(fig_path,"phiC_hist.png"))
hist(phi_saveC, cex.axis=2, freq=FALSE)
lines(phi_densC ~ phi_gridsC, col="red")
dev.off() 

lambda_gridsC = seq(min(lambda_saveC), max(lambda_saveC), length=100)
lambda_densC = dnorm(lambda_gridsC, s_lambda, sqrt(s_lambda))
png(paste0(fig_path,"lambdaC_hist.png"))
hist(lambda_saveC, cex.axis=2)
lines(lambda_densC ~ lambda_gridsC, col="red")
dev.off() 

eta_gridsC = seq(min(eta_saveC), max(eta_saveC), length=100)
eta_densC = dgamma(eta_gridsC, a_eta, scale=b_eta)
png(paste0(fig_path,"etaC_hist.png"))
hist(eta_saveC, cex.axis=2)
lines(eta_densC ~ eta_gridsC, col="red")
dev.off() 

theta_gridsT = seq(min(theta_saveT), max(theta_saveT), length=100)
theta_densT = dnorm(theta_gridsT, s_theta, sqrt(s_theta))
png(paste0(fig_path,"thetaT_hist.png"))
hist(theta_saveT, cex.axis=2, freq=FALSE)
lines(theta_densT ~ theta_gridsT, col="red")
dev.off() 

phi_gridsT = seq(min(phi_saveT), max(phi_saveT), length=100)
phi_densT = dgamma(phi_gridsT, a_phi, scale=b_phi)
png(paste0(fig_path,"phiT_hist.png"))
hist(phi_saveT, cex.axis=2, freq=FALSE)
lines(phi_densT ~ phi_gridsT, col="red")
dev.off() 

lambda_gridsT = seq(min(lambda_saveT), max(lambda_saveT), length=100)
lambda_densT = dnorm(lambda_gridsT, s_lambda, sqrt(s_lambda))
png(paste0(fig_path,"lambdaT_hist.png"))
hist(lambda_saveT, cex.axis=2)
lines(lambda_densT ~ lambda_gridsT, col="red")
dev.off() 

eta_gridsT = seq(min(eta_saveT), max(eta_saveT), length=100)
eta_densT = dgamma(eta_gridsT, a_eta, scale=b_eta)
png(paste0(fig_path,"etaT_hist.png"))
hist(eta_saveT, cex.axis=2)
lines(eta_densT ~ eta_gridsT, col="red")
dev.off() 
"""

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



epsilon_predC = zeros(nkeep)
xi_predC = zeros(nkeep)
for i in 1:nkeep
    re = rand(MvLogNormal(zeros(2), Sigma_e_saveC[i,:,:]), 1)
    epsilon_predC[i] = re[1]
    xi_predC[i] = re[2]
end

epsilon_predT = zeros(nkeep)
xi_predT = zeros(nkeep)
for i in 1:nkeep
    re = rand(MvLogNormal(zeros(2), Sigma_e_saveT[i,:,:]), 1)
    epsilon_predT[i] = re[1]
    xi_predT[i] = re[2]
end

c_eC = hyperC["c_e"]
C_eC = hyperC["C_e"]
c_eT = hyperT["c_e"]
C_eT = hyperT["C_e"]

epsilon_priorC = zeros(nkeep)
xi_priorC = zeros(nkeep)
for i in 1:nkeep
    Sigma_e_priorC = rand(InverseWishart(c_eC, C_eC), 1)[1]
    re = rand(MvLogNormal(zeros(2), Sigma_e_priorC), 1)
    epsilon_priorC[i] = re[1]
    xi_priorC[i] = re[2]
end 

epsilon_priorT = zeros(nkeep)
xi_priorT = zeros(nkeep)
for i in 1:nkeep
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

function survival_functional_parametric(theta, phi, sigma2_e, grids)

    nkeep = length(theta)
    ngrids = length(grids)
    density_pos = zeros(nkeep, ngrids)
    survival_pos = zeros(nkeep, ngrids)

    @showprogress for i in 1:nkeep
        nrep = 100
        
        epsilon_pos = rand(LogNormal(0, sqrt(sigma2_e[i])), nrep)

        for g in 1:ngrids
            for h in 1:nrep
                density_pos[i,g] += 1/nrep * pdf(LogLogistic(theta[i]/epsilon_pos[h], phi[i]), grids[g])
                survival_pos[i,g] += 1/nrep * ccdf(LogLogistic(theta[i]/epsilon_pos[h], phi[i]), grids[g])
            end
        end
    end

    return Dict("s" => survival_pos, "d" => density_pos)
end

surv_grids = range(0.001, 13, length=100)

survival_posC = survival_functional_parametric(
    theta_saveC, 
    phi_saveC, 
    Sigma_e_saveC[:,1,1],
    surv_grids
    )
    
survival_densC = survival_posC["d"]
survival_survC = survival_posC["s"]
survival_hazaC = survival_densC ./ survival_survC 

survival_posT = survival_functional_parametric(
    theta_saveT, 
    phi_saveT, 
    Sigma_e_saveT[:,1,1],
    surv_grids
    )
    
survival_densT = survival_posT["d"]
survival_survT = survival_posT["s"]
survival_hazaT = survival_densT ./ survival_survT 


gap_grids = range(0.001, 10, length=100)
gap_res_posC = survival_functional_parametric(
    lambda_saveC, 
    eta_saveC, 
    Sigma_e_saveC[:,2,2],
    gap_grids
    )
    
gap_densC = gap_res_posC["d"]
gap_survC = gap_res_posC["s"]
gap_hazaC = gap_densC ./ gap_survC

gap_res_posT = survival_functional_parametric(
    lambda_saveT, 
    eta_saveT, 
    Sigma_e_saveT[:,2,2],
    gap_grids
    )
    
gap_densT = gap_res_posT["d"]
gap_survT = gap_res_posT["s"]
gap_hazaT = gap_densT ./ gap_survT


function conditional_survival_probability_parametric(theta, phi, lambda, eta, Sigma_e, grids)

    nrep = 100 
    nkeep = length(theta)
    ngrids = length(grids)

    surv_surv = zeros(nkeep, ngrids, nrep)
    gap_surv0 = zeros(nkeep, nrep)

    @showprogress for i in 1:nkeep
        re = rand(MvLogNormal(zeros(2), Sigma_e[i,:,:]), nrep)
        epsilon = re[1,:]
        xi = re[2,:]

        for h in 1:nrep
            gap_surv0[i,h] = ccdf(LogLogistic(lambda[i]/xi[h], eta[i]), grids[1])
        end

        for g in 1:ngrids
            for h in 1:nrep
                surv_surv[i,g,h] = ccdf(LogLogistic(theta[i]/epsilon[h], phi[i]), grids[g])
            end
        end
    end

    cond = zeros(nkeep, ngrids)
    @showprogress for i in 1:nkeep
        upper = zeros(ngrids)
        lower = 0.0
        for h in 1:nrep
            lower += 1/nrep * surv_surv[i,1,h] * gap_surv0[i,h]
            for g in 2:ngrids
                upper[g] += 1/nrep * surv_surv[i,g,h] * gap_surv0[i,h]
            end
        end
        upper[1] = lower
        for g in 1:ngrids
            cond[i,g] = upper[g] / lower
        end
    end

    return cond
end

grids1 = range(0.5, 13, length=100)
grids2 = range(1,   13, length=100)
grids3 = range(2,   13, length=100)
grids4 = range(5,   13, length=100)

cond_probC_1 = conditional_survival_probability_parametric(
    theta_saveC, 
    phi_saveC, 
    eta_saveC, 
    lambda_saveC, 
    Sigma_e_saveC, 
    grids1 
)

cond_probC_2 = conditional_survival_probability_parametric(
    theta_saveC, 
    phi_saveC, 
    eta_saveC, 
    lambda_saveC, 
    Sigma_e_saveC, 
    grids2
)

cond_probC_3 = conditional_survival_probability_parametric(
    theta_saveC, 
    phi_saveC, 
    eta_saveC, 
    lambda_saveC, 
    Sigma_e_saveC, 
    grids3
)

cond_probC_4 = conditional_survival_probability_parametric(
    theta_saveC, 
    phi_saveC, 
    eta_saveC, 
    lambda_saveC, 
    Sigma_e_saveC, 
    grids4
)


cond_probT_1 = conditional_survival_probability_parametric(
    theta_saveT, 
    phi_saveT, 
    eta_saveT, 
    lambda_saveT, 
    Sigma_e_saveT, 
    grids1 
)

cond_probT_2 = conditional_survival_probability_parametric(
    theta_saveT, 
    phi_saveT, 
    eta_saveT, 
    lambda_saveT, 
    Sigma_e_saveT, 
    grids2
)

cond_probT_3 = conditional_survival_probability_parametric(
    theta_saveT, 
    phi_saveT, 
    eta_saveT, 
    lambda_saveT, 
    Sigma_e_saveT, 
    grids3
)

cond_probT_4 = conditional_survival_probability_parametric(
    theta_saveT, 
    phi_saveT, 
    eta_saveT, 
    lambda_saveT, 
    Sigma_e_saveT, 
    grids4
)



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
gap_p_surv = gap_p_surv + geom_line(data=gap_df_survT, aes(x=x,y=m), color="blue", linetype="dashed", size=1.5)
gap_p_surv = gap_p_surv + geom_ribbon(data=gap_df_survT, aes(x=x,ymin=l,ymax=h), fill="blue", alpha=0.5) 
gap_p_surv = gap_p_surv + ylab("Survival") + xlab("t") + ylim(0, 1.01)
ggsave(paste0(fig_path, "gap_survival.png"), gap_p_surv)
"""




@rput grids1 cond_probC_1 cond_probT_1
@rput grids2 cond_probC_2 cond_probT_2
@rput grids3 cond_probC_3 cond_probT_3
@rput grids4 cond_probC_4 cond_probT_4
R"""
cond_meanC_1 = apply(cond_probC_1, 2, mean, na.rm=TRUE) 
cond_quanC_1 = apply(cond_probC_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond_meanT_1 = apply(cond_probT_1, 2, mean, na.rm=TRUE) 
cond_quanT_1 = apply(cond_probT_1, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

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
p_cond_1 = p_cond_1 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1.01)
ggsave(paste0(fig_path, "conditional_probability_1.png"), p_cond_1)
"""

R"""
cond_meanC_2 = apply(cond_probC_2, 2, mean, na.rm=TRUE) 
cond_quanC_2 = apply(cond_probC_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond_meanT_2 = apply(cond_probT_2, 2, mean, na.rm=TRUE) 
cond_quanT_2 = apply(cond_probT_2, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

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
p_cond_2 = p_cond_2 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1.01)
ggsave(paste0(fig_path, "conditional_probability_2.png"), p_cond_2)
"""

R"""
cond_meanC_3 = apply(cond_probC_3, 2, mean, na.rm=TRUE) 
cond_quanC_3 = apply(cond_probC_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond_meanT_3 = apply(cond_probT_3, 2, mean, na.rm=TRUE) 
cond_quanT_3 = apply(cond_probT_3, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

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
p_cond_3 = p_cond_3 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1.01)
ggsave(paste0(fig_path, "conditional_probability_3.png"), p_cond_3)
"""

R"""
cond_meanC_4 = apply(cond_probC_4, 2, mean, na.rm=TRUE) 
cond_quanC_4 = apply(cond_probC_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)
cond_meanT_4 = apply(cond_probT_4, 2, mean, na.rm=TRUE) 
cond_quanT_4 = apply(cond_probT_4, 2, quantile, prob=c(0.025, 0.975), na.rm=TRUE)

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
p_cond_4 = p_cond_4 + xlab("t") + ylab("Conditional probability") + xlim(0,13) + ylim(0,1.01)
ggsave(paste0(fig_path, "conditional_probability_4.png"), p_cond_4)
"""


# @rput survival_predC survival_predT 
# @rput gap_predC gap_predT  
# @rput arrival_predC arrival_predT
# R"""
# source("//Users/yunzheli/Packages/BNPJointModel/src/visualizer.R")

# p_survival_predC = visualize_survival_times(survival_predC)
# ggsave(paste0(fig_path, "survival_predictiveC.png"), p_survival_predC)

# p_survival_predT = visualize_survival_times(survival_predT)
# ggsave(paste0(fig_path, "survival_predictiveT.png"), p_survival_predT)

# p_gap_predC = visualize_gap_times(gap_predC)
# ggsave(paste0(fig_path, "gap_predictiveC.png"), p_gap_predC)

# p_gap_predT = visualize_gap_times(gap_predT)
# ggsave(paste0(fig_path, "gap_predictiveT.png"), p_gap_predT)

# p_recurrentC = visualize_recurrent_events(arrival_predC, survival_predC)
# ggsave(paste0(fig_path, "recurrent_predictiveC.png"), p_recurrentC)

# p_recurrentT = visualize_recurrent_events(arrival_predT, survival_predT)
# ggsave(paste0(fig_path, "recurrent_predictiveT.png"), p_recurrentT)
# """

# Nvec_predC = length.(gap_predC)
# Nvec_predT = length.(gap_predT)
# @rput Nvec_predC Nvec_predT
# R"""
# png(paste0(fig_path, "number_of_reccurent_predC.png"))
# hist(Nvec_predC, main="", xlab="Ni", cex.axis=2)
# dev.off()

# png(paste0(fig_path, "number_of_reccurent_predC_cut.png"))
# hist(Nvec_predC[which(Nvec_predC<=10)], main="", xlab="Ni", breaks=seq(0,10,by=1), cex.axis=2)
# dev.off()

# png(paste0(fig_path, "number_of_reccurent_predT.png"))
# hist(Nvec_predT, main="", xlab="Ni", cex.axis=2)
# dev.off()

# png(paste0(fig_path, "number_of_reccurent_predT_cut.png"))
# hist(Nvec_predT[which(Nvec_predT<=10)], main="", xlab="Ni", breaks=seq(0,10,by=1), cex.axis=2)
# dev.off()
# """


# @rput survivalC survivalT 
# @rput gapC gapT  
# @rput arrivalC arrivalT
# @rput nuC nuT 
# R"""
# p_survivalC = visualize_survival_times(survivalC)
# ggsave(paste0(fig_path, "survival_originalC.png"), p_survivalC)

# p_survivalT = visualize_survival_times(survivalT)
# ggsave(paste0(fig_path, "survival_originalT.png"), p_survivalT)

# p_gapC = visualize_gap_times(gapC)
# ggsave(paste0(fig_path, "gap_originalC.png"), p_gapC)

# p_gapT = visualize_gap_times(gapT)
# ggsave(paste0(fig_path, "gap_originalT.png"), p_gapT)

# p_recurrentC = visualize_recurrent_events(arrivalC, survivalC)
# ggsave(paste0(fig_path, "recurrent_originalC.png"), p_recurrentC)

# p_recurrentT = visualize_recurrent_events(arrivalT, survivalT)
# ggsave(paste0(fig_path, "recurrent_originalT.png"), p_recurrentT)

# survC_KM = survival::survfit(survival::Surv(survivalC, nuC)~1)
# survT_KM = survival::survfit(survival::Surv(survivalT, nuT)~1)
# png(paste0(fig_path, "KM_surv.png"), height=480, width=480)
# plot(survC_KM, lwd=2, conf.int=FALSE, col="red", main="")
# lines(survT_KM, lwd=2, conf.int=FALSE, col="blue")
# dev.off()
# """

# NvecC = dataC["Nvec"]
# NvecT = dataT["Nvec"]
# @rput NvecC NvecT
# R"""
# png(paste0(fig_path, "number_of_reccurentC.png"))
# hist(NvecC, breaks=seq(0,10,by=1), main="", xlab="", cex.axis=2)
# dev.off()

# png(paste0(fig_path, "number_of_reccurentT.png"))
# hist(NvecT, breaks=seq(0,10,by=1), main="", xlab="", cex.axis=2)
# dev.off()
# """


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



# R"""
# KM_predC = survival::survfit(survival::Surv(unlist(survival_predC))~1)
# KM_predT = survival::survfit(survival::Surv(unlist(survival_predT))~1)
# png(paste0(fig_path, "KM_pred.png"), height=480, width=480)
# plot(survC_KM, lwd=2, conf.int=FALSE, col="red", main="", xlab="t")
# lines(survT_KM, lwd=2, conf.int=FALSE, col="blue")
# lines(KM_predC, lwd=2, lty=2, conf.int=FALSE, col="red")
# lines(KM_predT, lwd=2, lty=2, conf.int=FALSE, col="blue")
# legend("topright", col=c("red", "blue", "red", "blue"), lty=c(1,1,2,2), lwd=2, legend=c("3DRT, data", "IMRT, data", "3DRT, predicted data", "IMRT, predicted data"))
# dev.off()
# """


# @rput theta_predC phi_predC lambda_predC eta_predC
# @rput theta_predT phi_predT lambda_predT eta_predT
# R"""
# png(paste0(fig_path, "theta_predC.png"), height=480, width=480)
# hist(unlist(theta_predC), main="", xlab="", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "phi_predC.png"), height=480, width=480)
# hist(unlist(phi_predC), main="", xlab="", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "lambda_predC.png"), height=480, width=480)
# hist(unlist(lambda_predC), main="", xlab="", cex.axis=2)
# dev.off() 


# png(paste0(fig_path, "eta_predC.png"), height=480, width=480)
# hist(unlist(eta_predC), main="", xlab="", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "theta_predT.png"), height=480, width=480)
# hist(unlist(theta_predT), main="", xlab="", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "phi_predT.png"), height=480, width=480)
# hist(unlist(phi_predT), main="", xlab="", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "lambda_predT.png"), height=480, width=480)
# hist(unlist(lambda_predT), main="", xlab="", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "eta_predT.png"), height=480, width=480)
# hist(unlist(eta_predT), main="", xlab="", cex.axis=2)
# dev.off() 
# """

# # R"""
# # gapC_KM = survival::survfit(survival::Surv(unlist(gapC))~)
# # gap_KM_predC = survival::survfit(survival::Surv(unlist(gap_predC))~1)
# # gap_KM_predT = survival::survfit(survival::Surv(unlist(gap_predT))~1)
# # png(paste0(fig_path, "gap_KM_pred.png"), height=480, width=480)
# # plot(survC_KM, lwd=2, conf.int=FALSE, col="red", main="", xlab="t")
# # lines(survT_KM, lwd=2, conf.int=FALSE, col="blue")
# # lines(KM_predC, lwd=2, lty=2, conf.int=FALSE, col="red")
# # lines(KM_predT, lwd=2, lty=2, conf.int=FALSE, col="blue")
# # legend("topright", col=c("red", "blue", "red", "blue"), lty=c(1,1,2,2), lwd=2, legend=c("3DRT, data", "IMRT, data", "3DRT, predicted data", "IMRT, predicted data"))
# # dev.off()
# # """


# epsilon_predC = zeros(nkeepC)
# xi_predC = zeros(nkeepC)
# for i in 1:nkeepC
#     re = rand(MvLogNormal(zeros(2), Sigma_e_saveC[i,:,:]), 1)
#     epsilon_predC[i] = re[1]
#     xi_predC[i] = re[2]
# end

# epsilon_predT = zeros(nkeepT)
# xi_predT = zeros(nkeepT)
# for i in 1:nkeepT
#     re = rand(MvLogNormal(zeros(2), Sigma_e_saveT[i,:,:]), 1)
#     epsilon_predT[i] = re[1]
#     xi_predT[i] = re[2]
# end

# c_eC = hyperC["c_e"]
# C_eC = hyperC["C_e"]
# c_eT = hyperT["c_e"]
# C_eT = hyperT["C_e"]

# epsilon_priorC = zeros(nkeepC)
# xi_priorC = zeros(nkeepC)
# for i in 1:nkeepC 
#     Sigma_e_priorC = rand(InverseWishart(c_eC, C_eC), 1)[1]
#     re = rand(MvLogNormal(zeros(2), Sigma_e_priorC), 1)
#     epsilon_priorC[i] = re[1]
#     xi_priorC[i] = re[2]
# end 

# epsilon_priorT = zeros(nkeepT)
# xi_priorT = zeros(nkeepT)
# for i in 1:nkeepT 
#     Sigma_e_priorT = rand(InverseWishart(c_eT, C_eT), 1)[1]
#     re = rand(MvLogNormal(zeros(2), Sigma_e_priorT), 1)
#     epsilon_priorT[i] = re[1]
#     xi_priorT[i] = re[2]
# end 

# @rput epsilon_predC xi_predC
# @rput epsilon_predT xi_predT
# @rput epsilon_priorC xi_priorC
# @rput epsilon_priorT xi_priorT
# R"""
# png(paste0(fig_path, "random_effects_predC.png"), height=480, width=480)
# plot(epsilon_predC ~ xi_predC, main="", xlab="xi", ylab="epsilon", cex.axis=2)
# points(epsilon_priorC ~ xi_priorC, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
# dev.off() 

# png(paste0(fig_path, "random_effects_predT.png"), height=480, width=480)
# plot(epsilon_predT ~ xi_predT, main="", xlab="xi", ylab="epsilon", cex.axis=2)
# points(epsilon_priorT ~ xi_priorT, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
# dev.off() 

# png(paste0(fig_path, "random_effects_predC_recaled.png"), height=480, width=480)
# plot(epsilon_predC ~ xi_predC, main="", xlab="xi", ylab="epsilon", cex.axis=2, xlim=c(0,5), ylim=c(0,15))
# points(epsilon_priorC ~ xi_priorC, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
# dev.off() 

# png(paste0(fig_path, "random_effects_predT_recaled.png"), height=480, width=480)
# plot(epsilon_predT ~ xi_predT, main="", xlab="xi", ylab="epsilon", cex.axis=2, xlim=c(0,5), ylim=c(0,15))
# points(epsilon_priorT ~ xi_priorT, col=rgb(red = 0, green = 0, blue = 1, alpha = 0.5))
# dev.off() 
# """


# R"""
# png(paste0(fig_path, "epsilon_predC_hist.png"), height=480, width=480)
# hist(epsilon_predC, main="", xlab="xi", ylab="epsilon", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "epsilon_predT_hist.png"), height=480, width=480)
# hist(epsilon_predT, main="", xlab="xi", ylab="epsilon", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "xi_predC_hist.png"), height=480, width=480)
# hist(xi_predC, main="", xlab="xi", ylab="xi", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "xi_predT_hist.png"), height=480, width=480)
# hist(xi_predT, main="", xlab="xi", ylab="xi", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "epsilon_priorC_hist.png"), height=480, width=480)
# hist(epsilon_priorC, main="", xlab="xi", ylab="epsilon", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "epsilon_priorT_hist.png"), height=480, width=480)
# hist(epsilon_priorT, main="", xlab="xi", ylab="epsilon", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "xi_priorC_hist.png"), height=480, width=480)
# hist(xi_priorC, main="", xlab="xi", ylab="xi", cex.axis=2)
# dev.off() 

# png(paste0(fig_path, "xi_priorT_hist.png"), height=480, width=480)
# hist(xi_priorT, main="", xlab="xi", ylab="xi", cex.axis=2)
# dev.off() 
# """