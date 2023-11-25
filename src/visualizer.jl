using RCall 

function data_visualizer(dat, filename="")
    """This function visualizes the joint process of recurrent 
    and survival times. This function uses R interface

    Args:
        dat: Dict 
    """

    survival_times = dat["survival_times"]
    gap_times = dat["gap_times"]
    arrivival_times = dat["arrivival_times"]
    counts = dat["counts"]
    Gₛ = dat["dp_surv"]
    Gᵣ = dat["dp_gap"]
    Gₑ = dat["dp_re"]

    re_params = tuple.(Gₑ.atoms[1][1,:], Gₑ.atoms[1][2,:])
    nreps = 2000
    re_reps = sample(re_params, Weights(Gₑ.weights), nreps)

    ngrids = 200
    surv_grids = range(0.01, Base.maximum(survival_times), length = ngrids)

    surv_dens = zeros(ngrids)
    surv_surv = zeros(ngrids)
    gap_dens = zeros(ngrids)
    gap_surv = zeros(ngrids)
    gap_grids = zeros(ngrids)
    for i in 1:nreps
        surv_params = tuple.(exp.(Gₛ.atoms[1] .- re_reps[i][1]), sqrt.(Gₛ.atoms[2]))
        gap_params = tuple.(exp.(Gᵣ.atoms[1] .- re_reps[i][2]), sqrt.(Gᵣ.atoms[2]))

        survival_dist = MixtureModel(LogLogistic, surv_params, Gₛ.weights)
        for g in 1:ngrids
            surv_dens[g] += pdf.(survival_dist, surv_grids[g]) / nreps
            surv_surv[g] += ccdf.(survival_dist, surv_grids[g]) / nreps
        end
        if sum(counts) == 0 # no recurrent events 
            gap_grids = range(0.01, Base.maximum(survival_times), length = ngrids)
        else
            gap_grids = range(0.01, Base.maximum(Base.maximum(gap_times)), length = ngrids)
        end
        gap_dist = MixtureModel(LogLogistic, gap_params, Gᵣ.weights)
        for g in 1:ngrids
            gap_dens[g] = pdf.(gap_dist, gap_grids[g])
            gap_surv[g] = ccdf.(gap_dist, gap_grids[g])
        end 
    end
    surv_haza = surv_dens ./ surv_surv
    gap_haza = gap_dens ./ gap_surv

    @rput survival_times gap_times arrivival_times counts 
    @rput surv_dens surv_surv surv_haza surv_grids 
    @rput gap_dens gap_surv gap_haza gap_grids
    @rput filename
    R"""
    source("src/visualizer.R")
    p_gap = visualize_gap_times(gap_times)
    ggsave(paste0("figs/", filename, "/gap.png"), p_gap)
    p_surv = visualize_survival_times(survival_times)
    ggsave(paste0("figs/", filename, "/surv.png"), p_surv)
    p_recur = visualize_recurrent_events(arrivival_times, survival_times)
    ggsave(paste0("figs/", filename, "/recur.png"), p_recur)
    p_count = visualize_num_recurrent_events(counts)
    ggsave(paste0("figs/", filename, "/counts.png"), p_count)

    p_surv_func = visualize_survival_functional(surv_dens, surv_surv, surv_haza, surv_grids, survival_times)
    ggsave(paste0("figs/", filename, "/surv_dens.png"), p_surv_func$dens)
    ggsave(paste0("figs/", filename, "/surv_surv.png"), p_surv_func$surv)
    ggsave(paste0("figs/", filename, "/surv_haza.png"), p_surv_func$haza)

    p_gap_func = visualize_gap_functional(gap_dens, gap_surv, gap_haza, gap_grids, unlist(gap_times))
    ggsave(paste0("figs/", filename, "/gap_dens.png"), p_gap_func$dens)
    ggsave(paste0("figs/", filename, "/gap_surv.png"), p_gap_func$surv)
    ggsave(paste0("figs/", filename, "/gap_haza.png"), p_gap_func$haza)
    """
end


function conditional_probability(dat, t0, filename="")

    survival_times = dat["survival_times"]
    Gₛ = dat["dp_surv"]
    Gᵣ = dat["dp_gap"]
    Gₑ = dat["dp_re"]

    re_params = tuple.(Gₑ.atoms[1][1,:], Gₑ.atoms[1][2,:])
    nreps = 2000
    re_reps = sample(re_params, Weights(Gₑ.weights), nreps)

    ngrids = 200
    surv_grids = range(t0, Base.maximum(survival_times), length = ngrids)

    surv_surv = zeros(ngrids, nreps)
    surv_surv0 = zeros(nreps)
    gap_surv0 = zeros(nreps)
    for i in 1:nreps
        surv_params = tuple.(exp.(Gₛ.atoms[1] .- re_reps[i][1]), sqrt.(Gₛ.atoms[2]))

        survival_dist = MixtureModel(LogLogistic, surv_params, Gₛ.weights)
        for g in 1:ngrids
            surv_surv[g,i] = ccdf.(survival_dist, surv_grids[g])
        end
        surv_surv0[i] = ccdf(survival_dist, t0)

        gap_params = tuple.(exp.(Gᵣ.atoms[1] .- re_reps[i][2]), sqrt.(Gᵣ.atoms[2]))
        gap_dist = MixtureModel(LogLogistic, gap_params, Gᵣ.weights)

        gap_surv0[i] = ccdf(gap_dist, t0)
    end
    
    cond_surv = zeros(ngrids)
    for g in 1:ngrids 
        cond_surv[g] = mean(surv_surv[g,:] .* surv_surv0) 
    end 
    cond_surv = cond_surv ./ mean(surv_surv0 .* gap_surv0)
    
    @rput cond_surv 
end