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
    G = dat["dp"]

    surv_grids = range(0.01, Base.maximum(survival_times), length = 200)
    gap_grids = range(0.01, Base.maximum(Base.maximum(gap_times)), length = 200)

    surv_params = tuple.(G.θₛ, G.ϕₛ)
    survival_dist = MixtureModel(LogLogistic, surv_params, G.weights)
    surv_dens = pdf.(survival_dist, surv_grids)
    surv_surv = ccdf.(survival_dist, surv_grids)
    surv_haza = surv_dens ./ surv_surv

    gap_params = tuple.(G.θᵣ, G.ϕᵣ)
    gap_dist = MixtureModel(LogLogistic, gap_params, G.weights)
    gap_dens = pdf.(gap_dist, gap_grids)
    gap_surv = ccdf.(gap_dist, gap_grids)
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

    p_surv_func = visualize_survival_functional(surv_dens, surv_surv, surv_haza, surv_grids)
    ggsave(paste0("figs/", filename, "/surv_dens.png"), p_surv_func$dens)
    ggsave(paste0("figs/", filename, "/surv_surv.png"), p_surv_func$surv)
    ggsave(paste0("figs/", filename, "/surv_haza.png"), p_surv_func$haza)

    p_gap_func = visualize_gap_functional(gap_dens, gap_surv, gap_haza, gap_grids)
    ggsave(paste0("figs/", filename, "/gap_dens.png"), p_gap_func$dens)
    ggsave(paste0("figs/", filename, "/gap_surv.png"), p_gap_func$surv)
    ggsave(paste0("figs/", filename, "/gap_haza.png"), p_gap_func$haza)
    """
end