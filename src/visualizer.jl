using RCall 

function data_visualizer(dat)
    """This function visualizes the joint process of recurrent 
    and survival times. This function uses R interface

    Args:
        dat: Dict 
    """

    survival_times = dat["survival_times"]
    gap_times = dat["gap_times"]
    arrivival_times = dat["arrivival_times"]

    @rput survival_times gap_times arrivival_times
    R"""
    # print(getwd())
    source("src/visualizer.R")
    p_gap = visualize_gap_times(gap_times)
    ggsave("gap.png", p_gap)
    p_surv = visualize_survival_times(survival_times)
    ggsave("surv.png", p_surv)
    p_recur = visualize_recurrent_events(arrivival_times, survival_times)
    ggsave("recur.png", p_recur)
    """
end