library(ggplot2)

visualize_gap_times <- function(gap_times) {
    df <- data.frame(times = unlist(gap_times))
    p <- ggplot(data = df, aes(x = times)) +
         geom_histogram(aes(y = after_stat(density)), bins = 20)
    return(p)
}

visualize_survival_times <- function(survival_times) {
    df <- data.frame(times = unlist(survival_times))
    p <- ggplot(data = df, aes(x = times)) +
         geom_histogram(aes(y = after_stat(density)), bins = 20)
    return(p)
}

visualize_recurrent_events <- function(arrivial_times, survival_times) {
    n <- length(arrivial_times)
    p <- ggplot()
    for(i in 1:n){
        m <- length(arrivial_times[[i]])
        times <- c(0, unlist(arrivial_times[[i]]), survival_times[[i]])
        counts <- c(seq(0, m, by = 1), m)
        df <- data.frame(x = times, y = counts)
        p <- p + geom_step(data = df, mapping = aes(x = x, y = y), linetype = 3)
    }
    counts <- unlist(lapply(arrivial_times, function(x) length(x) + 1))
    df <- data.frame(x = survival_times, y = counts)
    p <- p + geom_point(data = df, aes(x = x, y = y), color = "red")
    return(p)
}