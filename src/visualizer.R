library(ggplot2)

visualize_gap_times <- function(gap_times) {
  df <- data.frame(times = unlist(gap_times))
  p <- ggplot(data = df, aes(x = times)) + # nolint: object_usage_linter.
    geom_histogram(aes(y = after_stat(density)), bins = 10)
  p <- p + theme_bw(base_size = 25)
  return(p)
}

visualize_survival_times <- function(survival_times) {
  df <- data.frame(times = unlist(survival_times))
  p <- ggplot(data = df, aes(x = times)) + # nolint: object_usage_linter.
    geom_histogram(aes(y = after_stat(density)), bins = 10)
  p <- p + theme_bw(base_size = 25)
  return(p)
}

visualize_recurrent_events <- function(arrivial_times, survival_times) {
  n <- length(arrivial_times)
  p <- ggplot() 
  pd <- position_dodge2(width = 0.01)
  survival_times = unlist(survival_times)
  
  for(i in 1:n){
    m <- length(arrivial_times[[i]])
    times <- c(0, unlist(arrivial_times[[i]]), survival_times[[i]])
    counts <- c(seq(0, m, by = 1), m)
    if(length(times) != length(counts)){
      print(times)
      print(counts)
      print(i)
      print(arrivial_times[[i]])
      print(survival_times[i])
    }
    df <- data.frame(x = times, y = counts)
    p <- p + geom_step(
      data = df,
      aes(x = x, y = y),
      position = pd,
      linetype = 3
    )
  }
  counts <- unlist(lapply(arrivial_times, length))
  df <- data.frame(x = survival_times, y = counts)
  print(head(df))
  p <- p + geom_point(
    data = df,
    aes(x = x, y = y),
    position = pd,
    color = "red"
  )
  p <- p + theme_bw(base_size = 25)
  p <- p + xlab("t") + ylab("number of recurrent events")
  return(p)
}

visualize_num_recurrent_events <- function(counts) {
  df <- data.frame(x = counts)
  p <- ggplot(data = df)
  p <- p + geom_histogram(aes(x = x, y = after_stat(density)), bins = 10)
  p <- p + xlab("number of recurrent events")
  p <- p + theme_bw(base_size = 25)
  return(p)
}

visualize_functional <- function(dens, surv, haza, grids, times) {
  df <- data.frame(d = dens, s = surv, h = haza, x = grids)
  df_times <- data.frame(x = times)
  p <- ggplot(data = df)
  p <- p + theme_bw(base_size = 25)
  p_dens <- p + geom_line(aes(x = x, y = d))
  p_dens <- p_dens + geom_histogram(
    data = df_times,
    aes(x = x, y = after_stat(density)),
    bins = 10,
    alpha = 0.5
  )
  p_surv <- p + geom_line(aes(x = x, y = s))
  p_haza <- p + geom_line(aes(x = x, y = h))
  return(list("dens" = p_dens, "surv" = p_surv, "haza" = p_haza))
}

visualize_survival_functional <- function(dens, surv, haza, grids, times) {
  res <- visualize_functional(dens, surv, haza, grids, times)
  p_dens <- res$dens + xlab("t") + ylab("Density")
  p_surv <- res$surv + xlab("t") + ylab("Survival")
  p_haza <- res$haza + xlab("t") + ylab("Hazard")
  return(list("dens" = p_dens, "surv" = p_surv, "haza" = p_haza))
}

visualize_gap_functional <- function(dens, surv, haza, grids, times) {
  res <- visualize_functional(dens, surv, haza, grids, times)
  p_dens <- res$dens + xlab("y") + ylab("Density")
  p_surv <- res$surv + xlab("y") + ylab("Survival")
  p_haza <- res$haza + xlab("y") + ylab("Hazard")
  return(list("dens" = p_dens, "surv" = p_surv, "haza" = p_haza))
}

KM_conditional_N0 <- function(surv, Nvec, nu, t0){
  index = which(surv>=t0 & Nvec==0)
  fit = survival::survfit(survival::Surv(surv[index], nu[index])~1)
  return(fit)
}