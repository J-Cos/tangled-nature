#!/usr/bin/env Rscript
# ──────────────────────────────────────────────────────────────────
# METE Analysis of TNM Ensemble
#
# Uses meteR to compare observed SADs against METE predictions.
# Input:  results/sad_snapshots.csv  (gen, sim_id, genome, abundance)
# Output: results/mete_metrics.csv   (gen, sim_id, S, N, kl_div,
#                                      r_squared, ks_pvalue,
#                                      shannon_obs, shannon_pred)
# ──────────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(meteR)
})

# ── Helper functions ────────────────────────────────────────────

#' Compute Shannon entropy from abundance vector
shannon <- function(n_vec) {
  n_vec <- n_vec[n_vec > 0]
  p <- n_vec / sum(n_vec)
  -sum(p * log(p))
}

#' Compute KL divergence: D_KL(P || Q) where P=observed, Q=predicted
#' Both are abundance vectors that get converted to probabilities
kl_divergence <- function(obs_p, pred_p) {
  # Add small epsilon to avoid log(0)
  eps <- 1e-12
  obs_p <- obs_p + eps
  pred_p <- pred_p + eps
  obs_p <- obs_p / sum(obs_p)
  pred_p <- pred_p / sum(pred_p)
  sum(obs_p * log(obs_p / pred_p))
}

#' Fit METE SAD and compute goodness-of-fit metrics
#' Returns a named list of metrics, or NULL if fitting fails
fit_mete_sad <- function(abundances) {
  S0 <- length(abundances)
  N0 <- sum(abundances)

  # Need at least 2 species and more individuals than species
  if (S0 < 2 || N0 < S0) {
    return(NULL)
  }

  tryCatch({
    # Fit METE
    esf <- meteESF(S0 = S0, N0 = N0)
    mete_sad <- sad(esf)

    # Observed rank-abundance (sorted descending)
    obs_ranks <- sort(abundances, decreasing = TRUE)

    # METE predicted rank-abundance
    # meteDist2Rank gives predicted abundances at each rank
    pred_ranks <- meteDist2Rank(mete_sad)

    # Ensure same length (should be S0 for both)
    len <- min(length(obs_ranks), length(pred_ranks))
    obs_r <- obs_ranks[1:len]
    pred_r <- pred_ranks[1:len]

    # R² of rank-abundance curve (log scale)
    log_obs <- log(obs_r + 1)
    log_pred <- log(pred_r + 1)
    ss_res <- sum((log_obs - log_pred)^2)
    ss_tot <- sum((log_obs - mean(log_obs))^2)
    r_sq <- ifelse(ss_tot > 0, 1 - ss_res / ss_tot, NA)

    # KS test (comparing distributions)
    ks_result <- tryCatch(
      ks.test(obs_ranks, pred_ranks)$p.value,
      error = function(e) NA
    )

    # KL divergence (on rank-abundance probabilities)
    kl <- kl_divergence(obs_r, pred_r)

    # Shannon entropy
    h_obs <- shannon(abundances)
    h_pred <- shannon(pred_ranks)

    list(
      S = S0,
      N = N0,
      kl_div = kl,
      r_squared = r_sq,
      ks_pvalue = ks_result,
      shannon_obs = h_obs,
      shannon_pred = h_pred
    )
  }, error = function(e) {
    NULL
  })
}


# ── Main analysis ────────────────────────────────────────────────

run_analysis <- function(results_dir) {
  sad_file <- file.path(results_dir, "sad_snapshots.csv")
  if (!file.exists(sad_file)) {
    cat("Error: SAD snapshots file not found at", sad_file, "\n")
    return(invisible(NULL))
  }

  cat("Loading SAD snapshots...\n")
  sad_data <- read.csv(sad_file, stringsAsFactors = FALSE)
  cat("  Loaded", nrow(sad_data), "species records\n")

  # Get unique (gen, sim_id) combinations
  snapshots <- unique(sad_data[, c("gen", "sim_id")])
  snapshots <- snapshots[order(snapshots$sim_id, snapshots$gen), ]
  n_snapshots <- nrow(snapshots)
  cat("  Found", n_snapshots, "unique snapshots\n")

  # Pre-split abundances by snapshot for efficient parallel access
  cat("  Pre-splitting SAD data by snapshot...\n")
  snap_keys <- paste(sad_data$gen, sad_data$sim_id, sep = "_")
  abd_by_snap <- split(sad_data$abundance, snap_keys)

  # Detect available cores
  n_cores <- parallel::detectCores(logical = FALSE)
  if (is.na(n_cores) || n_cores < 1) n_cores <- 1
  cat(sprintf("  Using %d cores for parallel METE fitting\n", n_cores))

  # Build task list
  task_list <- lapply(seq_len(n_snapshots), function(i) {
    list(gen = snapshots$gen[i], sim_id = snapshots$sim_id[i],
         key = paste(snapshots$gen[i], snapshots$sim_id[i], sep = "_"))
  })

  # Parallel METE fitting
  metrics_list <- parallel::mclapply(task_list, function(task) {
    abundances <- abd_by_snap[[task$key]]
    if (is.null(abundances) || length(abundances) == 0) return(NULL)

    result <- fit_mete_sad(abundances)
    if (is.null(result)) return(NULL)

    data.frame(
      gen = task$gen,
      sim_id = task$sim_id,
      S = result$S,
      N = result$N,
      kl_div = result$kl_div,
      r_squared = result$r_squared,
      ks_pvalue = result$ks_pvalue,
      shannon_obs = result$shannon_obs,
      shannon_pred = result$shannon_pred,
      stringsAsFactors = FALSE
    )
  }, mc.cores = n_cores)

  # Combine results
  metrics <- do.call(rbind, metrics_list[!sapply(metrics_list, is.null)])

  if (is.null(metrics) || nrow(metrics) == 0) {
    cat("Error: No valid METE fits produced\n")
    return(invisible(NULL))
  }

  out_path <- file.path(results_dir, "mete_metrics.csv")
  write.csv(metrics, out_path, row.names = FALSE)
  cat(sprintf("  → Wrote %d metrics rows to %s\n", nrow(metrics), out_path))

  # Summary statistics
  cat("\n=== METE Fit Summary ===\n")
  cat(sprintf("  Mean R²:       %.3f (SD %.3f)\n",
              mean(metrics$r_squared, na.rm = TRUE),
              sd(metrics$r_squared, na.rm = TRUE)))
  cat(sprintf("  Mean KL div:   %.3f (SD %.3f)\n",
              mean(metrics$kl_div, na.rm = TRUE),
              sd(metrics$kl_div, na.rm = TRUE)))
  cat(sprintf("  Mean KS p:     %.3f (SD %.3f)\n",
              mean(metrics$ks_pvalue, na.rm = TRUE),
              sd(metrics$ks_pvalue, na.rm = TRUE)))
  cat(sprintf("  Shannon obs:   %.3f, pred: %.3f\n",
              mean(metrics$shannon_obs, na.rm = TRUE),
              mean(metrics$shannon_pred, na.rm = TRUE)))

  invisible(metrics)
}

# ── Entry point ──────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)
results_dir <- if (length(args) > 0) args[1] else "results"
run_analysis(results_dir)
