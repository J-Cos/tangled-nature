#!/usr/bin/env Rscript
# ============================================================================
# Tangled Nature Model — Bayesian Analysis of Adiabatic Hysteresis Data
#
# Usage:
#   Rscript analysis.R                           # default: results.csv
#   Rscript analysis.R path/to/results.csv       # custom input
#
# Outputs posterior summaries and hypothesis test results to stdout.
# ============================================================================

suppressPackageStartupMessages({
  library(brms)
  library(dplyr)
  library(ggplot2)
})

# Check for marginaleffects (optional but recommended)
has_marginaleffects <- requireNamespace("marginaleffects", quietly = TRUE)
if (has_marginaleffects) library(marginaleffects)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else "results.csv"

cat("═══════════════════════════════════════════════════════════\n")
cat("  TNM Bayesian Analysis (brms)\n")
cat("═══════════════════════════════════════════════════════════\n")
cat(sprintf("  Input: %s\n", csv_path))

if (!file.exists(csv_path)) {
  stop(sprintf("File not found: %s", csv_path))
}

sim_data <- read.csv(csv_path, stringsAsFactors = FALSE)
sim_data$Phase <- factor(sim_data$Phase)
sim_data$Replicate <- factor(sim_data$Replicate)

cat(sprintf("  Rows: %d | Replicates: %d | Phases: %s\n",
    nrow(sim_data), length(unique(sim_data$Replicate)),
    paste(levels(sim_data$Phase), collapse = ", ")))
cat(sprintf("  Mu range: [%.4f, %.4f]\n", min(sim_data$Mu), max(sim_data$Mu)))
cat(sprintf("  N range: [%d, %d]\n", min(sim_data$N), max(sim_data$N)))
cat(sprintf("  S range: [%d, %d]\n", min(sim_data$S), max(sim_data$S)))
cat("\n")

# --------------------------------------------------------------------------
# Safety checks
# --------------------------------------------------------------------------

n_forward <- sum(sim_data$Phase == "Forward")
n_reverse <- sum(sim_data$Phase == "Reverse")
n_unique_mu <- length(unique(sim_data$Mu))

cat(sprintf("  Forward rows: %d | Reverse rows: %d | Unique Mu: %d\n",
    n_forward, n_reverse, n_unique_mu))

if (nrow(sim_data) < 10) {
  cat("\n⚠ WARNING: Very few data points. brms models may not converge.\n")
  cat("  Run the orchestrator in --mode full for reliable inference.\n\n")
}

# Compute N_ratio for EWS subsetting
baseline_N <- sim_data %>%
  filter(Phase == "Burn-in") %>%
  group_by(Replicate) %>%
  summarise(N_baseline = mean(N), .groups = "drop")

sim_data <- sim_data %>%
  left_join(baseline_N, by = "Replicate") %>%
  mutate(N_ratio = N / N_baseline)

# --------------------------------------------------------------------------
# Model 1: Hysteresis Test (State-Dependent Splines)
# --------------------------------------------------------------------------

cat("══════════════════════════════════════════════════════════\n")
cat("  Model 1: HYSTERESIS TEST\n")
cat("  N ~ s(Mu, by=Phase) + Phase + (1|Replicate)\n")
cat("══════════════════════════════════════════════════════════\n\n")

# Filter to Forward + Reverse only (exclude Burn-in for hysteresis test)
hyst_data <- sim_data %>% filter(Phase %in% c("Forward", "Reverse"))
hyst_data$Phase <- droplevels(hyst_data$Phase)

# Determine appropriate spline complexity
k_val <- min(15, max(3, floor(n_unique_mu / 2)))
cat(sprintf("  Using k=%d for splines (based on %d unique Mu values)\n\n", k_val, n_unique_mu))

if (nrow(hyst_data) >= 10 && n_unique_mu >= 3) {
  fit_hysteresis <- brm(
    bf(N ~ s(Mu, by = Phase, k = k_val) + Phase + (1 | Replicate)),
    family = student(),
    data = hyst_data,
    prior = c(
      prior(normal(0, 10000), class = "Intercept"),
      prior(gamma(2, 0.1), class = "nu")
    ),
    cores = 16, chains = 4, iter = 2000, warmup = 1000,
    control = list(max_treedepth = 12, adapt_delta = 0.9),
    silent = 0, refresh = 100
  )

  cat("  --- Hysteresis Model Summary ---\n")
  print(summary(fit_hysteresis))

  # Posterior predictive comparison at matched Mu values
  cat("\n  --- Hysteresis Inference ---\n")
  matched_mu <- sort(unique(hyst_data$Mu))
  cat(sprintf("  Testing Forward vs Reverse at %d matched Mu values\n", length(matched_mu)))

  if (has_marginaleffects) {
    tryCatch({
      # Use re_formula=NA to marginalise over random effects
      nd <- expand.grid(
        Mu = matched_mu,
        Phase = c("Forward", "Reverse")
      )
      comp <- comparisons(fit_hysteresis, variables = "Phase",
                          newdata = nd, re_formula = NA)
      cat("\n  Posterior difference (Forward - Reverse):\n")
      print(summary(comp))

      # Check if 95% CI excludes zero
      s <- summary(comp)
      if (all(s$conf.low > 0) || all(s$conf.high < 0)) {
        cat("\n  ★ RESULT: 95% CI systematically excludes zero → HYSTERESIS CONFIRMED\n")
      } else {
        cat("\n  ★ RESULT: 95% CI includes zero → No clear hysteresis detected\n")
      }
    }, error = function(e) {
      cat(sprintf("\n  ⚠ marginaleffects error: %s\n", conditionMessage(e)))
      cat("  Falling back to direct posterior comparison...\n")
      # Simple posterior comparison: predict at matched μ for both phases
      nd_fwd <- data.frame(Mu = matched_mu, Phase = factor("Forward", levels = levels(hyst_data$Phase)))
      nd_rev <- data.frame(Mu = matched_mu, Phase = factor("Reverse", levels = levels(hyst_data$Phase)))
      p_fwd <- fitted(fit_hysteresis, newdata = nd_fwd, re_formula = NA, summary = TRUE)
      p_rev <- fitted(fit_hysteresis, newdata = nd_rev, re_formula = NA, summary = TRUE)
      delta <- p_fwd[, "Estimate"] - p_rev[, "Estimate"]
      cat(sprintf("  Mean ΔN (Forward - Reverse) across μ: %.2f\n", mean(delta)))
      cat(sprintf("  Range: [%.2f, %.2f]\n", min(delta), max(delta)))
      if (all(delta > 0) || all(delta < 0)) {
        cat("\n  ★ RESULT: Posterior means consistently differ → HYSTERESIS DETECTED\n")
      } else {
        cat("\n  ★ RESULT: Posterior means cross zero → No clear hysteresis\n")
      }
    })
  } else {
    cat("  (Install 'marginaleffects' package for posterior comparisons)\n")
  }
} else {
  cat("  ⚠ Insufficient data for Model 1. Need ≥10 rows and ≥3 unique Mu.\n")
}

# --------------------------------------------------------------------------
# Model 2: EWS / Topological Masking Test
# --------------------------------------------------------------------------

cat("\n══════════════════════════════════════════════════════════\n")
cat("  Model 2: EARLY WARNING SIGNAL TEST\n")
cat("  Pre-collapse Forward data (N > 0.8 × N_baseline)\n")
cat("  METE_DKL ~ s(Mu) + (1|Replicate)\n")
cat("  Lambda_2  ~ s(Mu) + (1|Replicate)\n")
cat("══════════════════════════════════════════════════════════\n\n")

pre_collapse <- sim_data %>%
  filter(Phase == "Forward", N_ratio > 0.8) %>%
  filter(!is.na(METE_DKL), !is.na(Lambda_2))

cat(sprintf("  Pre-collapse Forward rows: %d\n", nrow(pre_collapse)))

if (nrow(pre_collapse) >= 10 && length(unique(pre_collapse$Mu)) >= 3) {

  k_ews <- min(10, max(3, floor(length(unique(pre_collapse$Mu)) / 2)))

  fit_ews <- brm(
    mvbf(
      bf(METE_DKL ~ s(Mu, k = k_ews) + (1 | Replicate)),
      bf(Lambda_2 ~ s(Mu, k = k_ews) + (1 | Replicate))
    ),
    data = pre_collapse,
    cores = 16, chains = 4, iter = 2000, warmup = 1000,
    control = list(max_treedepth = 12, adapt_delta = 0.9),
    silent = 0, refresh = 100
  )

  cat("  --- EWS Model Summary ---\n")
  print(summary(fit_ews))

  if (has_marginaleffects) {
    tryCatch({
      cat("\n  --- EWS Inference: Posterior Derivatives ---\n")

      nd_ews <- data.frame(Mu = sort(unique(pre_collapse$Mu)))

      # Derivative of D_KL w.r.t. Mu
      slopes_dkl <- slopes(fit_ews, resp = "METEDKL", variables = "Mu",
                           newdata = nd_ews, re_formula = NA)
      cat("\n  D_KL derivative w.r.t. Mu:\n")
      print(summary(slopes_dkl))

      # Derivative of Lambda_2 w.r.t. Mu
      slopes_lam <- slopes(fit_ews, resp = "Lambda2", variables = "Mu",
                           newdata = nd_ews, re_formula = NA)
      cat("\n  Lambda_2 derivative w.r.t. Mu:\n")
      print(summary(slopes_lam))

      # Inference
      s_dkl <- summary(slopes_dkl)
      s_lam <- summary(slopes_lam)
      dkl_positive <- all(s_dkl$conf.low > 0)
      lam_negative <- all(s_lam$conf.high < 0)

      if (dkl_positive && lam_negative) {
        cat("\n  ★ RESULT: D_KL increasing AND λ₂ decreasing pre-collapse\n")
        cat("    → TOPOLOGICAL MASKING CONFIRMED (Hypothesis C)\n")
      } else if (dkl_positive) {
        cat("\n  ★ RESULT: D_KL increasing but λ₂ not clearly decreasing\n")
        cat("    → Partial evidence for structural warping\n")
      } else if (lam_negative) {
        cat("\n  ★ RESULT: λ₂ decreasing but D_KL not clearly increasing\n")
        cat("    → Partial evidence for topological fragmentation\n")
      } else {
        cat("\n  ★ RESULT: Neither D_KL nor λ₂ show clear directional trends\n")
        cat("    → No evidence for topological masking\n")
      }
    }, error = function(e) {
      cat(sprintf("\n  ⚠ marginaleffects error: %s\n", conditionMessage(e)))
      cat("  Model summary above still provides valid posterior estimates.\n")
    })
  } else {
    cat("  (Install 'marginaleffects' package for derivative analysis)\n")
  }
} else {
  cat("  ⚠ Insufficient pre-collapse data for Model 2.\n")
  cat("    Need ≥10 rows with N > 0.8 × N_baseline and ≥3 unique Mu.\n")
}

cat("\n══════════════════════════════════════════════════════════\n")
cat("  ANALYSIS COMPLETE\n")
cat("══════════════════════════════════════════════════════════\n")
