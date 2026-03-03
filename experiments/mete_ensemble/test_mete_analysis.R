#!/usr/bin/env Rscript
# Unit tests for mete_analysis.R
# Run: Rscript test_mete_analysis.R

suppressPackageStartupMessages(library(meteR))

# Define the functions under test directly (avoids triggering run_analysis at source time)
shannon <- function(n_vec) {
  n_vec <- n_vec[n_vec > 0]
  p <- n_vec / sum(n_vec)
  -sum(p * log(p))
}

kl_divergence <- function(obs_p, pred_p) {
  eps <- 1e-12
  obs_p <- obs_p + eps
  pred_p <- pred_p + eps
  obs_p <- obs_p / sum(obs_p)
  pred_p <- pred_p / sum(pred_p)
  sum(obs_p * log(obs_p / pred_p))
}

fit_mete_sad <- function(abundances) {
  S0 <- length(abundances)
  N0 <- sum(abundances)
  if (S0 < 2 || N0 < S0) return(NULL)
  tryCatch({
    esf <- meteESF(S0 = S0, N0 = N0)
    mete_sad <- sad(esf)
    obs_ranks <- sort(abundances, decreasing = TRUE)
    pred_ranks <- meteDist2Rank(mete_sad)
    len <- min(length(obs_ranks), length(pred_ranks))
    obs_r <- obs_ranks[1:len]
    pred_r <- pred_ranks[1:len]
    log_obs <- log(obs_r + 1)
    log_pred <- log(pred_r + 1)
    ss_res <- sum((log_obs - log_pred)^2)
    ss_tot <- sum((log_obs - mean(log_obs))^2)
    r_sq <- ifelse(ss_tot > 0, 1 - ss_res / ss_tot, NA)
    ks_result <- tryCatch(ks.test(obs_ranks, pred_ranks)$p.value, error = function(e) NA)
    kl <- kl_divergence(obs_r, pred_r)
    h_obs <- shannon(abundances)
    h_pred <- shannon(pred_ranks)
    list(S = S0, N = N0, kl_div = kl, r_squared = r_sq,
         ks_pvalue = ks_result, shannon_obs = h_obs, shannon_pred = h_pred)
  }, error = function(e) NULL)
}

# Helper
assert <- function(cond, msg) {
  if (!cond) stop(paste("FAIL:", msg))
  cat(paste("PASS:", msg, "\n"))
}

# ── Test 1: Shannon entropy ──────────────────────────────────────
test_shannon <- function() {
  # Uniform distribution: H = log(S)
  h <- shannon(rep(10, 5))
  assert(abs(h - log(5)) < 1e-10, "Shannon entropy of uniform dist = log(5)")

  # Single species: H = 0
  h <- shannon(c(100))
  assert(abs(h - 0) < 1e-10, "Shannon entropy of single species = 0")

  # Known case: [1,1] -> log(2)
  h <- shannon(c(1, 1))
  assert(abs(h - log(2)) < 1e-10, "Shannon entropy of [1,1] = log(2)")
}

# ── Test 2: KL divergence ────────────────────────────────────────
test_kl_divergence <- function() {
  # KL(P || P) ≈ 0
  kl <- kl_divergence(c(10, 20, 30), c(10, 20, 30))
  assert(kl < 1e-6, "KL divergence of identical distributions ≈ 0")

  # KL(P || Q) > 0 for P != Q
  kl <- kl_divergence(c(10, 20, 30), c(30, 20, 10))
  assert(kl > 0, "KL divergence of different distributions > 0")
}

# ── Test 3: fit_mete_sad with known data ─────────────────────────
test_fit_basic <- function() {
  # Simple case: 10 species, 100 individuals
  abundances <- c(50, 15, 10, 7, 5, 4, 3, 3, 2, 1)
  result <- fit_mete_sad(abundances)
  assert(!is.null(result), "fit_mete_sad returns non-NULL for valid data")
  assert(result$S == 10, "S correctly reported as 10")
  assert(result$N == 100, "N correctly reported as 100")
  assert(is.numeric(result$r_squared), "R² is numeric")
  assert(is.numeric(result$kl_div), "KL divergence is numeric")
  assert(result$kl_div >= 0, "KL divergence is non-negative")
}

# ── Test 4: fit_mete_sad edge cases ──────────────────────────────
test_fit_edge_cases <- function() {
  # Single species — should return NULL
  result <- fit_mete_sad(c(100))
  assert(is.null(result), "fit_mete_sad returns NULL for S=1")

  # Two species
  result <- fit_mete_sad(c(90, 10))
  assert(!is.null(result), "fit_mete_sad works for S=2")

  # Empty — should return NULL
  result <- fit_mete_sad(integer(0))
  assert(is.null(result), "fit_mete_sad returns NULL for empty input")
}

# ── Test 5: METE predicts log-series-like SAD well ───────────────
test_mete_log_series_fit <- function() {
  # Generate METE's own predicted SAD, then check it fits itself well
  S0 <- 20
  N0 <- 500
  esf <- meteESF(S0 = S0, N0 = N0)
  pred <- meteDist2Rank(sad(esf))
  # Use METE's own prediction as "observed" — should fit perfectly
  result <- fit_mete_sad(round(pred))
  assert(!is.null(result), "fit_mete_sad works on METE-generated SAD")
  assert(result$r_squared > 0.9, paste("R² > 0.9 for METE's own SAD, got:", result$r_squared))
}

# ── Run all tests ────────────────────────────────────────────────
cat("=== METE Analysis Unit Tests ===\n\n")
test_shannon()
test_kl_divergence()
test_fit_basic()
test_fit_edge_cases()
test_mete_log_series_fit()
cat("\nAll tests passed!\n")
