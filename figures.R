#!/usr/bin/env Rscript
# ============================================================================
# PNAS-Quality Multi-Panel Figures for TNM Adiabatic Hysteresis Experiment
#
# Generates three publication figures:
#   Figure 1 — Hysteresis loop demonstration
#   Figure 2 — Early warning signals & topological masking
#   Figure 3 — Mechanistic summary: population, structure, topology
#
# Usage:
#   Rscript figures.R                        # default: results.csv
#   Rscript figures.R path/to/results.csv    # custom input
# ============================================================================

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

# Check for patchwork (multi-panel layout)
if (!requireNamespace("patchwork", quietly = TRUE)) {
  cat("Installing patchwork for multi-panel layout...\n")
  install.packages("patchwork", repos = "https://cloud.r-project.org")
}
library(patchwork)

# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

args <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[1] else "results.csv"

cat("═══════════════════════════════════════════════════════════\n")
cat("  PNAS Figure Generator — TNM Adiabatic Hysteresis\n")
cat("═══════════════════════════════════════════════════════════\n")
cat(sprintf("  Input: %s\n", csv_path))

if (!file.exists(csv_path)) stop(sprintf("File not found: %s", csv_path))

d <- read.csv(csv_path, stringsAsFactors = FALSE)
d$Phase <- factor(d$Phase, levels = c("Burn-in", "Forward", "Reverse"))
d$Replicate <- factor(d$Replicate)

cat(sprintf("  Rows: %d | Replicates: %d\n", nrow(d), length(unique(d$Replicate))))

# Compute baseline N per replicate
baseline <- d %>%
  filter(Phase == "Burn-in") %>%
  group_by(Replicate) %>%
  summarise(N_baseline = mean(N), .groups = "drop")

d <- d %>%
  left_join(baseline, by = "Replicate") %>%
  mutate(N_ratio = N / N_baseline)

# Subsets
fwd <- d %>% filter(Phase == "Forward")
rev <- d %>% filter(Phase == "Reverse")
fr  <- d %>% filter(Phase %in% c("Forward", "Reverse"))

# Pre-collapse: forward data where system is still > 80% of baseline
pre_collapse <- fwd %>% filter(N_ratio > 0.8)

# --------------------------------------------------------------------------
# PNAS Theme
# --------------------------------------------------------------------------

theme_pnas <- function(base_size = 8) {
  theme_classic(base_size = base_size) %+replace%
  theme(
    # Text
    text = element_text(family = "Helvetica", colour = "grey10"),
    plot.title = element_text(size = rel(1.2), face = "bold",
                              hjust = 0, margin = margin(b = 4)),
    plot.subtitle = element_text(size = rel(0.85), colour = "grey40",
                                 hjust = 0, margin = margin(b = 6)),
    axis.title = element_text(size = rel(1.0), face = "plain"),
    axis.text = element_text(size = rel(0.85), colour = "grey30"),
    strip.text = element_text(size = rel(0.9), face = "bold"),

    # Panel
    panel.grid.major = element_line(colour = "grey92", linewidth = 0.3),
    panel.grid.minor = element_blank(),
    panel.border = element_rect(colour = "grey60", fill = NA, linewidth = 0.4),
    axis.line = element_blank(),

    # Legend
    legend.position = "bottom",
    legend.title = element_text(size = rel(0.85), face = "italic"),
    legend.text = element_text(size = rel(0.8)),
    legend.key.size = unit(0.4, "cm"),
    legend.margin = margin(t = 2),

    # Margins
    plot.margin = margin(6, 8, 4, 6)
  )
}

# Colour palette
col_fwd <- "#D64933"   # burnt orange-red
col_rev <- "#2176AE"   # steel blue
col_fwd_fill <- "#D6493330"
col_rev_fill <- "#2176AE30"
col_accent <- "#4CAF50"  # green accent
col_warn <- "#FF9800"    # amber

# --------------------------------------------------------------------------
# Summary statistics
# --------------------------------------------------------------------------

fr_summary <- fr %>%
  group_by(Phase, Mu) %>%
  summarise(
    N_mean = mean(N), N_se = sd(N) / sqrt(n()),
    S_mean = mean(S), S_se = sd(S) / sqrt(n()),
    DKL_mean = mean(METE_DKL, na.rm = TRUE),
    DKL_se = sd(METE_DKL, na.rm = TRUE) / sqrt(sum(!is.na(METE_DKL))),
    L2_mean = mean(Lambda_2, na.rm = TRUE),
    L2_se = sd(Lambda_2, na.rm = TRUE) / sqrt(sum(!is.na(Lambda_2))),
    .groups = "drop"
  )

fwd_summary <- fr_summary %>% filter(Phase == "Forward")
rev_summary <- fr_summary %>% filter(Phase == "Reverse")

# Pre-collapse summary
pc_summary <- pre_collapse %>%
  group_by(Mu) %>%
  summarise(
    N_mean = mean(N), N_se = sd(N) / sqrt(n()),
    DKL_mean = mean(METE_DKL, na.rm = TRUE),
    DKL_se = sd(METE_DKL, na.rm = TRUE) / sqrt(sum(!is.na(METE_DKL))),
    L2_mean = mean(Lambda_2, na.rm = TRUE),
    L2_se = sd(Lambda_2, na.rm = TRUE) / sqrt(sum(!is.na(Lambda_2))),
    N_ratio_mean = mean(N_ratio),
    .groups = "drop"
  )

# ============================================================================
# FIGURE 1: ADIABATIC HYSTERESIS LOOP
# ============================================================================

cat("\n  Generating Figure 1: Hysteresis Loop...\n")

# Panel A: N vs μ — individual replicates + mean ± SE
p1a <- ggplot() +
  # Individual replicate traces (thin, transparent)
  geom_line(data = fwd, aes(x = Mu, y = N, group = Replicate),
            colour = col_fwd, alpha = 0.12, linewidth = 0.3) +
  geom_line(data = rev, aes(x = Mu, y = N, group = Replicate),
            colour = col_rev, alpha = 0.12, linewidth = 0.3) +
  # Mean ± SE ribbon
  geom_ribbon(data = fwd_summary,
              aes(x = Mu, ymin = N_mean - N_se, ymax = N_mean + N_se),
              fill = col_fwd, alpha = 0.25) +
  geom_ribbon(data = rev_summary,
              aes(x = Mu, ymin = N_mean - N_se, ymax = N_mean + N_se),
              fill = col_rev, alpha = 0.25) +
  # Mean lines
  geom_line(data = fwd_summary, aes(x = Mu, y = N_mean),
            colour = col_fwd, linewidth = 1.0) +
  geom_line(data = rev_summary, aes(x = Mu, y = N_mean),
            colour = col_rev, linewidth = 1.0) +
  # Arrows indicating direction
  annotate("text", x = max(fwd_summary$Mu) * 0.3,
           y = max(fwd_summary$N_mean) * 0.92,
           label = "Forward →", colour = col_fwd, size = 2.5, fontface = "bold") +
  annotate("text", x = max(rev_summary$Mu) * 0.6,
           y = max(rev_summary$N_mean) * 0.45,
           label = "← Reverse", colour = col_rev, size = 2.5, fontface = "bold") +
  labs(x = expression(paste("Abiotic stress, ", mu)),
       y = expression(paste("Population size, ", italic(N))),
       title = "A",
       subtitle = "Population hysteresis") +
  theme_pnas()

# Panel B: S vs μ
p1b <- ggplot() +
  geom_line(data = fwd, aes(x = Mu, y = S, group = Replicate),
            colour = col_fwd, alpha = 0.12, linewidth = 0.3) +
  geom_line(data = rev, aes(x = Mu, y = S, group = Replicate),
            colour = col_rev, alpha = 0.12, linewidth = 0.3) +
  geom_ribbon(data = fwd_summary,
              aes(x = Mu, ymin = S_mean - S_se, ymax = S_mean + S_se),
              fill = col_fwd, alpha = 0.25) +
  geom_ribbon(data = rev_summary,
              aes(x = Mu, ymin = S_mean - S_se, ymax = S_mean + S_se),
              fill = col_rev, alpha = 0.25) +
  geom_line(data = fwd_summary, aes(x = Mu, y = S_mean),
            colour = col_fwd, linewidth = 1.0) +
  geom_line(data = rev_summary, aes(x = Mu, y = S_mean),
            colour = col_rev, linewidth = 1.0) +
  labs(x = expression(paste("Abiotic stress, ", mu)),
       y = expression(paste("Species richness, ", italic(S))),
       title = "B",
       subtitle = "Diversity hysteresis") +
  theme_pnas()

# Panel C: N_ratio vs μ showing collapse threshold
p1c <- ggplot() +
  geom_hline(yintercept = 0.25, linetype = "dashed", colour = col_warn,
             linewidth = 0.5) +
  geom_hline(yintercept = 0.80, linetype = "dotted", colour = "grey50",
             linewidth = 0.4) +
  geom_line(data = fwd, aes(x = Mu, y = N_ratio, group = Replicate),
            colour = col_fwd, alpha = 0.12, linewidth = 0.3) +
  geom_line(data = rev, aes(x = Mu, y = N_ratio, group = Replicate),
            colour = col_rev, alpha = 0.12, linewidth = 0.3) +
  annotate("text", x = 0.2, y = 0.27, label = "Collapse threshold (25%)",
           colour = col_warn, size = 2, hjust = 0, fontface = "italic") +
  annotate("text", x = 0.2, y = 0.82,
           label = "Pre-collapse regime (80%)",
           colour = "grey50", size = 2, hjust = 0, fontface = "italic") +
  labs(x = expression(paste("Abiotic stress, ", mu)),
       y = expression(italic(N) / italic(N)[0]),
       title = "C",
       subtitle = "Normalised population decline") +
  scale_y_continuous(limits = c(0, 1.2), breaks = seq(0, 1.2, 0.2)) +
  theme_pnas()

# Panel D: Difference (Forward - Reverse) at matched μ
# Compute per-replicate difference at matched μ
matched <- inner_join(
  fwd %>% select(Replicate, Mu, N_fwd = N),
  rev %>% select(Replicate, Mu, N_rev = N),
  by = c("Replicate", "Mu")
) %>%
  mutate(Delta_N = N_fwd - N_rev)

delta_summary <- matched %>%
  group_by(Mu) %>%
  summarise(
    Delta_mean = mean(Delta_N),
    Delta_lo = quantile(Delta_N, 0.025),
    Delta_hi = quantile(Delta_N, 0.975),
    .groups = "drop"
  )

p1d <- ggplot(delta_summary, aes(x = Mu)) +
  geom_hline(yintercept = 0, linetype = "solid", colour = "grey50",
             linewidth = 0.3) +
  geom_ribbon(aes(ymin = Delta_lo, ymax = Delta_hi),
              fill = "#9C27B0", alpha = 0.2) +
  geom_line(aes(y = Delta_mean), colour = "#9C27B0", linewidth = 0.8) +
  labs(x = expression(paste("Abiotic stress, ", mu)),
       y = expression(Delta*italic(N) ~ "(Forward - Reverse)"),
       title = "D",
       subtitle = "Hysteresis gap (95% CI)") +
  theme_pnas()

fig1 <- (p1a | p1b) / (p1c | p1d) +
  plot_annotation(
    title = "Figure 1. Adiabatic hysteresis in a Tangled Nature ecosystem",
    subtitle = sprintf("%d replicate ecosystems (L=20, R=100) driven through forward stress escalation and reverse recovery protocol",
                       length(unique(d$Replicate))),
    theme = theme_pnas(base_size = 9) +
      theme(plot.title = element_text(size = 11, face = "bold"),
            plot.subtitle = element_text(size = 8, colour = "grey40"))
  )

ggsave("Figure_1_Hysteresis.pdf", fig1,
       width = 180, height = 150, units = "mm", dpi = 600)
ggsave("Figure_1_Hysteresis.png", fig1,
       width = 180, height = 150, units = "mm", dpi = 300)
cat("  ✓ Figure 1 saved\n")

# ============================================================================
# FIGURE 2: EARLY WARNING SIGNALS & TOPOLOGICAL MASKING
# ============================================================================

cat("  Generating Figure 2: Early Warning Signals...\n")

# Consistent x-axis range for panels A and B
mu_range <- range(fwd$Mu, na.rm = TRUE)

# Panel A: D_KL vs μ (forward only), pre-collapse highlighted
p2a <- ggplot() +
  # Full forward trace
  geom_point(data = fwd %>% filter(!is.na(METE_DKL)),
             aes(x = Mu, y = METE_DKL),
             colour = "grey70", size = 0.3, alpha = 0.3) +
  # Pre-collapse highlighted
  geom_point(data = pre_collapse %>% filter(!is.na(METE_DKL)),
             aes(x = Mu, y = METE_DKL),
             colour = col_fwd, size = 0.4, alpha = 0.4) +
  # LOESS smooth on pre-collapse
  geom_smooth(data = pre_collapse %>% filter(!is.na(METE_DKL)),
              aes(x = Mu, y = METE_DKL),
              method = "loess", span = 0.5, se = TRUE,
              colour = col_fwd, fill = col_fwd, alpha = 0.2,
              linewidth = 0.8) +
  # Vertical line at max pre-collapse μ
  geom_vline(xintercept = max(pre_collapse$Mu), linetype = "dashed",
             colour = col_warn, linewidth = 0.4) +
  annotate("text", x = max(pre_collapse$Mu) + 0.1,
           y = max(pre_collapse$METE_DKL, na.rm = TRUE) * 0.95,
           label = "Onset\nzone", colour = col_warn, size = 2,
           hjust = 0, fontface = "italic") +
  labs(x = expression(paste("Abiotic stress, ", mu)),
       y = expression(italic(D)[KL] ~ "(METE)"),
       title = "A",
       subtitle = "Structural divergence from METE") +
  coord_cartesian(xlim = mu_range) +
  theme_pnas()

# Panel B: λ₂ vs μ (forward only)
p2b <- ggplot() +
  geom_point(data = fwd %>% filter(!is.na(Lambda_2)),
             aes(x = Mu, y = Lambda_2),
             colour = "grey70", size = 0.3, alpha = 0.3) +
  geom_point(data = pre_collapse %>% filter(!is.na(Lambda_2)),
             aes(x = Mu, y = Lambda_2),
             colour = col_rev, size = 0.4, alpha = 0.4) +
  geom_smooth(data = pre_collapse %>% filter(!is.na(Lambda_2)),
              aes(x = Mu, y = Lambda_2),
              method = "loess", span = 0.5, se = TRUE,
              colour = col_rev, fill = col_rev, alpha = 0.2,
              linewidth = 0.8) +
  geom_vline(xintercept = max(pre_collapse$Mu), linetype = "dashed",
             colour = col_warn, linewidth = 0.4) +
  labs(x = expression(paste("Abiotic stress, ", mu)),
       y = expression(lambda[2] ~ "(algebraic connectivity)"),
       title = "B",
       subtitle = "Interaction network cohesion") +
  coord_cartesian(xlim = mu_range) +
  theme_pnas()

# Panel C: D_KL vs N_ratio (structural warping as population declines)
p2c <- ggplot() +
  geom_point(data = fwd %>% filter(!is.na(METE_DKL)),
             aes(x = N_ratio, y = METE_DKL),
             colour = col_fwd, size = 0.4, alpha = 0.3) +
  geom_smooth(data = fwd %>% filter(!is.na(METE_DKL)),
              aes(x = N_ratio, y = METE_DKL),
              method = "loess", span = 0.6, se = TRUE,
              colour = col_fwd, fill = col_fwd, alpha = 0.2,
              linewidth = 0.8) +
  geom_vline(xintercept = 0.80, linetype = "dotted", colour = "grey50",
             linewidth = 0.4) +
  geom_vline(xintercept = 0.25, linetype = "dashed", colour = col_warn,
             linewidth = 0.4) +
  scale_x_reverse() +
  labs(x = expression(italic(N) / italic(N)[0] ~ "(decreasing →)"),
       y = expression(italic(D)[KL] ~ "(METE)"),
       title = "C",
       subtitle = "Structural divergence vs population state") +
  theme_pnas()

# Panel D: λ₂ vs N_ratio
p2d <- ggplot() +
  geom_point(data = fwd %>% filter(!is.na(Lambda_2)),
             aes(x = N_ratio, y = Lambda_2),
             colour = col_rev, size = 0.4, alpha = 0.3) +
  geom_smooth(data = fwd %>% filter(!is.na(Lambda_2)),
              aes(x = N_ratio, y = Lambda_2),
              method = "loess", span = 0.6, se = TRUE,
              colour = col_rev, fill = col_rev, alpha = 0.2,
              linewidth = 0.8) +
  geom_vline(xintercept = 0.80, linetype = "dotted", colour = "grey50",
             linewidth = 0.4) +
  geom_vline(xintercept = 0.25, linetype = "dashed", colour = col_warn,
             linewidth = 0.4) +
  scale_x_reverse() +
  labs(x = expression(italic(N) / italic(N)[0] ~ "(decreasing →)"),
       y = expression(lambda[2] ~ "(algebraic connectivity)"),
       title = "D",
       subtitle = "Topological fragmentation vs population state") +
  theme_pnas()

fig2 <- (p2a | p2b) / (p2c | p2d) +
  plot_annotation(
    title = "Figure 2. Early warning signals and topological masking",
    subtitle = paste0("Forward press only. Pre-collapse regime (N > 80% baseline) ",
                      "highlighted. Grey points = post-onset data."),
    theme = theme_pnas(base_size = 9) +
      theme(plot.title = element_text(size = 11, face = "bold"),
            plot.subtitle = element_text(size = 8, colour = "grey40"))
  )

ggsave("Figure_2_EWS.pdf", fig2,
       width = 180, height = 150, units = "mm", dpi = 600)
ggsave("Figure_2_EWS.png", fig2,
       width = 180, height = 150, units = "mm", dpi = 300)
cat("  ✓ Figure 2 saved\n")

# ============================================================================
# FIGURE 3: MECHANISTIC SUMMARY — THREE-PANEL SYNTHESIS
# ============================================================================

cat("  Generating Figure 3: Mechanistic Summary...\n")

# Panel A: Phase portrait — N Forward vs N Reverse at matched μ
p3a <- ggplot(matched, aes(x = N_fwd, y = N_rev)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed",
              colour = "grey60", linewidth = 0.3) +
  geom_point(aes(colour = Mu), size = 0.6, alpha = 0.5) +
  scale_colour_viridis_c(option = "inferno", name = expression(mu),
                         guide = guide_colourbar(barwidth = 4,
                                                 barheight = 0.4)) +
  labs(x = expression(italic(N)[Forward]),
       y = expression(italic(N)[Reverse]),
       title = "A",
       subtitle = "State comparison at matched stress") +
  coord_equal() +
  theme_pnas()

# Panel B: Dual-axis concept — D_KL and N on same μ axis (forward only)
# Using faceted approach for clarity
dual_data <- fwd %>%
  filter(!is.na(METE_DKL)) %>%
  select(Replicate, Mu, N, METE_DKL) %>%
  group_by(Mu) %>%
  summarise(N_mean = mean(N), DKL_mean = mean(METE_DKL), .groups = "drop") %>%
  pivot_longer(cols = c(N_mean, DKL_mean),
               names_to = "Metric", values_to = "Value") %>%
  mutate(Metric = case_when(
    Metric == "N_mean" ~ "Population (N)",
    Metric == "DKL_mean" ~ "METE D[KL]"
  ))

p3b <- ggplot(dual_data, aes(x = Mu, y = Value)) +
  geom_line(colour = col_fwd, linewidth = 0.6) +
  geom_point(colour = col_fwd, size = 0.5, alpha = 0.6) +
  facet_wrap(~ Metric, scales = "free_y", ncol = 1,
             strip.position = "left") +
  labs(x = expression(paste("Abiotic stress, ", mu)),
       y = NULL,
       title = "B",
       subtitle = "Macroscopic vs structural response") +
  theme_pnas() +
  theme(strip.placement = "outside",
        strip.text.y.left = element_text(angle = 90, size = rel(0.8)))

# Panel C: Replicate-wise collapse μ distribution
collapse_mu <- fwd %>%
  group_by(Replicate) %>%
  filter(N_ratio <= 0.25) %>%
  summarise(Mu_collapse = min(Mu), .groups = "drop")

# Replicates that never reached collapse
n_no_collapse <- length(setdiff(levels(d$Replicate),
                                 as.character(collapse_mu$Replicate)))

p3c <- ggplot(collapse_mu, aes(x = Mu_collapse)) +
  geom_histogram(bins = 15, fill = col_fwd, colour = "white",
                 alpha = 0.7, linewidth = 0.3) +
  geom_vline(xintercept = mean(collapse_mu$Mu_collapse),
             linetype = "dashed", colour = "grey30", linewidth = 0.5) +
  annotate("text",
           x = mean(collapse_mu$Mu_collapse) + 0.1,
           y = Inf, vjust = 2,
           label = sprintf("Mean = %.1f", mean(collapse_mu$Mu_collapse)),
           colour = "grey30", size = 2.5, fontface = "italic") +
  labs(x = expression(mu[collapse]),
       y = "Number of replicates",
       title = "C",
       subtitle = "Distribution of collapse stress thresholds") +
  theme_pnas()

fig3 <- (p3a | p3b | p3c) +
  plot_annotation(
    title = "Figure 3. Mechanistic synthesis of adiabatic ecosystem response",
    subtitle = paste0("(A) State-dependent trajectories reveal path dependence. ",
                      "(B) Population decline precedes structural divergence. ",
                      "(C) Stochastic variation in collapse threshold."),
    theme = theme_pnas(base_size = 9) +
      theme(plot.title = element_text(size = 11, face = "bold"),
            plot.subtitle = element_text(size = 7.5, colour = "grey40"))
  )

ggsave("Figure_3_Synthesis.pdf", fig3,
       width = 183, height = 80, units = "mm", dpi = 600)
ggsave("Figure_3_Synthesis.png", fig3,
       width = 183, height = 80, units = "mm", dpi = 300)
cat("  ✓ Figure 3 saved\n")

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

cat("\n══════════════════════════════════════════════════════════\n")
cat("  FIGURE GENERATION COMPLETE\n")
cat("══════════════════════════════════════════════════════════\n")
cat("  Figure_1_Hysteresis.pdf / .png  — 4-panel hysteresis loop\n")
cat("  Figure_2_EWS.pdf / .png         — 4-panel EWS & topology\n")
cat("  Figure_3_Synthesis.pdf / .png   — 3-panel mechanistic summary\n")
cat("══════════════════════════════════════════════════════════\n")
