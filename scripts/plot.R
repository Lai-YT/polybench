#!/usr/bin/env Rscript
library(tidyverse)
library(argparse)

parser <- ArgumentParser()
parser$add_argument("dir", metavar = "DIR", help = "The directory of the data")
parser$add_argument("output", metavar = "OUTPUT", help = "The output file")
args <- parser$parse_args()

baseline <- read_csv(
  file.path(args$dir, "x86-64.csv"),
  col_names = c("name"),
  show_col_types = FALSE,
)
baseline <- baseline |>
  rowwise() |>
  mutate(mean = mean(c_across(2:6)), variance = var(c_across(2:6))) |>
  ungroup() |>
  mutate(type = "baseline")

ppcg <- read_csv(
  file.path(args$dir, "x86-64-ppcg.csv"),
  col_names = c("name"),
  show_col_types = FALSE,
)
ppcg <- ppcg |>
  rowwise() |>
  mutate(mean = mean(c_across(2:6)), variance = var(c_across(2:6))) |>
  ungroup() |>
  mutate(type = "ppcg")

data <- bind_rows(baseline, ppcg) |>
  # For the names, split with "/" and take the first part.
  mutate(name = str_split(name, "/") |> map_chr(first))

speedup <- data |>
  # Calculate the speedup.
  filter(type == "baseline") |>
  left_join(data |> filter(type == "ppcg"), by = "name") |>
  mutate(speedup = mean.x / mean.y) |>
  filter(!is.na(speedup)) |>
  select(name, speedup)

# Get the geomean of all the benchmark speedups.
geomean <- speedup |>
  summarise(geomean = exp(mean(log(speedup)))) |>
  pull(geomean)

# Plot the speedup.
p <- ggplot(data = speedup,
            mapping = aes(x = speedup, y = fct_reorder(name, speedup), )) + geom_col() +
  # Have the right side be 5x more than the greatest speedup to avoid cutting off the label.
  scale_x_log10(
    limits = c(NA, max(speedup$speedup) * 5),
  ) +
  # show label on each bar.
  labs(
    x = "Speedup",
    y = "Benchmark",
    title = "Speedup of PPCG over Baseline",
    subtitle = sprintf("Geomean: %.2f", geomean)
  ) +
  geom_label(aes(label = round(speedup, 2)), hjust = -0.1)

ggsave(args$output, p, scale = 2)
