# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Goal

Predict soccer match outcomes using player-level statistics (xG, progressive passes, pressures, defensive actions) rather than team-level aggregates. The central hypothesis is that disaggregating to player level adds predictive power over bookmaker odds or Elo-based baselines.

## Data Sources

- **FBref** — player-level per-match stats (primary feature source)
- **Understat** — xG timelines per match
- **Football-Data.co.uk** — historical results + bookmaker odds (baseline reference)

## Architecture

The project is in early scaffolding. When building out, the expected pipeline is:

1. **Data ingestion** — scrape/fetch from FBref, Understat, Football-Data.co.uk
2. **Feature engineering** — player stats aggregated by minutes played, rolling 5-game form windows, squad depth (starter vs bench gap), home/away splits per player
3. **Modeling** — baseline (bookmaker implied prob), logistic regression on team aggregates, gradient boosting (XGBoost/LightGBM) on player features, optional Elo comparison
4. **Evaluation** — temporal cross-validation (train/test split must respect time ordering), calibration metrics (not just accuracy), class imbalance handling for draws

## Key ML Constraints

- **Temporal cross-validation is required** — never shuffle across time boundaries
- **Probability calibration matters** — evaluate with Brier score and log-loss, not just accuracy
- **Multicollinearity** — players on the same team have correlated stats; use regularization or explicit feature selection
- **Class imbalance** — draws are underrepresented; address with class weights or resampling
