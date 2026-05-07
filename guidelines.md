Project: Predicting Match Outcomes from Player-Level Stats in Soccer

Problem statement Can individual player performance metrics (rather than team-level aggregates) predict match outcomes better than market odds or Elo-based baselines?
Data

FBref (free, very rich) — player-level stats per match: xG, progressive passes, pressures, defensive actions, etc.
Understat — xG timelines per match

Football-Data.co.uk — historical results + bookmaker odds (your baseline)
Feature engineering — this is where the ML discussion gets rich:
Aggregate player stats weighted by minutes played
Form windows (last 5 games rolling average)
Squad depth features (starter vs bench quality gap)
Home/away splits per player
Models
Baseline: bookmaker implied probability (hard to beat, honest reference)
Logistic regression on team aggregates (simple baseline)
Gradient boosting (XGBoost/LightGBM) on player-level features
Optional: compare with an Elo-only model
What makes it novel The novelty is the granularity — most public models use team stats. You'd be explicitly testing whether disaggregating to player level adds predictive power, which is a clean, falsifiable hypothesis.
ML discussion depth
Feature selection and multicollinearity (players on the same team have correlated stats)
Class imbalance (draws are hard to predict)
Calibration of probabilities, not just accuracy
Overfitting risk with many player features → regularization discussion
Train/test split has to respect time ordering → temporal cross-validation

