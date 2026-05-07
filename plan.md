# Plan: Week-1 End-to-End Pipeline

## Goal

Build the smallest possible pipeline that runs end-to-end and produces a real evaluation number. No player features yet. No XGBoost. Just a skeleton that proves the data joins work and gives an honest baseline to beat.

**Success criterion:** a single log-loss number for logistic regression vs bookmaker baseline on a temporal holdout, reproducible from a clean checkout.

---

## Scope

**In scope:**
- Load Premier League 2022–23 from Football-Data.co.uk (results + bookmaker odds)
- Load PL 2022–23 match-level team stats from FBref (xG for/against)
- Build canonical team-name mapping (Football-Data ↔ FBref name conventions)
- Generate a `match_id` from `(date, home_team, away_team)` as the join key
- One rolling feature: 5-game team xG differential (home minus away, lag-safe)
- Logistic regression, 3-class (H/D/A)
- Temporal holdout: last 20% of matches by date
- Bookmaker baseline: overround-normalized implied probabilities
- Metrics: log-loss (primary), Brier score (secondary)

**Not in scope (deferred to next week):**
- Player-level features
- FBref per-player per-match data (not needed for team-level baseline)
- Understat xG timelines
- XGBoost / LightGBM
- Calibration (Platt scaling, isotonic regression) — add in week 2
- Multiple leagues or seasons

---

## File Structure

```
player-match-prediction/
├── data/
│   └── raw/          # downloaded CSVs, never committed
├── src/
│   ├── load.py       # load_football_data(), load_fbref_team()
│   ├── join.py       # team_name_map, build_match_id(), join_sources()
│   ├── features.py   # rolling_xg_diff() — lag-safe, feature_cutoff enforced
│   ├── evaluate.py   # bookmaker_baseline(), log_loss_brier()
│   └── pipeline.py   # main() — orchestrates load → join → features → model → eval
├── notebooks/
│   └── 01_eda.ipynb  # optional exploratory work
├── requirements.txt
└── README.md
```

---

## Implementation Steps

### Step 1: Data download helpers (`src/load.py`)

- `load_football_data(season, league)` — reads `E0.csv` from Football-Data.co.uk
  - Output: DataFrame with columns `date, home_team, away_team, result, B365H, B365D, B365A`
- `load_fbref_team(season, league)` — reads FBref match logs (xG for/against per team)
  - Output: DataFrame with columns `date, team, xg_for, xg_against`
  - FBref is scraped via `soccerdata` library (handles rate limiting + caching)

### Step 2: Join logic (`src/join.py`)

Team name mapping is the hardest part. Hardcode it as a dict for PL 2022–23. Every mismatch surfaces here and must be fixed before the join.

```python
TEAM_NAME_MAP = {
    "Man United": "Manchester Utd",
    "Man City": "Manchester City",
    # ... full mapping
}
```

`build_match_id(date, home, away)` → `"2022-08-06_arsenal_chelsea"` (lowercased, sorted)

`join_sources(fd_df, fbref_df)` → match-level DataFrame with both result and xG.

### Step 3: Feature engineering (`src/features.py`)

`rolling_xg_diff(df, window=5)` — for each match, compute rolling mean of (xg_for - xg_against) over the 5 preceding matches for each team.

**Leakage rule, enforced by assertion:** features for match M use only data from matches with `date < match_date`. The function must raise if this is violated.

### Step 4: Bookmaker baseline (`src/evaluate.py`)

Convert decimal odds (B365H, B365D, B365A) to calibrated probabilities:

```python
def overround_normalize(h, d, a):
    inv = [1/h, 1/d, 1/a]
    total = sum(inv)
    return inv[0]/total, inv[1]/total, inv[2]/total
```

This is the baseline to beat.

### Step 5: Model + evaluation (`src/pipeline.py`)

1. Temporal split: sort by date, last 20% is test set
2. Fit `sklearn.linear_model.LogisticRegression` on train
3. Compute log-loss and Brier score for model vs bookmaker baseline on test
4. Print comparison table

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Team name mismatches break the join | Hardcode full PL 2022–23 mapping; assert zero unmatched rows |
| Lineup leakage (match M features include match M data) | Assert `feature_date < match_date` inside `rolling_xg_diff` |
| Random k-fold gives inflated metrics | Use only temporal split — no k-fold anywhere in this pipeline |
| FBref scraping blocked / rate-limited | Use `soccerdata` library with local cache; one-time download |
| Bookmaker baseline comparison is unfair | Normalize overround before computing log-loss baseline |

---

## Expected Output

```
=== PL 2022-23 Temporal Holdout (last 20%) ===
Metric          Logistic Reg    Bookmaker
---             ---             ---
Log-loss        X.XXX           X.XXX
Brier score     X.XXX           X.XXX
Accuracy        XX.X%           XX.X%

Bookmaker beats model on log-loss: [yes/no]
```

If bookmaker log-loss < model log-loss (likely), that's the honest baseline the rest of the project tries to chip away at.

---

## Python Dependencies

```
pandas
scikit-learn
soccerdata        # FBref + other sources, handles caching
requests
jupyter           # optional, for EDA notebook
```


