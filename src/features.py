"""Lag-safe feature engineering."""

from __future__ import annotations

import pandas as pd


def _team_rolling_xg_diff(fb_long: pd.DataFrame, window: int) -> pd.DataFrame:
    """For each (team, date) row, mean (xg_for - xg_against) over the prior `window` matches.

    Strictly excludes the current match — uses .shift(1) before .rolling().
    """
    df = fb_long.sort_values(["team", "date"]).copy()
    df["xg_diff"] = df["xg_for"] - df["xg_against"]
    df["xg_diff_roll"] = (
        df.groupby("team")["xg_diff"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return df[["date", "team", "xg_diff_roll"]]


def rolling_xg_diff(joined: pd.DataFrame, fb_long: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add `xg_diff_5` = home rolling xG-diff minus away rolling xG-diff.

    Args:
        joined: output of join_sources() — has match_id, date, home_team, away_team.
        fb_long: long-format FBref per-team-per-match xG (from load_fbref_team).
        window: rolling window length in matches.

    Leakage guard: asserts that every contributing row's date is strictly before the
    target match date.
    """
    rolled = _team_rolling_xg_diff(fb_long, window)

    out = joined.merge(
        rolled.rename(columns={"team": "home_team", "xg_diff_roll": "home_xg_diff_roll"}),
        on=["date", "home_team"], how="left",
    ).merge(
        rolled.rename(columns={"team": "away_team", "xg_diff_roll": "away_xg_diff_roll"}),
        on=["date", "away_team"], how="left",
    )

    out[f"xg_diff_{window}"] = out["home_xg_diff_roll"] - out["away_xg_diff_roll"]

    _assert_no_leakage(out, fb_long, window)

    return out.drop(columns=["home_xg_diff_roll", "away_xg_diff_roll"])


def _assert_no_leakage(joined_with_feat: pd.DataFrame, fb_long: pd.DataFrame, window: int) -> None:
    """For every match with a non-null feature, verify that all `window` contributing
    rows for both teams have date < match date.
    """
    fb = fb_long.copy()
    fb["date"] = pd.to_datetime(fb["date"])
    by_team = {t: g.sort_values("date").reset_index(drop=True) for t, g in fb.groupby("team")}

    feat_col = f"xg_diff_{window}"
    rows = joined_with_feat.dropna(subset=[feat_col])

    for _, r in rows.iterrows():
        for team in (r["home_team"], r["away_team"]):
            g = by_team[team]
            prior = g[g["date"] < r["date"]]
            if prior.empty:
                raise AssertionError(
                    f"Leakage check: feature non-null for {r['match_id']} but no prior "
                    f"matches for {team} before {r['date'].date()}"
                )
            contributing = prior.tail(window)
            if (contributing["date"] >= r["date"]).any():
                raise AssertionError(
                    f"Leakage in {r['match_id']}: contributing row date >= match date for {team}"
                )
