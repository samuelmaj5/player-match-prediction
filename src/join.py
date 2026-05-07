"""Team-name normalization, match_id construction, and source joining."""

from __future__ import annotations

import pandas as pd

# Map: Football-Data.co.uk name -> Understat name (as returned by soccerdata).
TEAM_NAME_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Nott'm Forest": "Nottingham Forest",
    "Southampton": "Southampton",
    "Tottenham": "Tottenham",
    "West Ham": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
}


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("'", "")


def build_match_id(date, home: str, away: str) -> str:
    d = pd.to_datetime(date).strftime("%Y-%m-%d")
    return f"{d}_{_slug(home)}_{_slug(away)}"


def join_sources(fd_df: pd.DataFrame, fbref_df: pd.DataFrame) -> pd.DataFrame:
    """Join Football-Data results/odds with FBref team-level xG.

    Asserts: every Football-Data row finds an FBref match. Surface mismatches loudly.
    """
    fd = fd_df.copy()
    fd["home_team_canonical"] = fd["home_team"].map(TEAM_NAME_MAP)
    fd["away_team_canonical"] = fd["away_team"].map(TEAM_NAME_MAP)

    missing = fd[fd["home_team_canonical"].isna() | fd["away_team_canonical"].isna()]
    if not missing.empty:
        unmapped = sorted(set(missing["home_team"]).union(missing["away_team"]) - set(TEAM_NAME_MAP))
        raise ValueError(f"Unmapped Football-Data teams: {unmapped}")

    fd["match_id"] = [
        build_match_id(d, h, a)
        for d, h, a in zip(fd["date"], fd["home_team_canonical"], fd["away_team_canonical"])
    ]

    fb = fbref_df.copy()
    fb["date"] = pd.to_datetime(fb["date"])

    # Pivot FBref long -> wide on (date, home, away). Need opponent column to pair rows.
    # The loader already gives us per-team rows; we reconstruct by matching team to home/away.
    home_xg = fb.rename(columns={"team": "home_team_canonical",
                                 "xg_for": "home_xg", "xg_against": "away_xg_check"})
    home_xg = home_xg[["date", "home_team_canonical", "home_xg", "away_xg_check"]]

    merged = fd.merge(
        home_xg,
        on=["date", "home_team_canonical"],
        how="left",
    )

    # Cross-check: pull away team xG_for to confirm pairing
    away_xg = fb.rename(columns={"team": "away_team_canonical", "xg_for": "away_xg"})
    away_xg = away_xg[["date", "away_team_canonical", "away_xg"]]
    merged = merged.merge(away_xg, on=["date", "away_team_canonical"], how="left")

    unmatched = merged[merged["home_xg"].isna() | merged["away_xg"].isna()]
    if not unmatched.empty:
        raise ValueError(
            f"{len(unmatched)} matches did not join to FBref xG. "
            f"First few: {unmatched[['date', 'home_team', 'away_team']].head().to_dict('records')}"
        )

    return merged[[
        "match_id", "date", "home_team_canonical", "away_team_canonical",
        "result", "B365H", "B365D", "B365A", "home_xg", "away_xg",
    ]].rename(columns={"home_team_canonical": "home_team", "away_team_canonical": "away_team"})
