"""Data loaders for Football-Data.co.uk and FBref (via soccerdata)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

FOOTBALL_DATA_LEAGUE_CODES = {
    "ENG-Premier League": "E0",
    "ENG-Championship": "E1",
    "ESP-La Liga": "SP1",
    "ITA-Serie A": "I1",
    "GER-Bundesliga": "D1",
    "FRA-Ligue 1": "F1",
}


def _season_code(season: str) -> str:
    # "2022-2023" -> "2223"
    start, end = season.split("-")
    return start[-2:] + end[-2:]


def load_football_data(season: str, league: str = "ENG-Premier League") -> pd.DataFrame:
    """Download (with on-disk cache) results + B365 odds from football-data.co.uk.

    Returns: date, home_team, away_team, result, B365H, B365D, B365A.
    """
    code = FOOTBALL_DATA_LEAGUE_CODES[league]
    season_code = _season_code(season)
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/{code}.csv"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"footballdata_{league.replace(' ', '_')}_{season}.csv"
    if not cache_path.exists():
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)

    df = pd.read_csv(cache_path, encoding="latin-1")
    df = df.rename(
        columns={
            "Date": "date",
            "HomeTeam": "home_team",
            "AwayTeam": "away_team",
            "FTR": "result",
        }
    )
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)
    return df[["date", "home_team", "away_team", "result", "B365H", "B365D", "B365A"]]


def load_team_xg(season: str, league: str = "ENG-Premier League") -> pd.DataFrame:
    """Per-match team xG via Understat (soccerdata).

    FBref's schedule page no longer exposes xG/xGA columns through soccerdata,
    so Understat is used for the xG signal. Returns long format:
    date, team, xg_for, xg_against.
    """
    import soccerdata as sd

    us = sd.Understat(leagues=league, seasons=season, data_dir=RAW_DIR / "understat")
    sched = us.read_schedule().reset_index()
    sched["date"] = pd.to_datetime(sched["date"]).dt.normalize()
    sched = sched[sched["is_result"]].copy()

    home = sched[["date", "home_team", "home_xg", "away_xg"]].rename(
        columns={"home_team": "team", "home_xg": "xg_for", "away_xg": "xg_against"}
    )
    away = sched[["date", "away_team", "away_xg", "home_xg"]].rename(
        columns={"away_team": "team", "away_xg": "xg_for", "home_xg": "xg_against"}
    )
    out = pd.concat([home, away], ignore_index=True)
    out = out.dropna(subset=["xg_for", "xg_against"])
    return (
        out[["date", "team", "xg_for", "xg_against"]]
        .sort_values(["date", "team"])
        .reset_index(drop=True)
    )
