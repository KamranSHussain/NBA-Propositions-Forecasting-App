"""Data fetching and feature engineering for NBA player prop modeling.

This module provides a single public pipeline function, ``get_nba_data``, that
returns:
1. Historical training data with leakage-safe shifted rolling features.
2. Current player feature state for inference.
3. Current team/opponent context feature state for inference.
"""

from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2026
ROLLING_WINDOWS: tuple[int, ...] = (5, 10)

PLAYER_STATS_TO_ROLL: tuple[str, ...] = (
    "PTS",
    "MIN",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "AST",
    "REB",
    "TOV",
    "STL",
    "PF",
)

TEAM_OFFENSE_COLS: tuple[str, ...] = ("PTS", "AST", "FG3A", "FGA")
TEAM_DEFENSE_COLS: tuple[str, ...] = ("opponentScore", "STL", "BLK", "DREB")

PLAYER_INFERENCE_COLS: tuple[str, ...] = (
    "Rolling_5G_Games_Played",
    "Rolling_10G_Games_Played",
    "Rolling_5G_MIN",
    "Rolling_10G_MIN",
    "Rolling_5G_PTS",
    "Rolling_10G_PTS",
    "Rolling_5G_FGA",
    "Rolling_10G_FGA",
    "Rolling_5G_FG_PCT",
    "Rolling_10G_FG_PCT",
    "Rolling_5G_FG3A",
    "Rolling_10G_FG3A",
    "Rolling_5G_FTA",
    "Rolling_10G_FTA",
    "Rolling_5G_AST",
    "Rolling_10G_AST",
    "Rolling_5G_REB",
    "Rolling_10G_REB",
    "Rolling_5G_TOV",
    "Rolling_10G_TOV",
)

TEAM_INFERENCE_COLS: tuple[str, ...] = (
    "Team_Rolling_5G_PTS",
    "Team_Rolling_10G_PTS",
    "Team_Rolling_5G_FGA",
    "Team_Rolling_10G_FGA",
    "Opp_Rolling_5G_opponentScore",
    "Opp_Rolling_10G_opponentScore",
    "Opp_Rolling_5G_STL",
    "Opp_Rolling_10G_STL",
    "Opp_Rolling_5G_BLK",
    "Opp_Rolling_10G_BLK",
)


def _season_strings(start_year: int, end_year: int) -> list[str]:
    """Build season labels like ``2015-16`` from year bounds."""
    return [f"{year}-{str(year + 1)[-2:]}" for year in range(start_year, end_year)]


def fetch_nba_api_data(season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch regular season + playoff game logs for one season."""
    print(f"  -> Fetching {season}...")

    rs_players = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    rs_teams = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="T",
        season_type_all_star="Regular Season",
    ).get_data_frames()[0]
    rs_players["is_playoff"] = 0

    time.sleep(2)

    po_players = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="P",
        season_type_all_star="Playoffs",
    ).get_data_frames()[0]
    po_teams = leaguegamelog.LeagueGameLog(
        season=season,
        player_or_team_abbreviation="T",
        season_type_all_star="Playoffs",
    ).get_data_frames()[0]
    po_players["is_playoff"] = 1

    time.sleep(2)

    players_df = pd.concat([rs_players, po_players], ignore_index=True)
    teams_df = pd.concat([rs_teams, po_teams], ignore_index=True)
    return players_df, teams_df


def fetch_multiple_seasons(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch and combine multi-season player/team logs from NBA API."""
    seasons = _season_strings(start_year=start_year, end_year=end_year)
    all_players_data: list[pd.DataFrame] = []
    all_teams_data: list[pd.DataFrame] = []

    print(f"Starting historical data fetch from {start_year} to {end_year}...")
    for season in seasons:
        try:
            players_df, teams_df = fetch_nba_api_data(season)
            all_players_data.append(players_df)
            all_teams_data.append(teams_df)
        except Exception as exc:
            print(f"Failed to fetch data for {season}. Error: {exc}")
            time.sleep(5)

    players_raw = pd.concat(all_players_data, ignore_index=True)
    teams_raw = pd.concat(all_teams_data, ignore_index=True)

    players_raw["GAME_DATE"] = pd.to_datetime(players_raw["GAME_DATE"])
    teams_raw["GAME_DATE"] = pd.to_datetime(teams_raw["GAME_DATE"])

    print(f"Success! Loaded {len(players_raw)} total player game logs.")
    return players_raw, teams_raw


def _add_player_rolling_features(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Add unshifted rolling player box-score and percentage features."""
    for stat in PLAYER_STATS_TO_ROLL:
        for window in windows:
            col = f"Rolling_{window}G_{stat}"
            df[col] = df.groupby("PLAYER_ID")[stat].transform(
                lambda series: series.rolling(window=window, min_periods=1).mean()
            )

    for window in windows:
        df[f"Rolling_{window}G_FG_PCT"] = np.where(
            df[f"Rolling_{window}G_FGA"] > 0,
            df[f"Rolling_{window}G_FGM"] / df[f"Rolling_{window}G_FGA"],
            0,
        )
        df[f"Rolling_{window}G_FG3_PCT"] = np.where(
            df[f"Rolling_{window}G_FG3A"] > 0,
            df[f"Rolling_{window}G_FG3M"] / df[f"Rolling_{window}G_FG3A"],
            0,
        )
        df[f"Rolling_{window}G_FT_PCT"] = np.where(
            df[f"Rolling_{window}G_FTA"] > 0,
            df[f"Rolling_{window}G_FTM"] / df[f"Rolling_{window}G_FTA"],
            0,
        )
        df[f"Rolling_{window}G_Games_Played"] = df.groupby("PLAYER_ID")["GAME_ID"].transform(
            lambda series: series.rolling(window=window, min_periods=1).count()
        )

    return df


def _build_team_matchups(teams_raw: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Create team-vs-opponent rolling context features for each team game row."""
    matchups = pd.merge(teams_raw, teams_raw, on="GAME_ID", suffixes=("", "_OPP"))
    matchups = matchups[matchups["TEAM_ID"] != matchups["TEAM_ID_OPP"]].copy()
    matchups = matchups.sort_values(by=["TEAM_ID", "GAME_DATE"]).reset_index(drop=True)

    for stat in TEAM_OFFENSE_COLS:
        for window in windows:
            matchups[f"Team_Rolling_{window}G_{stat}"] = matchups.groupby("TEAM_ID")[stat].transform(
                lambda series: series.rolling(window=window, min_periods=1).mean()
            )

    matchups["opponentScore"] = matchups["PTS_OPP"]
    for stat in TEAM_DEFENSE_COLS:
        for window in windows:
            matchups[f"Opp_Rolling_{window}G_{stat}"] = matchups.groupby("TEAM_ID")[stat].transform(
                lambda series: series.rolling(window=window, min_periods=1).mean()
            )

    return matchups


def get_nba_data(
    start_year: int = DEFAULT_START_YEAR,
    end_year: int = DEFAULT_END_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch and transform raw logs into training and inference datasets."""
    players_raw, teams_raw = fetch_multiple_seasons(start_year=start_year, end_year=end_year)

    df = players_raw.copy()
    df["home"] = df["MATCHUP"].str.contains("vs.").astype(int)
    df["MIN"] = df["MIN"].astype(float)
    df = df[df["MIN"] > 0].dropna(subset=["MIN", "PTS", "GAME_DATE"]).reset_index(drop=True)
    df = df.sort_values(by=["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df["days_of_rest"] = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
    df["days_of_rest"] = df["days_of_rest"].fillna(10).clip(upper=10)

    df = _add_player_rolling_features(df=df, windows=ROLLING_WINDOWS)

    current_players = df.groupby("PLAYER_ID").tail(1).copy()

    rolling_cols_player = [col for col in df.columns if "Rolling_" in col]
    df[rolling_cols_player] = df.groupby("PLAYER_ID")[rolling_cols_player].shift(1)

    matchups = _build_team_matchups(teams_raw=teams_raw, windows=ROLLING_WINDOWS)
    current_teams = matchups.groupby("TEAM_ID").tail(1).copy()

    rolling_cols_team = [
        col
        for col in matchups.columns
        if "Team_Rolling_" in col or "Opp_Rolling_" in col
    ]
    matchups[rolling_cols_team] = matchups.groupby("TEAM_ID")[rolling_cols_team].shift(1)

    team_profiles = matchups[["GAME_ID", "TEAM_ID", "TEAM_ID_OPP", *rolling_cols_team]]
    final_df = pd.merge(df, team_profiles, on=["GAME_ID", "TEAM_ID"], how="left")
    final_df = final_df.rename(columns={"TEAM_ID_OPP": "opponentTeamId"})

    columns_of_interest = [
        "GAME_ID",
        "GAME_DATE",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "opponentTeamId",
        "PTS",
        "is_playoff",
        "home",
        "days_of_rest",
        *PLAYER_INFERENCE_COLS,
        *TEAM_INFERENCE_COLS,
    ]

    player_context_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "days_of_rest", *PLAYER_INFERENCE_COLS]
    current_players = current_players[player_context_cols].copy()

    team_meta_cols = ["TEAM_ID"]
    for optional_col in ("TEAM_ABBREVIATION", "TEAM_NAME"):
        if optional_col in current_teams.columns:
            team_meta_cols.append(optional_col)
    current_teams = current_teams[[*team_meta_cols, *TEAM_INFERENCE_COLS]].copy()

    final_df = final_df[columns_of_interest].copy()

    final_df = final_df.dropna(
        subset=["Rolling_5G_PTS", "Opp_Rolling_5G_opponentScore"]
    ).reset_index(drop=True)

    return final_df, current_players.reset_index(drop=True), current_teams.reset_index(drop=True)


__all__ = [
    "DEFAULT_START_YEAR",
    "DEFAULT_END_YEAR",
    "ROLLING_WINDOWS",
    "PLAYER_INFERENCE_COLS",
    "TEAM_INFERENCE_COLS",
    "fetch_nba_api_data",
    "fetch_multiple_seasons",
    "get_nba_data",
]
