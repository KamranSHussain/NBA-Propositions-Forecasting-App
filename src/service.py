"""Training and inference service layer for player-prop quantile regression.

This module is designed for app workflows where a user:
1. Chooses a train/test split date.
2. Trains a model.
3. Chooses home/away teams and game type.
4. Gets player-level floor/median/ceiling predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch
from nba_api.stats.endpoints import commonteamroster
from torch.utils.data import DataLoader, TensorDataset

from src.data import TEAM_INFERENCE_COLS
from src.model import DEFAULT_QUANTILES, PinballLoss, PlayerPropNN

TARGET_COLUMN = "PTS"
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 75
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_ROSTER_CAP = 18

ID_COLUMNS: tuple[str, ...] = (
    "GAME_ID",
    "GAME_DATE",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ID",
    "opponentTeamId",
)

CONTEXT_COLUMNS: tuple[str, ...] = (
    "is_playoff",
    "home",
    "days_of_rest",
)


@dataclass
class ModelArtifacts:
    """Objects and metadata produced by model training."""

    model: PlayerPropNN
    feature_columns: list[str]
    quantiles: tuple[float, ...]
    train_end_date: pd.Timestamp
    train_rows: int
    test_rows: int
    train_loss: float
    test_loss: float
    feature_mean: pd.Series
    feature_std: pd.Series


@dataclass
class TestSetEvaluation:
    """Evaluation bundle for model performance on held-out test rows."""

    summary: dict[str, float]
    quantile_metrics: pd.DataFrame
    predictions: pd.DataFrame


def _to_timestamp(split_date: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Normalize split_date into a pandas Timestamp."""
    parsed = pd.to_datetime(split_date)
    if pd.isna(parsed):
        raise ValueError("Invalid split_date. Provide a valid date-like value.")
    return pd.Timestamp(parsed)


def feature_columns_from_frame(df: pd.DataFrame) -> list[str]:
    """Infer model feature columns from the processed training frame."""
    excluded = set(ID_COLUMNS) | {TARGET_COLUMN}
    return [col for col in df.columns if col not in excluded]


def _split_train_test(df: pd.DataFrame, split_date: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split processed data by date into train/test segments."""
    if "GAME_DATE" not in df.columns:
        raise ValueError("Dataframe must contain GAME_DATE for split logic.")

    split_df = df.copy()
    split_df["GAME_DATE"] = pd.to_datetime(split_df["GAME_DATE"])
    train_df = split_df[split_df["GAME_DATE"] <= split_date].copy()
    test_df = split_df[split_df["GAME_DATE"] > split_date].copy()

    if train_df.empty:
        raise ValueError("Train split is empty. Move split_date later.")
    if test_df.empty:
        raise ValueError("Test split is empty. Move split_date earlier.")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _standardize_train_test(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Apply train-fit z-score scaling and return transformed frames + stats."""
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0).replace(0, 1.0)

    train_scaled = (train_x - mean) / std
    test_scaled = (test_x - mean) / std
    return train_scaled, test_scaled, mean, std


def _to_tensor(values: np.ndarray) -> torch.Tensor:
    """Convert numpy arrays to float32 torch tensors."""
    return torch.tensor(values.astype(np.float32), dtype=torch.float32)


def train_model(
    df: pd.DataFrame,
    split_date: str | date | datetime | pd.Timestamp,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    random_seed: int = 42,
) -> ModelArtifacts:
    """Train quantile regression model using a date-based train/test split."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in training dataframe.")

    split_ts = _to_timestamp(split_date)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    feature_cols = feature_columns_from_frame(df)
    train_df, test_df = _split_train_test(df=df, split_date=split_ts)

    train_x = train_df[feature_cols].copy()
    test_x = test_df[feature_cols].copy()
    train_x, test_x, feature_mean, feature_std = _standardize_train_test(train_x=train_x, test_x=test_x)

    train_y = train_df[[TARGET_COLUMN]].to_numpy()
    test_y = test_df[[TARGET_COLUMN]].to_numpy()

    train_ds = TensorDataset(_to_tensor(train_x.to_numpy()), _to_tensor(train_y))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = PlayerPropNN(input_size=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = PinballLoss(quantiles=quantiles)

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        train_loss = float(
            criterion(model(_to_tensor(train_x.to_numpy())), _to_tensor(train_y)).item()
        )
        test_loss = float(
            criterion(model(_to_tensor(test_x.to_numpy())), _to_tensor(test_y)).item()
        )

    return ModelArtifacts(
        model=model,
        feature_columns=feature_cols,
        quantiles=tuple(float(q) for q in quantiles),
        train_end_date=split_ts,
        train_rows=len(train_df),
        test_rows=len(test_df),
        train_loss=train_loss,
        test_loss=test_loss,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )


def team_lookup(current_teams: pd.DataFrame) -> pd.DataFrame:
    """Return unique team metadata for UI selectors (id, abbreviation, name)."""
    lookup_cols = ["TEAM_ID"]
    for optional_col in ("TEAM_ABBREVIATION", "TEAM_NAME"):
        if optional_col in current_teams.columns:
            lookup_cols.append(optional_col)
    return current_teams[lookup_cols].drop_duplicates().reset_index(drop=True)


def _normalized_features(features: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    """Apply training-time scaling stats to inference features."""
    aligned = features.copy()

    # Keep inference resilient if app-side feature assembly misses a column.
    for col in artifacts.feature_columns:
        if col not in aligned.columns:
            default_value = float(artifacts.feature_mean[col]) if col in artifacts.feature_mean.index else 0.0
            aligned[col] = default_value

    aligned = aligned[artifacts.feature_columns].copy()
    return (aligned - artifacts.feature_mean) / artifacts.feature_std


def _team_context_row(current_teams: pd.DataFrame, team_id: int | str) -> pd.Series:
    """Fetch one team context row by TEAM_ID."""
    team_rows = current_teams[current_teams["TEAM_ID"] == team_id]
    if team_rows.empty:
        raise ValueError(f"TEAM_ID {team_id} not found in current_teams.")
    return team_rows.iloc[0]


def _players_for_team(current_players: pd.DataFrame, team_id: int | str) -> pd.DataFrame:
    """Fetch players currently associated with a team."""
    if "TEAM_ID" not in current_players.columns:
        raise ValueError("current_players must include TEAM_ID for team-based predictions.")
    players = current_players[current_players["TEAM_ID"] == team_id].copy()
    if players.empty:
        raise ValueError(f"No players found for TEAM_ID {team_id}.")

    # Defensive dedupe in case upstream data ever includes repeated player rows.
    if "PLAYER_ID" in players.columns:
        players = players.drop_duplicates(subset=["PLAYER_ID"], keep="first")

    return players.reset_index(drop=True)


def _limit_roster_size(players: pd.DataFrame, max_players: int = DEFAULT_ROSTER_CAP) -> pd.DataFrame:
    """Keep a realistic roster-sized subset using recent usage proxies."""
    if players.empty or len(players) <= max_players:
        return players.reset_index(drop=True)

    ranked = players.copy()
    sort_cols: list[str] = []
    ascending: list[bool] = []

    for col in ("Rolling_10G_Games_Played", "Rolling_5G_Games_Played", "Rolling_10G_MIN", "Rolling_5G_MIN"):
        if col in ranked.columns:
            ranked[col] = pd.to_numeric(ranked[col], errors="coerce").fillna(0.0)
            sort_cols.append(col)
            ascending.append(False)

    if "PLAYER_NAME" in ranked.columns:
        sort_cols.append("PLAYER_NAME")
        ascending.append(True)

    if sort_cols:
        ranked = ranked.sort_values(by=sort_cols, ascending=ascending)

    return ranked.head(max_players).reset_index(drop=True)


def _default_roster_season(today: date | None = None) -> str:
    """Return NBA season label (e.g. 2025-26) for current date context."""
    today = today or date.today()
    start_year = today.year if today.month >= 9 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


@lru_cache(maxsize=256)
def _official_roster_player_ids(team_id: int, season: str) -> set[int]:
    """Fetch official roster player IDs for a team/season from NBA API."""
    roster_df = commonteamroster.CommonTeamRoster(team_id=team_id, season=season).get_data_frames()[0]
    if "PLAYER_ID" not in roster_df.columns:
        return set()
    return {int(pid) for pid in roster_df["PLAYER_ID"].dropna().tolist()}


def _filter_to_official_roster(players: pd.DataFrame, team_id: int | str, season: str) -> pd.DataFrame:
    """Restrict players to official NBA roster for the chosen team and season."""
    try:
        team_id_int = int(team_id)
    except (TypeError, ValueError):
        return _limit_roster_size(players)

    try:
        official_ids = _official_roster_player_ids(team_id=team_id_int, season=season)
    except Exception:
        # Fall back to game-log derived players if roster endpoint fails.
        return _limit_roster_size(players)

    if not official_ids or "PLAYER_ID" not in players.columns:
        return _limit_roster_size(players)

    filtered = players[players["PLAYER_ID"].astype(int).isin(official_ids)].copy()
    if not filtered.empty:
        return _limit_roster_size(filtered)

    # If IDs mismatch for any reason, avoid returning a bloated historical roster.
    return _limit_roster_size(players)


def _build_matchup_features(
    team_players: pd.DataFrame,
    own_team: pd.Series,
    opp_team: pd.Series,
    is_playoff: bool,
    home_flag: int,
) -> pd.DataFrame:
    """Assemble player-level inference features for one side of a matchup."""
    feature_df = team_players.copy()
    feature_df["is_playoff"] = int(is_playoff)
    feature_df["home"] = int(home_flag)

    for col in TEAM_INFERENCE_COLS:
        # Team offense from own team, opponent profile from opposing team.
        if col.startswith("Team_Rolling_"):
            feature_df[col] = own_team[col]
        else:
            feature_df[col] = opp_team[col]

    return feature_df


def _predict_from_features(feature_df: pd.DataFrame, artifacts: ModelArtifacts) -> np.ndarray:
    """Run model inference and return quantile predictions as numpy array."""
    model_input = _normalized_features(feature_df, artifacts=artifacts)
    with torch.no_grad():
        preds = artifacts.model(_to_tensor(model_input.to_numpy())).numpy()
    return preds


def predict_matchup(
    artifacts: ModelArtifacts,
    current_players: pd.DataFrame,
    current_teams: pd.DataFrame,
    home_team_id: int | str,
    away_team_id: int | str,
    is_playoff: bool,
    roster_season: str | None = None,
    enforce_official_roster: bool = True,
) -> pd.DataFrame:
    """Predict player props for all players on home and away teams."""
    home_team = _team_context_row(current_teams=current_teams, team_id=home_team_id)
    away_team = _team_context_row(current_teams=current_teams, team_id=away_team_id)

    home_players = _players_for_team(current_players=current_players, team_id=home_team_id)
    away_players = _players_for_team(current_players=current_players, team_id=away_team_id)

    if enforce_official_roster:
        season = roster_season or _default_roster_season()
        home_players = _filter_to_official_roster(home_players, team_id=home_team_id, season=season)
        away_players = _filter_to_official_roster(away_players, team_id=away_team_id, season=season)

    home_features = _build_matchup_features(
        team_players=home_players,
        own_team=home_team,
        opp_team=away_team,
        is_playoff=is_playoff,
        home_flag=1,
    )
    away_features = _build_matchup_features(
        team_players=away_players,
        own_team=away_team,
        opp_team=home_team,
        is_playoff=is_playoff,
        home_flag=0,
    )

    home_preds = _predict_from_features(home_features, artifacts=artifacts)
    away_preds = _predict_from_features(away_features, artifacts=artifacts)

    pred_cols = [f"q{int(q * 100)}" for q in artifacts.quantiles]

    home_out = home_features[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "home"]].copy()
    away_out = away_features[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "home"]].copy()

    for idx, col in enumerate(pred_cols):
        home_out[col] = home_preds[:, idx]
        away_out[col] = away_preds[:, idx]

    output = pd.concat([home_out, away_out], ignore_index=True)
    output["is_playoff"] = int(is_playoff)
    output = output.sort_values(by=["home", "q50"], ascending=[False, False]).reset_index(drop=True)
    return output


def get_matchup_rosters(
    current_players: pd.DataFrame,
    home_team_id: int | str,
    away_team_id: int | str,
    roster_season: str | None = None,
    enforce_official_roster: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return home/away roster frames for selected teams before prediction."""
    home_players = _players_for_team(current_players=current_players, team_id=home_team_id)
    away_players = _players_for_team(current_players=current_players, team_id=away_team_id)

    if enforce_official_roster:
        season = roster_season or _default_roster_season()
        home_players = _filter_to_official_roster(home_players, team_id=home_team_id, season=season)
        away_players = _filter_to_official_roster(away_players, team_id=away_team_id, season=season)

    return home_players.reset_index(drop=True), away_players.reset_index(drop=True)


def model_summary(artifacts: ModelArtifacts) -> dict[str, Any]:
    """Return lightweight model metadata for UI display."""
    return {
        "train_end_date": str(artifacts.train_end_date.date()),
        "train_rows": artifacts.train_rows,
        "test_rows": artifacts.test_rows,
        "train_loss": round(artifacts.train_loss, 6),
        "test_loss": round(artifacts.test_loss, 6),
        "feature_count": len(artifacts.feature_columns),
        "quantiles": list(artifacts.quantiles),
    }


def _pinball_loss_numpy(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Compute pinball loss for one quantile using numpy arrays."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def evaluate_test_set(df: pd.DataFrame, artifacts: ModelArtifacts) -> TestSetEvaluation:
    """Evaluate trained artifacts on the date-split test set."""
    _, test_df = _split_train_test(df=df, split_date=artifacts.train_end_date)

    feature_df = test_df[artifacts.feature_columns].copy()
    preds = _predict_from_features(feature_df=feature_df, artifacts=artifacts)
    y_true = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    quantile_rows: list[dict[str, float]] = []
    for idx, q in enumerate(artifacts.quantiles):
        y_q = preds[:, idx]
        quantile_rows.append(
            {
                "quantile": float(q),
                "pinball_loss": _pinball_loss_numpy(y_true, y_q, q),
                "empirical_coverage": float(np.mean(y_true <= y_q)),
                "mean_prediction": float(np.mean(y_q)),
            }
        )

    quantile_metrics = pd.DataFrame(quantile_rows)

    median_idx = int(np.argmin(np.abs(np.array(artifacts.quantiles) - 0.5)))
    y_median = preds[:, median_idx]
    mae = float(np.mean(np.abs(y_true - y_median)))
    rmse = float(np.sqrt(np.mean((y_true - y_median) ** 2)))

    y_var = float(np.var(y_true))
    r2 = float(1.0 - (np.mean((y_true - y_median) ** 2) / y_var)) if y_var > 0 else float("nan")

    interval_width = float("nan")
    interval_coverage = float("nan")
    try:
        q10_idx = artifacts.quantiles.index(0.10)
        q90_idx = artifacts.quantiles.index(0.90)
        q10 = preds[:, q10_idx]
        q90 = preds[:, q90_idx]
        interval_width = float(np.mean(q90 - q10))
        interval_coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
    except ValueError:
        # If q10/q90 are not configured, keep interval diagnostics as NaN.
        pass

    summary = {
        "test_rows": float(len(test_df)),
        "mae_q50": mae,
        "rmse_q50": rmse,
        "r2_q50": r2,
        "interval_width_q10_q90": interval_width,
        "interval_coverage_q10_q90": interval_coverage,
    }

    pred_cols = [f"q{int(q * 100)}" for q in artifacts.quantiles]
    predictions = test_df[["GAME_DATE", "PLAYER_NAME", "TEAM_ID", TARGET_COLUMN]].copy()
    predictions = predictions.rename(columns={TARGET_COLUMN: "actual"})
    for idx, col in enumerate(pred_cols):
        predictions[col] = preds[:, idx]

    return TestSetEvaluation(
        summary=summary,
        quantile_metrics=quantile_metrics,
        predictions=predictions.reset_index(drop=True),
    )


__all__ = [
    "ModelArtifacts",
    "TestSetEvaluation",
    "feature_columns_from_frame",
    "team_lookup",
    "get_matchup_rosters",
    "train_model",
    "predict_matchup",
    "model_summary",
    "evaluate_test_set",
]
