"""Training and inference service layer for player-prop quantile regression.

This module is designed for app workflows where a user:
1. Chooses a train/test split date.
2. Trains a model.
3. Chooses home/away teams and game type.
4. Gets player-level floor/median/ceiling predictions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import torch
from nba_api.stats.endpoints import commonteamroster
from torch.utils.data import DataLoader, TensorDataset

from src.data import RAW_PLAYER_SEQUENCE_COLS, TEAM_INFERENCE_COLS
from src.model import DEFAULT_QUANTILES, PinballLoss, PlayerPropTransformer

TARGET_COLUMN = "PTS"
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 200
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-5
DEFAULT_ROSTER_CAP = 18
DEFAULT_EARLY_STOPPING_PATIENCE = 12
DEFAULT_EARLY_STOPPING_MIN_DELTA = 1e-4
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_SEQUENCE_LENGTH = 20

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

    model: torch.nn.Module
    feature_columns: list[str]
    quantiles: tuple[float, ...]
    train_end_date: pd.Timestamp
    train_rows: int
    test_rows: int
    epochs_trained: int
    train_loss: float
    val_loss: float
    test_loss: float
    feature_mean: pd.Series
    feature_std: pd.Series
    model_type: str = "transformer"
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH


@dataclass
class TestSetEvaluation:
    """Evaluation bundle for model performance on held-out test rows."""

    summary: dict[str, float]
    quantile_metrics: pd.DataFrame
    predictions: pd.DataFrame
    player_interval_profile: pd.DataFrame
    games_bucket_metrics: pd.DataFrame
    outlier_summary: dict[str, float]


def _to_timestamp(split_date: str | date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Normalize split_date into a pandas Timestamp."""
    parsed = pd.to_datetime(split_date)
    if pd.isna(parsed):
        raise ValueError("Invalid split_date. Provide a valid date-like value.")
    return pd.Timestamp(parsed)


def feature_columns_from_frame(df: pd.DataFrame) -> list[str]:
    """Infer model feature columns from the processed training frame."""
    preferred_sequence_features = [
        *CONTEXT_COLUMNS,
        *RAW_PLAYER_SEQUENCE_COLS,
        *TEAM_INFERENCE_COLS,
    ]
    available_preferred = [col for col in preferred_sequence_features if col in df.columns]
    if available_preferred:
        return available_preferred

    # Fallback for legacy frames that do not expose raw sequence columns.
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


def _to_tensor(values: np.ndarray) -> torch.Tensor:
    """Convert numpy arrays to float32 torch tensors."""
    return torch.tensor(values.astype(np.float32), dtype=torch.float32)


def _build_history_tensors(
    df: pd.DataFrame,
    feature_cols: list[str],
    history_len: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed-length per-player history tensors from ordered game rows."""
    ordered = df.copy()
    ordered["GAME_DATE"] = pd.to_datetime(ordered["GAME_DATE"])
    ordered = ordered.sort_values(by=["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)

    feat_matrix = np.nan_to_num(
        ordered[feature_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    targets = np.nan_to_num(
        ordered[[TARGET_COLUMN]].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    player_ids = ordered["PLAYER_ID"].to_numpy()

    n_rows = len(ordered)
    feat_dim = len(feature_cols)
    seq = np.zeros((n_rows, history_len, feat_dim), dtype=np.float32)
    pad_mask = np.ones((n_rows, history_len), dtype=bool)

    start_idx = 0
    while start_idx < n_rows:
        end_idx = start_idx
        current_player = player_ids[start_idx]
        while end_idx < n_rows and player_ids[end_idx] == current_player:
            end_idx += 1

        for local_idx in range(end_idx - start_idx):
            global_idx = start_idx + local_idx
            hist_start = max(0, local_idx - history_len)
            history = feat_matrix[start_idx + hist_start : start_idx + local_idx]
            if len(history) == 0:
                # Avoid all-masked rows, which can yield non-finite attention outputs.
                seq[global_idx, -1, :] = 0.0
                pad_mask[global_idx, -1] = False
                continue

            hist_len = history.shape[0]
            seq[global_idx, history_len - hist_len :, :] = history
            pad_mask[global_idx, history_len - hist_len :] = False

        start_idx = end_idx

    return ordered, seq, pad_mask, targets


def train_model(
    df: pd.DataFrame,
    split_date: str | date | datetime | pd.Timestamp,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA,
    val_fraction: float = DEFAULT_VALIDATION_FRACTION,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    random_seed: int = 42,
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
) -> ModelArtifacts:
    """Train transformer quantile model with temporal validation early stopping."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' in training dataframe.")
    if max_epochs < 1:
        raise ValueError("max_epochs must be >= 1.")
    if early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be >= 1.")
    if early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1.")

    split_ts = _to_timestamp(split_date)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    feature_cols = feature_columns_from_frame(df)
    train_df, test_df = _split_train_test(df=df, split_date=split_ts)
    split_df = df.copy()
    split_df["GAME_DATE"] = pd.to_datetime(split_df["GAME_DATE"])

    train_mask_df = split_df["GAME_DATE"] <= split_ts
    train_features = split_df.loc[train_mask_df, feature_cols].apply(pd.to_numeric, errors="coerce")
    feature_mean = train_features.mean(axis=0).fillna(0.0)
    feature_std = (
        train_features.std(axis=0)
        .replace(0.0, 1.0)
        .replace([np.inf, -np.inf], 1.0)
        .fillna(1.0)
    )

    normalized_df = split_df.copy()
    normalized_df[feature_cols] = normalized_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    normalized_df[feature_cols] = (normalized_df[feature_cols] - feature_mean) / feature_std
    normalized_df[feature_cols] = (
        normalized_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    ordered_df, seq, pad_mask, targets = _build_history_tensors(
        df=normalized_df,
        feature_cols=feature_cols,
        history_len=sequence_length,
    )

    ordered_train_idx = ordered_df[ordered_df["GAME_DATE"] <= split_ts].sort_values("GAME_DATE").index.to_numpy()
    ordered_test_idx = ordered_df[ordered_df["GAME_DATE"] > split_ts].index.to_numpy()

    val_size = max(1, int(len(ordered_train_idx) * val_fraction))
    fit_size = len(ordered_train_idx) - val_size
    if fit_size < 1:
        raise ValueError("Not enough train rows for validation split. Increase date range.")

    fit_idx = ordered_train_idx[:fit_size]
    val_idx = ordered_train_idx[fit_size:]

    train_ds = TensorDataset(
        _to_tensor(seq[fit_idx]),
        torch.tensor(pad_mask[fit_idx], dtype=torch.bool),
        _to_tensor(targets[fit_idx]),
    )
    val_seq_tensor = _to_tensor(seq[val_idx])
    val_mask_tensor = torch.tensor(pad_mask[val_idx], dtype=torch.bool)
    val_y_tensor = _to_tensor(targets[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = PlayerPropTransformer(
        input_size=len(feature_cols),
        quantiles=quantiles,
        max_len=max(sequence_length + 1, 32),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = PinballLoss(quantiles=quantiles)

    best_state_dict = None
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_seq, batch_mask, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_seq, padding_mask=batch_mask)
            loss = criterion(preds, batch_y)
            if not torch.isfinite(loss):
                raise ValueError(
                    "Non-finite training loss detected. "
                    "Try lowering learning_rate and/or sequence_length."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            current_val_loss = float(
                criterion(model(val_seq_tensor, padding_mask=val_mask_tensor), val_y_tensor).item()
            )
        if not np.isfinite(current_val_loss):
            raise ValueError(
                "Validation loss became non-finite. "
                "Try lowering learning_rate and/or sequence_length."
            )

        if current_val_loss < (best_val_loss - early_stopping_min_delta):
            best_val_loss = current_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    model.eval()
    with torch.no_grad():
        fit_seq_tensor = _to_tensor(seq[fit_idx])
        fit_mask_tensor = torch.tensor(pad_mask[fit_idx], dtype=torch.bool)
        fit_y_tensor = _to_tensor(targets[fit_idx])

        test_seq_tensor = _to_tensor(seq[ordered_test_idx])
        test_mask_tensor = torch.tensor(pad_mask[ordered_test_idx], dtype=torch.bool)
        test_y_tensor = _to_tensor(targets[ordered_test_idx])

        train_loss = float(
            criterion(model(fit_seq_tensor, padding_mask=fit_mask_tensor), fit_y_tensor).item()
        )
        val_loss = float(
            criterion(model(val_seq_tensor, padding_mask=val_mask_tensor), val_y_tensor).item()
        )
        test_loss = float(
            criterion(model(test_seq_tensor, padding_mask=test_mask_tensor), test_y_tensor).item()
        )

    return ModelArtifacts(
        model=model,
        feature_columns=feature_cols,
        quantiles=tuple(float(q) for q in quantiles),
        train_end_date=split_ts,
        train_rows=len(train_df),
        test_rows=len(test_df),
        epochs_trained=best_epoch if best_epoch > 0 else max_epochs,
        train_loss=train_loss,
        val_loss=val_loss,
        test_loss=test_loss,
        feature_mean=feature_mean,
        feature_std=feature_std,
        model_type="transformer",
        sequence_length=sequence_length,
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
    normalized = (aligned - artifacts.feature_mean) / artifacts.feature_std
    normalized = normalized.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return normalized


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

    for col in ("MIN", "AST", "REB", "STL"):
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
        # Team features come from own team row; opponent features come from opponent row.
        if col.startswith("Team_"):
            feature_df[col] = own_team.get(col, 0.0)
        elif col.startswith("Opp_"):
            mirrored_team_col = col.replace("Opp_", "Team_", 1)
            feature_df[col] = opp_team.get(mirrored_team_col, opp_team.get(col, 0.0))
        else:
            feature_df[col] = opp_team.get(col, 0.0)

    return feature_df


def _build_inference_sequence(
    history_df: pd.DataFrame,
    current_features: pd.DataFrame,
    artifacts: ModelArtifacts,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded sequence of prior games plus the current matchup token."""
    feature_cols = artifacts.feature_columns
    seq_len = int(getattr(artifacts, "sequence_length", DEFAULT_SEQUENCE_LENGTH))

    history_matrix = np.nan_to_num(
        history_df[feature_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    current_matrix = np.nan_to_num(
        current_features[feature_cols].to_numpy(dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    seq = np.zeros((seq_len, len(feature_cols)), dtype=np.float32)
    pad_mask = np.ones((seq_len,), dtype=bool)

    history_slots = max(0, seq_len - 1)
    history_len = min(history_slots, history_matrix.shape[0])
    if history_len > 0:
        seq[history_slots - history_len : history_slots, :] = history_matrix[-history_len:]
        pad_mask[history_slots - history_len : history_slots] = False

    seq[-1, :] = current_matrix[0]
    pad_mask[-1] = False
    return seq, pad_mask


def _predict_from_features(
    feature_df: pd.DataFrame,
    current_history_df: pd.DataFrame,
    artifacts: ModelArtifacts,
) -> np.ndarray:
    """Run model inference on 20-step player histories and return quantile predictions."""
    seq_len = int(getattr(artifacts, "sequence_length", DEFAULT_SEQUENCE_LENGTH))
    feature_cols = artifacts.feature_columns

    seq_batch: list[np.ndarray] = []
    mask_batch: list[np.ndarray] = []
    for _, current_row in feature_df.iterrows():
        player_id = current_row["PLAYER_ID"]
        history_rows = current_history_df[
            (current_history_df["PLAYER_ID"] == player_id)
            & (current_history_df["GAME_DATE"] < current_row["GAME_DATE"])
        ].copy()
        history_rows = history_rows.sort_values(by=["GAME_DATE", "GAME_ID"]).reset_index(drop=True)

        # Match train-time scaling for both history tokens and the current matchup token.
        history_rows_norm = _normalized_features(history_rows, artifacts=artifacts)
        current_row_norm = _normalized_features(current_row.to_frame().T, artifacts=artifacts)

        seq, pad_mask = _build_inference_sequence(history_rows_norm, current_row_norm, artifacts)
        seq_batch.append(seq)
        mask_batch.append(pad_mask)

    if not seq_batch:
        return np.empty((0, len(artifacts.quantiles)), dtype=np.float32)

    seq_tensor = _to_tensor(np.stack(seq_batch, axis=0))
    mask_tensor = torch.tensor(np.stack(mask_batch, axis=0), dtype=torch.bool)

    with torch.no_grad():
        preds = artifacts.model(
            seq_tensor,
            padding_mask=mask_tensor,
        ).numpy()
    return preds


def _predict_test_set_with_transformer(
    df: pd.DataFrame,
    artifacts: ModelArtifacts,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate test-split predictions for transformer artifacts."""
    feature_cols = artifacts.feature_columns
    sequence_length = int(getattr(artifacts, "sequence_length", DEFAULT_SEQUENCE_LENGTH))

    full_df = df.copy()
    full_df["GAME_DATE"] = pd.to_datetime(full_df["GAME_DATE"])
    full_df[feature_cols] = full_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    full_df[feature_cols] = (full_df[feature_cols] - artifacts.feature_mean) / artifacts.feature_std
    full_df[feature_cols] = full_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ordered_df, seq, pad_mask, _ = _build_history_tensors(
        df=full_df,
        feature_cols=feature_cols,
        history_len=sequence_length,
    )

    test_idx = ordered_df[ordered_df["GAME_DATE"] > artifacts.train_end_date].index.to_numpy()
    if len(test_idx) == 0:
        raise ValueError("No test rows found for transformer evaluation.")

    seq_tensor = _to_tensor(seq[test_idx])
    mask_tensor = torch.tensor(pad_mask[test_idx], dtype=torch.bool)
    with torch.no_grad():
        preds = artifacts.model(seq_tensor, padding_mask=mask_tensor).numpy()

    return ordered_df.loc[test_idx].reset_index(drop=True), preds


def predict_matchup(
    artifacts: ModelArtifacts,
    current_players: pd.DataFrame,
    current_teams: pd.DataFrame,
    history_df: pd.DataFrame,
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

    home_preds = _predict_from_features(home_features, current_history_df=history_df, artifacts=artifacts)
    away_preds = _predict_from_features(away_features, current_history_df=history_df, artifacts=artifacts)

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
        "epochs_trained": artifacts.epochs_trained,
        "train_loss": round(artifacts.train_loss, 6),
        "val_loss": round(artifacts.val_loss, 6),
        "test_loss": round(artifacts.test_loss, 6),
        "feature_count": len(artifacts.feature_columns),
        "quantiles": list(artifacts.quantiles),
        "model_type": "transformer",
        "sequence_length": int(getattr(artifacts, "sequence_length", DEFAULT_SEQUENCE_LENGTH)),
    }


def _pinball_loss_numpy(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Compute pinball loss for one quantile using numpy arrays."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def evaluate_test_set(df: pd.DataFrame, artifacts: ModelArtifacts) -> TestSetEvaluation:
    """Evaluate trained artifacts on the date-split test set."""
    full_df = df.copy()
    full_df["GAME_DATE"] = pd.to_datetime(full_df["GAME_DATE"])
    total_games_by_player = (
        full_df.groupby("PLAYER_ID")["GAME_ID"].nunique().astype(float)
    )
    mean_minutes_by_player = (
        pd.to_numeric(full_df.get("MIN"), errors="coerce")
        .groupby(full_df["PLAYER_ID"])
        .mean()
        .astype(float)
    )

    test_df, preds = _predict_test_set_with_transformer(df=df, artifacts=artifacts)
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
    q10 = np.full_like(y_true, np.nan, dtype=float)
    q90 = np.full_like(y_true, np.nan, dtype=float)
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

    for idx, q in enumerate(artifacts.quantiles):
        y_q = preds[:, idx]
        q_label = f"q{int(q * 100)}"
        mae_q = float(np.mean(np.abs(y_true - y_q)))
        rmse_q = float(np.sqrt(np.mean((y_true - y_q) ** 2)))
        r2_q = float(1.0 - (np.mean((y_true - y_q) ** 2) / y_var)) if y_var > 0 else float("nan")
        summary[f"mae_{q_label}"] = mae_q
        summary[f"rmse_{q_label}"] = rmse_q
        summary[f"r2_{q_label}"] = r2_q

    pred_cols = [f"q{int(q * 100)}" for q in artifacts.quantiles]
    predictions = test_df[["GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", TARGET_COLUMN]].copy()
    predictions = predictions.rename(columns={TARGET_COLUMN: "actual"})
    for idx, col in enumerate(pred_cols):
        predictions[col] = preds[:, idx]

    predictions["residual_q50"] = predictions["actual"] - y_median
    predictions["abs_error_q50"] = predictions["residual_q50"].abs()
    predictions["interval_width_q10_q90"] = q90 - q10
    predictions["within_interval_q10_q90"] = (
        (predictions["actual"] >= q10) & (predictions["actual"] <= q90)
    )
    predictions["total_games_in_dataset"] = (
        predictions["PLAYER_ID"].map(total_games_by_player).fillna(0).astype(float)
    )
    predictions["mean_minutes_in_dataset"] = (
        predictions["PLAYER_ID"].map(mean_minutes_by_player).fillna(0).astype(float)
    )

    # IQR-based residual outlier detection is robust to skewed error distributions.
    residual_q1 = float(predictions["residual_q50"].quantile(0.25))
    residual_q3 = float(predictions["residual_q50"].quantile(0.75))
    residual_iqr = residual_q3 - residual_q1
    lower_bound = residual_q1 - (1.5 * residual_iqr)
    upper_bound = residual_q3 + (1.5 * residual_iqr)
    predictions["is_outlier"] = (
        (predictions["residual_q50"] < lower_bound) | (predictions["residual_q50"] > upper_bound)
    )

    outlier_mask = predictions["is_outlier"]
    non_outlier_mask = ~outlier_mask

    outlier_summary = {
        "outlier_count": float(outlier_mask.sum()),
        "outlier_rate": float(outlier_mask.mean()) if len(predictions) > 0 else float("nan"),
        "iqr_lower_bound": lower_bound,
        "iqr_upper_bound": upper_bound,
        "outlier_mae_q50": float(predictions.loc[outlier_mask, "abs_error_q50"].mean())
        if outlier_mask.any()
        else float("nan"),
        "outlier_rmse_q50": float(np.sqrt(np.mean(predictions.loc[outlier_mask, "residual_q50"] ** 2)))
        if outlier_mask.any()
        else float("nan"),
        "non_outlier_mae_q50": float(predictions.loc[non_outlier_mask, "abs_error_q50"].mean())
        if non_outlier_mask.any()
        else float("nan"),
        "non_outlier_rmse_q50": float(
            np.sqrt(np.mean(predictions.loc[non_outlier_mask, "residual_q50"] ** 2))
        )
        if non_outlier_mask.any()
        else float("nan"),
    }

    player_interval_profile = (
        predictions.groupby(["PLAYER_ID", "PLAYER_NAME"], as_index=False)
        .agg(
            total_games_in_dataset=("total_games_in_dataset", "max"),
            mean_minutes_in_dataset=("mean_minutes_in_dataset", "max"),
            test_rows=("PLAYER_ID", "size"),
            mean_interval_width_q10_q90=("interval_width_q10_q90", "mean"),
            mean_abs_error_q50=("abs_error_q50", "mean"),
            outlier_rate=("is_outlier", "mean"),
        )
        .sort_values(by=["total_games_in_dataset", "mean_interval_width_q10_q90"], ascending=[False, False])
        .reset_index(drop=True)
    )

    bins = [-np.inf, 25, 100, np.inf]
    labels = ["few_games_1_25", "mid_games_26_100", "many_games_101_plus"]
    predictions["games_bucket"] = pd.cut(
        predictions["total_games_in_dataset"], bins=bins, labels=labels
    )

    bucket_rows: list[dict[str, float | str]] = []
    for bucket_label in labels:
        bucket_df = predictions[predictions["games_bucket"] == bucket_label].copy()
        if bucket_df.empty:
            bucket_rows.append(
                {
                    "games_bucket": bucket_label,
                    "rows": 0.0,
                    "players": 0.0,
                    "mae_q50": float("nan"),
                    "rmse_q50": float("nan"),
                    "r2_q50": float("nan"),
                    "interval_coverage_q10_q90": float("nan"),
                    "avg_interval_width_q10_q90": float("nan"),
                }
            )
            continue

        y_bucket = bucket_df["actual"].to_numpy(dtype=float)
        y_bucket_pred = bucket_df["q50"].to_numpy(dtype=float)
        mse_bucket = float(np.mean((y_bucket - y_bucket_pred) ** 2))
        var_bucket = float(np.var(y_bucket))
        bucket_rows.append(
            {
                "games_bucket": bucket_label,
                "rows": float(len(bucket_df)),
                "players": float(bucket_df["PLAYER_ID"].nunique()),
                "mae_q50": float(np.mean(np.abs(y_bucket - y_bucket_pred))),
                "rmse_q50": float(np.sqrt(mse_bucket)),
                "r2_q50": float(1.0 - (mse_bucket / var_bucket)) if var_bucket > 0 else float("nan"),
                "interval_coverage_q10_q90": float(bucket_df["within_interval_q10_q90"].mean()),
                "avg_interval_width_q10_q90": float(bucket_df["interval_width_q10_q90"].mean()),
            }
        )

    games_bucket_metrics = pd.DataFrame(bucket_rows)

    quantile_metrics = quantile_metrics.copy()
    quantile_metrics["nominal_quantile"] = quantile_metrics["quantile"]
    quantile_metrics["calibration_gap"] = (
        quantile_metrics["empirical_coverage"] - quantile_metrics["nominal_quantile"]
    )

    return TestSetEvaluation(
        summary=summary,
        quantile_metrics=quantile_metrics,
        predictions=predictions.reset_index(drop=True),
        player_interval_profile=player_interval_profile,
        games_bucket_metrics=games_bucket_metrics,
        outlier_summary=outlier_summary,
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
