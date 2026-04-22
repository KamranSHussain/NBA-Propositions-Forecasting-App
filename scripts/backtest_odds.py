"""Backtest Odds API dumps against model test-set predictions.

Usage:
    python scripts/backtest_partner_odds.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import string
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import get_nba_data
from src.service import evaluate_test_set


def _rolling_end_year_exclusive(today: pd.Timestamp | None = None) -> int:
    """Return end_year (exclusive) so current season is included automatically."""
    today = today or pd.Timestamp.today()
    return int(today.year + (1 if today.month >= 9 else 0))


def _normalize_player_name(name: object) -> str:
    """Normalize player names for robust joins."""
    if pd.isna(name):
        return ""
    value = unicodedata.normalize("NFKD", str(name))
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower()
    value = value.translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _load_artifacts(artifact_path: Path):
    """Load model artifacts from disk with torch version compatibility."""
    try:
        return torch.load(artifact_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(artifact_path, map_location="cpu")


def _flatten_player_points_odds(odds_root: Path) -> pd.DataFrame:
    """Flatten historical event odds JSON files to player_points rows."""
    rows: list[dict[str, object]] = []

    for fp in odds_root.glob("*.json"):
        try:
            raw = json.loads(fp.read_text())
        except Exception:
            continue

        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        if not isinstance(data, dict):
            continue

        event_id = data.get("id")
        if not event_id:
            continue

        snapshot_ts = raw.get("timestamp") if isinstance(raw, dict) else None
        for bookmaker in data.get("bookmakers", []):
            book_key = bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                if market.get("key") != "player_points":
                    continue

                for outcome in market.get("outcomes", []):
                    side = outcome.get("name")
                    player_name = outcome.get("description")
                    point = outcome.get("point")
                    price = outcome.get("price")

                    if not player_name or point is None or side not in ("Over", "Under"):
                        continue

                    rows.append(
                        {
                            "event_id": str(event_id),
                            "snapshot_ts": snapshot_ts,
                            "bookmaker": book_key,
                            "player_name": player_name,
                            "player_key": _normalize_player_name(player_name),
                            "line": float(point),
                            "side": side.lower(),
                            "price": float(price) if price is not None else np.nan,
                        }
                    )

    if not rows:
        return pd.DataFrame()

    odds_long = pd.DataFrame(rows)
    odds_wide = (
        odds_long.pivot_table(
            index=["event_id", "snapshot_ts", "bookmaker", "player_name", "player_key", "line"],
            columns="side",
            values="price",
            aggfunc="first",
        )
        .reset_index()
        .rename(columns={"over": "over_odds", "under": "under_odds"})
    )
    odds_wide.columns.name = None
    return odds_wide


def _build_event_commence_map(odds_root: Path) -> pd.DataFrame:
    """Build event_id -> commence date (UTC) map from odds snapshots."""
    rows: list[dict[str, object]] = []
    for fp in odds_root.glob("*.json"):
        try:
            raw = json.loads(fp.read_text())
        except Exception:
            continue

        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        if not isinstance(data, dict):
            continue

        event_id = data.get("id")
        commence_time = data.get("commence_time")
        if not event_id or not commence_time:
            continue

        rows.append({"event_id": str(event_id), "commence_time": commence_time})

    if not rows:
        return pd.DataFrame(columns=["event_id", "commence_date_utc"])

    out = pd.DataFrame(rows).drop_duplicates(subset=["event_id"], keep="first")
    out["commence_date_utc"] = pd.to_datetime(out["commence_time"], errors="coerce", utc=True).dt.date
    return out[["event_id", "commence_date_utc"]]


def _build_event_team_map(odds_root: Path) -> pd.DataFrame:
    """Build event_id -> home/away team names from odds snapshots."""
    rows: list[dict[str, object]] = []
    for fp in odds_root.glob("*.json"):
        try:
            raw = json.loads(fp.read_text())
        except Exception:
            continue

        data = raw.get("data", {}) if isinstance(raw, dict) else {}
        if not isinstance(data, dict):
            continue

        event_id = data.get("id")
        home_team = data.get("home_team")
        away_team = data.get("away_team")
        if not event_id or not home_team or not away_team:
            continue

        rows.append(
            {
                "event_id": str(event_id),
                "home_team": str(home_team),
                "away_team": str(away_team),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["event_id", "home_team", "away_team"])

    return pd.DataFrame(rows).drop_duplicates(subset=["event_id"], keep="first")


def run_backtest(
    artifact_path: Path,
    odds_root: Path,
    event_map_path: Path,
    output_csv: Path,
    start_year: int = 2020,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Run backtest by joining model test predictions to flattened odds lines."""
    artifacts = _load_artifacts(artifact_path)

    end_year = _rolling_end_year_exclusive()
    train_df, _, current_teams = get_nba_data(start_year=start_year, end_year=end_year)
    test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)

    preds = test_eval.predictions.copy()
    preds["GAME_DATE"] = pd.to_datetime(preds["GAME_DATE"])

    id_map = train_df[["PLAYER_ID", "GAME_DATE", "GAME_ID"]].copy()
    id_map["GAME_DATE"] = pd.to_datetime(id_map["GAME_DATE"])
    id_map = id_map.drop_duplicates(subset=["PLAYER_ID", "GAME_DATE", "GAME_ID"])
    id_map = id_map.drop_duplicates(subset=["PLAYER_ID", "GAME_DATE"], keep="first")

    preds = preds.merge(id_map, on=["PLAYER_ID", "GAME_DATE"], how="left")
    preds["game_id_norm"] = (
        preds["GAME_ID"].astype(str).str.replace(".0", "", regex=False).str.zfill(10)
    )
    preds["player_key"] = preds["PLAYER_NAME"].map(_normalize_player_name)
    team_lookup = current_teams[["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]].dropna().drop_duplicates()
    preds = preds.merge(team_lookup, on="TEAM_ID", how="left")

    odds = _flatten_player_points_odds(odds_root)
    if odds.empty:
        raise ValueError("No player_points odds rows found in historical odds directory.")

    game_event = pd.read_csv(event_map_path)
    game_event["event_id"] = game_event["event_id"].astype(str)
    game_event["game_id_norm"] = game_event["game_id"].astype(str).str.zfill(10)

    odds_join = odds.merge(game_event[["event_id", "game_id_norm"]], on="event_id", how="left")

    # Reduce duplicates by preferring primary books, then earliest snapshot.
    book_rank = {"draftkings": 0, "fanduel": 1}
    odds_join["book_rank"] = odds_join["bookmaker"].map(book_rank).fillna(9)
    odds_join["snapshot_ts_parsed"] = pd.to_datetime(
        odds_join["snapshot_ts"], errors="coerce", utc=True
    )
    odds_join = odds_join.sort_values(
        ["game_id_norm", "player_key", "line", "book_rank", "snapshot_ts_parsed"]
    )
    odds_best = odds_join.drop_duplicates(
        subset=["game_id_norm", "player_key", "line"], keep="first"
    ).copy()

    merged = odds_best.merge(
        preds[
            [
                "game_id_norm",
                "GAME_DATE",
                "PLAYER_ID",
                "PLAYER_NAME",
                "player_key",
                "TEAM_ID",
                "TEAM_ABBREVIATION",
                "TEAM_NAME",
                "actual",
                "q10",
                "q50",
                "q90",
            ]
        ],
        on=["game_id_norm", "player_key"],
        how="inner",
    )

    if merged.empty:
        raise ValueError("Joined backtest frame is empty. Check game and name alignment.")

    # Integrity checks: validate event->game mapping and event date alignment.
    merged["game_id_num"] = pd.to_numeric(merged["game_id_norm"], errors="coerce")
    map_num = game_event[["event_id", "game_id"]].copy()
    map_num["game_id_num"] = pd.to_numeric(map_num["game_id"], errors="coerce")
    merged = merged.merge(
        map_num[["event_id", "game_id_num"]],
        on="event_id",
        how="left",
        suffixes=("", "_from_map"),
    )
    game_id_match = merged["game_id_num"] == merged["game_id_num_from_map"]
    if not bool(game_id_match.all()):
        bad = int((~game_id_match).sum())
        raise ValueError(f"Detected {bad} rows where event_id->game_id mapping mismatched.")

    event_dates = _build_event_commence_map(odds_root)
    merged = merged.merge(event_dates, on="event_id", how="left")
    event_teams = _build_event_team_map(odds_root)
    merged = merged.merge(event_teams, on="event_id", how="left")
    merged["game_date"] = pd.to_datetime(merged["GAME_DATE"], errors="coerce").dt.date
    merged["date_delta_days"] = (
        pd.to_datetime(merged["game_date"]) - pd.to_datetime(merged["commence_date_utc"])
    ).dt.days
    # NBA local game dates can differ from UTC commence date by +/-1 day.
    bad_date_alignment = merged[merged["date_delta_days"].abs() > 1]
    if not bad_date_alignment.empty:
        raise ValueError(
            "Detected rows where |GAME_DATE - commence_date_utc| > 1 day. "
            f"Rows impacted: {len(bad_date_alignment)}"
        )

    merged["team"] = np.where(
        merged["TEAM_NAME"].astype(str).eq(merged["home_team"].astype(str)),
        merged["home_team"],
        np.where(merged["TEAM_NAME"].astype(str).eq(merged["away_team"].astype(str)), merged["away_team"], merged["TEAM_NAME"]),
    )
    merged["opponent"] = np.where(
        merged["team"].astype(str).eq(merged["home_team"].astype(str)),
        merged["away_team"],
        np.where(merged["team"].astype(str).eq(merged["away_team"].astype(str)), merged["home_team"], ""),
    )

    merged = merged[merged["bookmaker"].astype(str).str.lower().isin({"draftkings", "fanduel"})].copy()
    if merged.empty:
        raise ValueError("No FanDuel or DraftKings rows remained after filtering the backtest export.")

    merged["model_recommendation"] = np.where(
        merged["q50"] > merged["line"],
        "over",
        np.where(merged["q50"] < merged["line"], "under", "push"),
    )
    merged["actual_side"] = np.where(
        merged["actual"] > merged["line"],
        "over",
        np.where(merged["actual"] < merged["line"], "under", "push"),
    )
    merged["status"] = np.where(
        merged["model_recommendation"] == "push",
        "push",
        np.where(merged["model_recommendation"] == merged["actual_side"], "correct", "incorrect"),
    )
    merged.loc[merged["actual_side"] == "push", "status"] = "push"
    merged["edge"] = merged["q50"] - merged["line"]

    graded = merged[merged["status"].isin(["correct", "incorrect"])].copy()
    accuracy = float((graded["status"] == "correct").mean()) if len(graded) else float("nan")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    summary: dict[str, float | int | str] = {
        "test_predictions_rows": int(len(preds)),
        "odds_player_points_rows": int(len(odds_best)),
        "merged_rows": int(len(merged)),
        "merged_unique_games": int(merged["game_id_norm"].nunique()),
        "graded_rows": int(len(graded)),
        "accuracy_pct": float(round(accuracy * 100, 2)) if np.isfinite(accuracy) else float("nan"),
        "correct": int((merged["status"] == "correct").sum()),
        "incorrect": int((merged["status"] == "incorrect").sum()),
        "push": int((merged["status"] == "push").sum()),
        "game_id_match_rate_pct": 100.0,
        "max_abs_date_delta_days": int(merged["date_delta_days"].abs().max()),
        "min_game_date": str(pd.to_datetime(merged["GAME_DATE"]).min().date()),
        "max_game_date": str(pd.to_datetime(merged["GAME_DATE"]).max().date()),
        "output_csv": str(output_csv),
    }
    return merged, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest partner odds against model test predictions.")
    parser.add_argument("--artifact", default="models/player_prop_artifacts_opp28.pt")
    parser.add_argument("--odds-dir", default="betting data/historical_event_odds_v4")
    parser.add_argument("--event-map", default="betting data/game_id_event_id_bijective.csv")
    parser.add_argument(
        "--output",
        default="betting data/backtests/partner_odds_backtest.csv",
        help="Output CSV path for merged backtest rows.",
    )
    parser.add_argument("--start-year", type=int, default=2020)
    args = parser.parse_args()

    merged, summary = run_backtest(
        artifact_path=Path(args.artifact),
        odds_root=Path(args.odds_dir),
        event_map_path=Path(args.event_map),
        output_csv=Path(args.output),
        start_year=args.start_year,
    )

    print("--- BACKTEST COMPLETE ---")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nTop 10 by absolute edge:")
    top_cols = [
        "GAME_DATE",
        "PLAYER_NAME",
        "line",
        "q50",
        "actual",
        "model_recommendation",
        "actual_side",
        "status",
        "edge",
        "bookmaker",
    ]
    top = merged.assign(abs_edge=merged["edge"].abs()).sort_values("abs_edge", ascending=False)
    print(top[top_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
