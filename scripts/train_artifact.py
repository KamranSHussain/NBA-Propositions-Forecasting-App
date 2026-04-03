"""Train and persist model artifacts for app startup loading.

Usage:
    python scripts/train_artifact.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import torch

# Ensure imports from src/ work when executing this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import get_nba_data
from src.service import evaluate_test_set, model_summary, train_model

DEFAULT_START_YEAR = 2020
DEFAULT_SPLIT_DATE = "2024-06-18"
DEFAULT_OUTPUT_PATH = Path("models/player_prop_artifacts.pt")


def _rolling_end_year_exclusive(today: date | None = None) -> int:
    """Return end_year (exclusive) so current season is included automatically."""
    today = today or date.today()
    return today.year + (1 if today.month >= 9 else 0)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for artifact training job."""
    parser = argparse.ArgumentParser(description="Train and save NBA prop model artifact")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    parser.add_argument(
        "--end-year",
        type=int,
        default=_rolling_end_year_exclusive(),
        help="Exclusive upper year bound (auto-resolves to current season if omitted).",
    )
    parser.add_argument("--split-date", type=str, default=DEFAULT_SPLIT_DATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    """Train model artifact and save it to disk."""
    args = parse_args()

    if args.end_year <= args.start_year:
        raise ValueError("end-year must be greater than start-year.")

    print(f"Fetching data from {args.start_year} to {args.end_year} (exclusive)...")
    train_df, _, _ = get_nba_data(start_year=args.start_year, end_year=args.end_year)

    print(f"Training model with fixed split date {args.split_date}...")
    artifacts = train_model(
        df=train_df,
        split_date=args.split_date,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.patience,
    )

    test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifacts, args.output)

    print(f"Saved artifact to {args.output}")
    print("Model summary:")
    for key, value in model_summary(artifacts).items():
        print(f"  - {key}: {value}")

    print("Test summary:")
    for key, value in test_eval.summary.items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
