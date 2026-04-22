# NBA Prop Forecasting App

Streamlit application for NBA player points forecasting with a transformer quantile model, live FanDuel prop ingestion, and historical backtest evaluation.

## Overview

This repository combines three workflows in one app:

- Matchup forecasting: player-level quantile projections (`q10`, `q50`, `q90`) for selected home/away teams.
- Live prop research: pull upcoming FanDuel player points lines, join to model outputs, and surface recommendation cards.
- Historical evaluation: read backtest CSV output and visualize recommendation accuracy and bankroll progression.

## Repository Structure

```text
NBA-Prop-Forecasting-App/
├── app.py                          # Streamlit entrypoint and UI/page logic
├── requirements.txt                # Python dependencies
├── LICENSE
├── README.md
├── src/
│   ├── data.py                     # NBA API ingestion + feature engineering
│   ├── model.py                    # Transformer model and pinball loss
│   ├── service.py                  # Train/infer/evaluate service layer
│   └── fanduel_live.py             # Live odds loader + disk cache
├── scripts/
│   ├── train_artifact.py           # Train and save model artifact
│   └── backtest_odds.py            # Historical odds backtest utility
├── models/
│   └── player_prop_artifacts_opp28.pt   # Default artifact loaded by app
└── betting data/
	├── backtests/
	│   └── partner_odds_backtest.csv    # Historical eval input for app charts
	├── box_scores/                 # Supporting backtest data
	└── historical_event_odds_v4/   # Cached/raw historical event odds payloads
```

## Modeling Design

### Target and Outputs

- Target variable: player points (`PTS`).
- Primary outputs: quantile predictions (typically `q10`, `q50`, `q90`).

### Input Tensor Shape

- Runtime shape is `batch_size x sequence_length x feature_count`.
- Current defaults are approximately `N x 20 x 28`.

### Sequence Semantics

- Training/evaluation: 20-step historical context per player.
- Live inference: prior history plus a current matchup token in the final slot.
- Short histories are zero-padded and masked.

### Core Context Features

- Game context: `home`, `is_playoff`, `days_of_rest`.
- Player recent box score sequence features.
- Opponent team last-game context features (`Opp_LastGame_*`).

## Data Pipeline

### Historical Dataset Construction

`src/data.py`:

- Pulls multi-season regular season + playoff logs from `nba_api`.
- Sets `is_playoff` from source season type.
- Derives `home` from `MATCHUP` (`vs.` indicates home).
- Builds team last-game features with leakage-safe shifts.
- Builds opponent last-game features by pairing game-level team rows.

### Live Odds Pipeline

`src/fanduel_live.py`:

- Calls The Odds API event list endpoint.
- Calls event-level odds endpoint for each selected event.
- Filters to bookmaker `fanduel` and market `player_points`.
- Emits normalized rows with over/under decimal odds.

## Live Prediction and Matching Logic

In Betting Lines page (`app.py`):

- Live rows carry `home_team` and `away_team` from event payloads.
- Team IDs are resolved from current team metadata.
- `predict_matchup(...)` is called per unique home/away pairing.
- Each live prop row is matched to home or away roster by `(normalized_player_name, TEAM_ID)`.
- Recommendation cards are created only after successful model-line join.

## Recommendation Rule

Current strict rule in `app.py`:

- `edge <= -4.0`
- model pick is `over` or `under`
- selected side decimal odds `>= 1.81`

This governs which picks are marked as recommended.

## App Pages

### Predict Matchup

- Choose home and away teams.
- Run model inference for both rosters.
- Inspect player quantile tables.

### Betting Lines

- Uses a startup snapshot of live FanDuel lines.
- Displays a card-style live pick board.
- Shows detailed joined table (model + line fields).
- Includes historical evaluation charts below live picks.

### Test Stats

- Displays artifact metadata and split diagnostics.
- Shows calibration/performance visualizations on held-out data.

## Caching and Refresh Behavior

The app intentionally reduces API calls with layered caching:

- Streamlit data/resource caches for datasets and artifact loading.
- Disk cache in `.cache/live_odds` (TTL currently 120s, stale fallback enabled).
- Session startup snapshot for live lines so reruns do not repeatedly fetch.

Behavior summary:

- On new app launch/session: live fetch path runs (or serves fresh disk cache).
- On Streamlit rerun within same session: startup snapshot is reused.

## Configuration

### Environment Variables

- `ODDS_API_KEY` (preferred) or `THE_ODDS_API_KEY`.

### Important App Constants

Defined in `app.py`:

- `MODEL_ARTIFACT_PATH = models/player_prop_artifacts_opp28.pt`
- `BACKTEST_EVAL_PATH = betting data/backtests/partner_odds_backtest.csv`
- `RECOMMENDER_EDGE_MAX = -4.0`
- `RECOMMENDER_MIN_DECIMAL_ODDS = 1.81`
- `LIVE_MAX_EVENTS = 8`

## Setup

1. Create a virtual environment.
2. Activate it.
3. Install dependencies.
4. Export your odds API key.
5. Run Streamlit.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ODDS_API_KEY=your_key_here
streamlit run app.py
```

## Training and Backtesting

Train/refresh artifact:

```bash
python scripts/train_artifact.py
```

Run historical odds backtest utility:

```bash
python scripts/backtest_odds.py
```

## Troubleshooting

- No live rows: verify `ODDS_API_KEY`, quota, and endpoint availability.
- Empty recommended picks: check current lines, model matches, and recommender thresholds.
- Missing projections: ensure artifact exists at configured path and team/player mappings resolve.
- Time display: pick-board game times are displayed in UTC.

## License

See `LICENSE`.
