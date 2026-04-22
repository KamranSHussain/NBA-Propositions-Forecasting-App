# NBA Prop Forecasting App

Streamlit app for NBA player points prop forecasting using a transformer quantile model.

## What This Repo Does

- Loads multi-season NBA data from `nba_api`.
- Uses a pretrained transformer artifact to produce player-level quantile forecasts.
- Pulls live FanDuel player points lines via The Odds API event endpoints.
- Builds a live pick board by joining live lines with model projections.
- Shows historical backtest evaluation charts from a CSV.

## Project Structure

- `app.py`: Streamlit UI and app workflow.
- `src/data.py`: historical data fetch + feature engineering.
- `src/model.py`: transformer model and quantile loss.
- `src/service.py`: training/inference/evaluation services.
- `src/fanduel_live.py`: live FanDuel prop loader and local cache.
- `scripts/train_artifact.py`: artifact training entrypoint.
- `scripts/backtest_odds.py`: odds backtest utility.
- `models/`: trained model artifacts.
- `betting data/`: historical backtest inputs and related data.

## Core Modeling Setup

- Target: player `PTS`.
- Sequence length: 20 time steps.
- Active feature set in the default artifact: 28 features.
- Typical quantiles: `q10`, `q50`, `q90`.

### Input Shape

The model processes a tensor shaped like:

- `batch_size x sequence_length x feature_count`
- For the default artifact this is approximately `N x 20 x 28`.

## App Pages

### Predict Matchup

- Select home and away teams.
- Generate roster-level forecasts from the loaded artifact.
- View player quantiles for both teams.

### Betting Lines

- Uses startup snapshot of live FanDuel lines.
- Matches players to home/away team predictions.
- Builds recommendation cards and a detailed table.
- Includes historical backtest charts under the live section.

### Test Stats

- Displays artifact metadata.
- Shows model diagnostics and calibration-style charts.

## Live Odds Source

Live props are fetched from The Odds API using:

- `GET /v4/sports/basketball_nba/events`
- `GET /v4/sports/basketball_nba/events/{event_id}/odds`

The loader filters for:

- bookmaker: `fanduel`
- market: `player_points`

Environment variable required:

- `ODDS_API_KEY` (or `THE_ODDS_API_KEY`)

## Caching

The app uses multiple cache layers:

- Streamlit data/resource caching for datasets and artifacts.
- Local disk cache for live odds in `.cache/live_odds` with TTL and stale fallback.
- Session-state startup snapshot for live lines to reduce repeated API calls on reruns.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies.
3. Set odds API key.
4. Run Streamlit.

Example:

- `python3 -m venv venv`
- `source venv/bin/activate`
- `pip install -r requirements.txt`
- `export ODDS_API_KEY=your_key_here`
- `streamlit run app.py`

## Training / Refreshing Artifacts

To train a new artifact:

- `python scripts/train_artifact.py`

Artifacts are loaded from `models/` and configured in `app.py`.

## Notes

- Live game times are displayed in UTC in the pick board.
- If a live player cannot be matched to either roster, the row remains unassigned and drops from projected output.
- Historical charts rely on `betting data/backtests/partner_odds_backtest.csv`.

## License

See `LICENSE`.
