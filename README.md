# NBA Prop Forecasting App

Streamlit application for NBA player points forecasting with a custom transformer quantile-regression model, live FanDuel prop ingestion, and historical betting-line evaluation.

## What The App Does

This project combines three related workflows:

- **Predict Matchup**: generate player-level points forecasts for a selected home/away matchup.
- **Betting Lines**: join live FanDuel player-points lines to model forecasts and surface recommendation cards.
- **Test Stats**: inspect held-out model performance, calibration, and interval behavior on the test split.

The app is built around a **custom PyTorch transformer** trained with **pinball loss** for quantile regression.

## Current Default Configuration

The app currently uses the original **3-quantile model**:

- Artifact: `models/player_prop_artifacts_opp28.pt`
- Quantiles: `q10`, `q50`, `q90`
- Historical betting evaluation file: `betting data/backtests/partner_odds_backtest.csv`

These are the files the app loads by default in `app.py`.

## Repository Structure

```text
NBA-Prop-Forecasting-App/
├── app.py
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── data.py
│   ├── model.py
│   ├── service.py
│   └── fanduel_live.py
├── scripts/
│   ├── train_artifact.py
│   └── backtest_odds.py
├── models/
│   └── player_prop_artifacts_opp28.pt
└── betting data/
    ├── backtests/
    │   └── partner_odds_backtest.csv
    └── historical_event_odds_v4/
```

## Model Design

### Prediction Target

- Target variable: player points (`PTS`)
- Default forecast outputs: `q10`, `q50`, `q90`

### Why Quantiles?

Quantiles make the output more useful than a single point estimate:

- `q10`: lower-end outcome
- `q50`: median projection
- `q90`: upper-end outcome

This gives both a central forecast and an uncertainty band.

### Model Architecture

The forecasting model is a **transformer encoder** over fixed-length historical player-game sequences.

Important choices:

- **Sequence length**: 20 historical timesteps
- **Loss**: multi-quantile pinball loss
- **Context features**: home/away flag, playoff flag, days of rest, recent player box score features, opponent last-game team context
- **Cold-start handling**: learned fallback token for short-history players

## Data Pipeline

Historical data comes from `nba_api` and is processed in `src/data.py`.

Key engineering choices:

- multi-season regular season + playoff game logs
- leakage-safe shifts for prior-game features
- opponent context built from team-level last-game data
- padding and masking for short player histories

## Data Sources

This project uses two main external data sources:

- **NBA API (`nba_api`)**
  - used to pull historical NBA player and team game logs
  - used to build the training dataset for the forecasting model
  - used to derive player sequence features, team context, opponent context, and roster-based inference inputs

- **The Odds API**
  - used to pull live and historical sportsbook player-points betting lines
  - used to power the live Betting Lines page
  - used to build the historical betting-line backtest exports for recommendation evaluation
  - requires a personal API key for live odds and historical odds endpoints
  - the free Starter plan is rate-limited to **500 credits per month**
  - get a key here: [https://the-odds-api.com/](https://the-odds-api.com/)

## Train/Test Split

This project uses a **time-based split.**

- Split date: `2024-06-18`
- Train rows: `GAME_DATE <= split date`
- Test rows: `GAME_DATE > split date`

## Betting Recommendation Rule

The current live recommendation rule is intentionally simple:

- consider only **unders**
- require the betting line to be **closer to `q90` than to the median (`q50`)**
- require under decimal odds to be at least **1.81**

This is the rule currently described and used in `app.py`.

### Why This Rule

This selection method was chosen for three practical reasons:

- it stays aligned with the default **3-quantile** model, which only predicts `q10`, `q50`, and `q90`
- it is easy to explain visually: when the line sits closer to `q90` than to `q50`, the market line is already deep into the model's upper range, which makes the **under** the more defensible side
- it avoids pretending the model is a fully calibrated probability engine; instead of forcing a more complex EV formula, it uses a transparent distance-based rule that can be justified directly from the forecast interval

The odds floor (`1.81`) is used as a basic price filter so recommended bets are not taken at overly expensive odds.

## Historical Betting Evaluation

The historical betting export is built by joining model test-split predictions to archived sportsbook lines.

Current evaluation design:

- bookmakers kept: **FanDuel** and **DraftKings**
- rows are matched by `game_id` and normalized player name
- outputs include model quantiles, actual result, sportsbook odds, and grading status

The backtest utility is in `scripts/backtest_odds.py`.

For the current app, only `betting data/backtests/partner_odds_backtest.csv` is required at runtime.
The archived Odds API JSON files in `betting data/historical_event_odds_v4/` and the
`betting data/game_id_event_id_bijective.csv` mapping are only needed if you want to regenerate the backtest export.

## App Pages

### Predict Matchup

- choose a home team and away team
- run the model for both rosters
- inspect player forecast tables

### Betting Lines

- loads a startup snapshot of live FanDuel player-points lines
- joins those lines to model forecasts
- shows recommendation cards and historical recommendation charts

### Test Stats

- shows artifact metadata
- shows held-out accuracy metrics
- shows empirical quantile calibration
- shows interval width behavior vs player data volume

## Setup

The Betting Lines page and odds backtest scripts require a valid **The Odds API** key.
The free Starter plan is currently limited to **500 credits per month**. You can sign up
for a key here: [https://the-odds-api.com/](https://the-odds-api.com/)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
export ODDS_API_KEY=your_key_here
streamlit run app.py
```

## Training

Train the default 3-quantile artifact:

```bash
python scripts/train_artifact.py
```

To save a training run to a different artifact path:

```bash
python scripts/train_artifact.py --output models/my_experiment.pt
```

## Backtesting

Run the default historical betting export:

```bash
python scripts/backtest_odds.py
```

Run a backtest against a specific artifact and output path:

```bash
python scripts/backtest_odds.py \
  --artifact models/my_experiment.pt \
  --output "betting data/backtests/my_experiment_backtest.csv"
```

## Troubleshooting

- No live rows: verify `ODDS_API_KEY` / `THE_ODDS_API_KEY`
- Missing artifact: make sure the configured model file exists in `models/`
- Empty historical charts: make sure the expected backtest CSV exists in `betting data/backtests/`
- Slow startup: first-run dataset fetches depend on `nba_api`

## License

See `LICENSE`.
