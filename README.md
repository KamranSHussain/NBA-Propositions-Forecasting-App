# NBA Player Proposition Forecasting App

A NBA player-prop forecasting project with a production-style Streamlit interface.

This repository walks through the full workflow of:
1. Pulling real NBA data from `nba_api`.
2. Building leakage-safe rolling player and team features.
3. Training a PyTorch quantile-regression model offline.
4. Evaluating out-of-sample performance.
5. Loading a pre-trained artifact and producing matchup-specific player projections (`q10`, `q50`, `q90`) with roster-aware filtering.

## Overview

Sports betting and prop modeling benefit from uncertainty-aware forecasts. Instead of only predicting one point estimate, this project predicts a **distribution band** for each player:

1. `q10`: conservative floor outcome.
2. `q50`: median outcome.
3. `q90`: ceiling outcome.

The app is designed to be easy for non-ML users:
1. Auto-load processed data from 2020 to present.
2. Auto-load a pre-trained model artifact.
3. Choose a matchup and playoff flag.
4. View roster previews and quantile projections for both teams.

## Educational Goals

This project demonstrates a practical ML system with a clear separation of concerns:

1. **Data Engineering**
Build rolling statistics and opponent context from noisy game logs while avoiding leakage.

2. **Modeling for Uncertainty**
Use pinball loss for simultaneous quantile regression in PyTorch.

3. **Evaluation Beyond One Metric**
Track MAE/RMSE/R2 plus interval width and empirical coverage.

4. **Application Layer Design**
Expose a clean service API and connect it to an interactive Streamlit front end.

## Methods and Architecture

### 1. Data Source (`nba_api`)

The pipeline uses `LeagueGameLog` for both:
1. Player logs (`P`).
2. Team logs (`T`).

For each season in a configured range, it pulls:
1. Regular season data.
2. Playoff data.

### 2. Feature Engineering (`src/data.py`)

The processing pipeline creates three outputs:
1. `final_df`: model-ready training data.
2. `current_players`: latest player feature snapshot for inference.
3. `current_teams`: latest team/opponent context for inference.

Key feature groups:
1. Player rolling stats over 5 and 10 games (points, minutes, attempts, assists, rebounds, turnovers, percentages).
2. Team rolling offense context.
3. Opponent rolling defense context.
4. Game context features: `home`, `is_playoff`, `days_of_rest`.

Leakage prevention:
1. Rolling features are shifted so each row uses only prior information at train time.

### Complete Model Feature List

The model target is `PTS`.

The model uses all non-ID, non-target columns at training time. In the current pipeline, that means the following **33 features**:

| Feature | Short Description |
| --- | --- |
| `is_playoff` | `1` if playoff game, else `0` for regular season. |
| `home` | `1` if player is at home, `0` if away. |
| `days_of_rest` | Days since the player's previous game (capped in preprocessing). |
| `Rolling_5G_Games_Played` | Number of games played in the player's last 5-game window. |
| `Rolling_10G_Games_Played` | Number of games played in the player's last 10-game window. |
| `Rolling_5G_MIN` | Player average minutes over last 5 games. |
| `Rolling_10G_MIN` | Player average minutes over last 10 games. |
| `Rolling_5G_PTS` | Player average points over last 5 games. |
| `Rolling_10G_PTS` | Player average points over last 10 games. |
| `Rolling_5G_FGA` | Player average field-goal attempts over last 5 games. |
| `Rolling_10G_FGA` | Player average field-goal attempts over last 10 games. |
| `Rolling_5G_FG_PCT` | Player rolling FG% over last 5 games. |
| `Rolling_10G_FG_PCT` | Player rolling FG% over last 10 games. |
| `Rolling_5G_FG3A` | Player average 3-point attempts over last 5 games. |
| `Rolling_10G_FG3A` | Player average 3-point attempts over last 10 games. |
| `Rolling_5G_FTA` | Player average free-throw attempts over last 5 games. |
| `Rolling_10G_FTA` | Player average free-throw attempts over last 10 games. |
| `Rolling_5G_AST` | Player average assists over last 5 games. |
| `Rolling_10G_AST` | Player average assists over last 10 games. |
| `Rolling_5G_REB` | Player average rebounds over last 5 games. |
| `Rolling_10G_REB` | Player average rebounds over last 10 games. |
| `Rolling_5G_TOV` | Player average turnovers over last 5 games. |
| `Rolling_10G_TOV` | Player average turnovers over last 10 games. |
| `Team_Rolling_5G_PTS` | Team average points scored over its last 5 games. |
| `Team_Rolling_10G_PTS` | Team average points scored over its last 10 games. |
| `Team_Rolling_5G_FGA` | Team average field-goal attempts over its last 5 games. |
| `Team_Rolling_10G_FGA` | Team average field-goal attempts over its last 10 games. |
| `Opp_Rolling_5G_opponentScore` | Opponent's recent points allowed proxy over last 5 games. |
| `Opp_Rolling_10G_opponentScore` | Opponent's recent points allowed proxy over last 10 games. |
| `Opp_Rolling_5G_STL` | Opponent average steals over last 5 games. |
| `Opp_Rolling_10G_STL` | Opponent average steals over last 10 games. |
| `Opp_Rolling_5G_BLK` | Opponent average blocks over last 5 games. |
| `Opp_Rolling_10G_BLK` | Opponent average blocks over last 10 games. |

Columns excluded from modeling:
1. `GAME_ID`
2. `GAME_DATE`
3. `PLAYER_ID`
4. `PLAYER_NAME`
5. `TEAM_ID`
6. `opponentTeamId`
7. `PTS` (target)

### 3. Model (`src/model.py`)

`PlayerPropNN` is a feedforward neural network with dropout regularization and 3-output head:
1. Output dimension = number of quantiles (`0.10`, `0.50`, `0.90`).

`PinballLoss` is used for quantile regression.

For a quantile `q`, target `y`, prediction `y_hat`:

`L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))`

Interpretation:
1. Underestimation is penalized differently than overestimation.
2. This asymmetry lets the model learn probabilistic bands instead of a single average.

### 4. Service Layer (`src/service.py`)

Core functions:
1. `train_model(...)`: date-split training and artifact creation.
2. `evaluate_test_set(...)`: test metrics and per-row predictions.
3. `predict_matchup(...)`: home/away player projections.
4. `get_matchup_rosters(...)`: roster preview before prediction.
5. `team_lookup(...)`: team selector metadata.

Roster logic:
1. Official roster filtering is enforced through `CommonTeamRoster`.
2. If roster API fails or mismatches occur, fallback is capped to a realistic roster size.
3. Defensive dedupe by `PLAYER_ID` prevents duplicate player rows.

## Streamlit App Workflow (`app.py`)

The app now follows an automated setup workflow.

### Startup (Automatic)
1. Fetch and cache processed NBA data from `2020` to the current season.
2. Load pre-trained model artifacts from `models/player_prop_artifacts.pt`.
3. Compute and render test diagnostics (calibration, outlier behavior, and data-volume comparisons).
4. Show artifact metadata (split date and train/test rows).

### Predict Matchup
1. Select home and away teams by name.
2. Toggle playoffs flag.
3. Review **official roster preview** for both teams.
4. Calculate predictions.
5. View player-level quantile tables for each team.

### Evaluation Diagnostics (Automatic)
The app now includes startup-computed evaluation components on the held-out test split:
1. Headline metrics: `MAE`, `RMSE`, `R2` for `q50`.
2. Per-quantile error table: `q10`, `q50`, `q90` for `MAE`, `RMSE`, `R2`.
3. Empirical calibration plot: nominal quantile vs empirical coverage.
4. Outlier summary: robust IQR-based outlier flags on `q50` residuals.
5. Data-volume analysis: interval width (`q90-q10`) vs total games per player in the full dataset.
6. Bucketed performance table for player history volume groups (`1-25`, `26-100`, `101+` total games).

State behavior:
1. Predictions are reset when matchup inputs change to avoid stale outputs.

## Installation and Setup

### Prerequisites

1. Python `3.10+` recommended.
2. Internet access (required for NBA API requests).

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd NBA-Prop-Forecasting-App
```

### 2. Create Virtual Environment

macOS/Linux:

```bash
python3 -m venv env
source env/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

## Train/Refresh Model Artifact (Maintainer Workflow)

The user-facing app does not train models. Maintainers refresh artifacts offline and save them to disk.

Default behavior:
1. Data window starts at `2020`.
2. End year is rolling and includes the current season.
3. Split date is fixed at `2024-06-18`.

```bash
python scripts/train_artifact.py
```

Optional overrides:

```bash
python scripts/train_artifact.py --start-year 2020 --end-year 2026 --split-date 2024-06-18 --output models/player_prop_artifacts.pt
```

## Programmatic Usage (Optional)

### Train and Save Artifact

```python
from src.data import get_nba_data
from src.service import train_model, evaluate_test_set
import torch

train_df, current_players, current_teams = get_nba_data(start_year=2020, end_year=2026)

artifacts = train_model(
	 df=train_df,
	 split_date="2024-06-18",
	 max_epochs=200,
	 early_stopping_patience=12,
	 batch_size=256,
	 learning_rate=1e-3,
)

test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)
print(test_eval.summary)
torch.save(artifacts, "models/player_prop_artifacts.pt")
```

### Predict Matchup

```python
from src.service import predict_matchup, team_lookup

teams = team_lookup(current_teams)
home_team_id = int(teams.iloc[0]["TEAM_ID"])
away_team_id = int(teams.iloc[1]["TEAM_ID"])

predictions = predict_matchup(
	 artifacts=artifacts,
	 current_players=current_players,
	 current_teams=current_teams,
	 home_team_id=home_team_id,
	 away_team_id=away_team_id,
	 is_playoff=False,
	 enforce_official_roster=True,
)

print(predictions.head())
```

## Repository Structure

```text
NBA-Proposition-Forecasting-App/
|- app.py                    # Streamlit dashboard
|- requirements.txt          # Python dependencies
|- LICENSE
|- README.md
|- src/
|  |- data.py                # Data ingestion + feature engineering
|  |- model.py               # Quantile model + pinball loss
|  |- service.py             # Train/eval/predict service layer
```

## Metrics and Interpretation Notes

1. `MAE (q50)` and `RMSE (q50)` measure median-forecast point error.
2. `R2 (q50)` shows explained variance for median predictions.
3. `Interval Width (q10-q90)` captures uncertainty spread.
4. `Coverage (q10-q90)` estimates how often true outcomes fall inside the predicted band.

Practical interpretation:
1. Narrow intervals with low coverage can indicate overconfidence.
2. Very wide intervals with high coverage can indicate underconfident forecasts.

## Troubleshooting

1. **NBA API rate/network issues**
	1. Retry after a short wait.
	2. Ensure stable internet.
	3. Reduce year range while testing.

2. **Long data load times**
	1. Multi-season pulls are expensive.
	2. Start with narrower windows for iteration.

3. **Roster oddities**
	1. App enforces official roster filtering.
	2. If API lookup fails, capped fallback logic is used.

4. **Reproducibility differences**
	1. Neural network training is stochastic (shuffle/dropout).
	2. Small metric variation across runs is expected.

## License

This project is distributed under the terms in `LICENSE`.
