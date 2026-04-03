"""Streamlit app for NBA player prop quantile predictions."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from src.data import get_nba_data
from src.service import (
    evaluate_test_set,
    get_matchup_rosters,
    model_summary,
    predict_matchup,
    team_lookup,
)

st.set_page_config(page_title="NBA Player Prop Predictor", layout="wide")

st.markdown(
    """
<style>
:root {
    --sport-bg: #0b1220;
    --sport-surface: #111b2f;
    --sport-card: #16233d;
    --sport-accent: #f97316;
    --sport-accent-2: #22c55e;
    --sport-text: #e5e7eb;
    --sport-muted: #94a3b8;
    --sport-border: #223250;
}

@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600;700;800&family=Teko:wght@500;600;700&display=swap');

.stApp {
    background:
        radial-gradient(circle at 10% 15%, rgba(249, 115, 22, 0.18), transparent 35%),
        radial-gradient(circle at 90% 10%, rgba(34, 197, 94, 0.15), transparent 35%),
        linear-gradient(180deg, #0a101c 0%, var(--sport-bg) 100%);
    color: var(--sport-text);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1627 0%, #0d1a2f 100%);
    border-right: 1px solid var(--sport-border);
}

h1, h2, h3 {
    font-family: 'Teko', sans-serif !important;
    letter-spacing: 0.03em;
    color: #f8fafc;
}

p, label, div, span {
    font-family: 'Barlow', sans-serif;
}

.sport-hero {
    border: 1px solid var(--sport-border);
    background: linear-gradient(135deg, rgba(249, 115, 22, 0.14), rgba(22, 35, 61, 0.9));
    border-radius: 14px;
    padding: 0.9rem 1rem 0.8rem;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(3, 10, 23, 0.35);
}

.sport-hero h2 {
    margin: 0;
    line-height: 1;
    font-size: 2rem;
}

.sport-hero p {
    margin: 0.2rem 0 0;
    color: #cbd5e1;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(17, 27, 47, 0.96), rgba(14, 23, 40, 0.96));
    border: 1px solid var(--sport-border);
    border-radius: 12px;
    padding: 0.65rem 0.8rem;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.02);
}

[data-testid="stMetricLabel"] {
    color: var(--sport-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #f8fafc;
    font-family: 'Teko', sans-serif;
    letter-spacing: 0.02em;
}

.stButton > button {
    border-radius: 10px;
    border: 1px solid #fb923c;
    background: linear-gradient(180deg, #fb923c, #f97316);
    color: #0f172a;
    font-weight: 700;
    letter-spacing: 0.01em;
}

.stButton > button:hover {
    border-color: #fdba74;
    background: linear-gradient(180deg, #fdba74, #fb923c);
}

.stButton > button:focus {
    box-shadow: 0 0 0 0.2rem rgba(249, 115, 22, 0.35);
}

[data-baseweb="select"] > div,
.stDateInput > div,
.stNumberInput > div,
.stTextInput > div,
[data-testid="stSlider"] {
    background: var(--sport-surface);
}

[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--sport-border);
    border-radius: 12px;
    overflow: hidden;
}

[data-testid="stExpander"] {
    border: 1px solid var(--sport-border);
    border-radius: 10px;
    background: rgba(17, 27, 47, 0.8);
}

.stCaption {
    color: var(--sport-muted) !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="sport-hero">
  <h2>NBA Player Prop Predictor</h2>
  <p>Quantile-driven projections for matchup-focused prop research.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.caption("Data and model load automatically, so you can jump straight to matchup quantile predictions.")

DATA_START_YEAR = 2020
TRAIN_TEST_SPLIT_DATE = pd.Timestamp("2024-06-18")
MODEL_ARTIFACT_PATH = Path("models/player_prop_artifacts.pt")


def _rolling_end_year_exclusive(today: date | None = None) -> int:
    """Return end_year (exclusive) so current season is included automatically."""
    today = today or date.today()
    return today.year + (1 if today.month >= 9 else 0)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Load and cache rolling processed datasets from 2020 to present."""
    end_year = _rolling_end_year_exclusive()
    train_df, current_players, current_teams = get_nba_data(start_year=DATA_START_YEAR, end_year=end_year)
    return train_df, current_players, current_teams, end_year


@st.cache_resource(show_spinner=False)
def load_pretrained_artifacts(artifact_path: str):
    """Load pre-trained model artifacts from disk."""
    path = Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model artifact at '{path}'. Run scripts/train_artifact.py to generate it."
        )
    try:
        # Artifact files are generated locally by this project and include Python objects.
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # Backward compatibility for older torch versions without weights_only.
        return torch.load(path, map_location="cpu")


def _team_label(row: pd.Series) -> str:
    """Build readable team labels for selectors."""
    parts: list[str] = []
    if "TEAM_ABBREVIATION" in row and pd.notna(row["TEAM_ABBREVIATION"]):
        parts.append(str(row["TEAM_ABBREVIATION"]))
    if "TEAM_NAME" in row and pd.notna(row["TEAM_NAME"]):
        parts.append(str(row["TEAM_NAME"]))

    return " - ".join(parts) if parts else "Unknown Team"


def _build_team_name_map(teams_df: pd.DataFrame) -> dict[int, str]:
    """Create TEAM_ID -> team label map for display tables."""
    name_map: dict[int, str] = {}
    for _, row in teams_df.iterrows():
        try:
            team_id = int(row["TEAM_ID"])
        except (TypeError, ValueError):
            continue
        name_map[team_id] = _team_label(row)
    return name_map


def _ensure_state_defaults() -> None:
    """Initialize required session state keys."""
    st.session_state.setdefault("data_loaded", False)
    st.session_state.setdefault("train_df", None)
    st.session_state.setdefault("current_players", None)
    st.session_state.setdefault("current_teams", None)
    st.session_state.setdefault("artifacts", None)
    st.session_state.setdefault("data_end_year", None)
    st.session_state.setdefault("init_error", None)
    st.session_state.setdefault("test_eval", None)
    st.session_state.setdefault("test_eval_error", None)
    st.session_state.setdefault("latest_predictions", None)
    st.session_state.setdefault("last_matchup_key", None)


_ensure_state_defaults()

with st.sidebar:
    st.header("Setup")
    st.caption("Data and model are prepared automatically.")
    st.caption(f"Data window: {DATA_START_YEAR} to present")
    st.caption(f"Fixed split date: {TRAIN_TEST_SPLIT_DATE.date()}")

if not st.session_state.data_loaded or st.session_state.artifacts is None:
    try:
        with st.spinner("Loading data and model artifacts..."):
            train_df, current_players, current_teams, data_end_year = load_datasets()
            artifacts = load_pretrained_artifacts(str(MODEL_ARTIFACT_PATH))

        st.session_state.train_df = train_df
        st.session_state.current_players = current_players
        st.session_state.current_teams = current_teams
        st.session_state.artifacts = artifacts
        st.session_state.data_end_year = int(data_end_year)
        st.session_state.latest_predictions = None
        st.session_state.last_matchup_key = None
        st.session_state.data_loaded = True
        st.session_state.init_error = None

        try:
            st.session_state.test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)
            st.session_state.test_eval_error = None
        except Exception as eval_exc:
            st.session_state.test_eval = None
            st.session_state.test_eval_error = str(eval_exc)
    except Exception as exc:
        st.session_state.init_error = str(exc)
        st.session_state.data_loaded = False

if not st.session_state.data_loaded:
    st.error(
        f"Startup initialization failed: {st.session_state.init_error}"
    )
    st.info("Generate the artifact with: python scripts/train_artifact.py")
    st.stop()

train_df: pd.DataFrame = st.session_state.train_df
current_players: pd.DataFrame = st.session_state.current_players
current_teams: pd.DataFrame = st.session_state.current_teams
artifacts = st.session_state.artifacts

if st.session_state.test_eval is None and st.session_state.test_eval_error is None:
    try:
        with st.spinner("Computing test diagnostics..."):
            st.session_state.test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)
    except Exception as eval_exc:
        st.session_state.test_eval_error = str(eval_exc)

col_a, col_b, col_c = st.columns(3)
col_a.metric("Training Rows", f"{len(train_df):,}")
col_b.metric("Current Players", f"{len(current_players):,}")
col_c.metric("Current Teams", f"{len(current_teams):,}")

st.divider()
st.subheader("Model Setup")
summary = model_summary(artifacts)
info_1, info_2, info_3 = st.columns(3)
info_1.metric("Artifact Split Date", summary["train_end_date"])
info_2.metric("Artifact Train Rows", f"{int(summary['train_rows']):,}")
info_3.metric("Artifact Test Rows", f"{int(summary['test_rows']):,}")

st.caption(
    f"Data seasons fetched: {DATA_START_YEAR}-{st.session_state.data_end_year - 1} | "
    f"Artifact split date: {TRAIN_TEST_SPLIT_DATE.date()}"
)

if summary["train_end_date"] != str(TRAIN_TEST_SPLIT_DATE.date()):
    st.warning(
        "Loaded artifact was trained with a different split date. "
        f"Expected {TRAIN_TEST_SPLIT_DATE.date()}, got {summary['train_end_date']}."
    )

st.divider()
st.subheader("Model Evaluation Diagnostics")

if st.session_state.test_eval_error:
    st.warning(f"Could not compute test diagnostics: {st.session_state.test_eval_error}")

if st.session_state.test_eval is not None:
    test_eval = st.session_state.test_eval

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (q50)", f"{test_eval.summary['mae_q50']:.3f}")
    m2.metric("RMSE (q50)", f"{test_eval.summary['rmse_q50']:.3f}")
    m3.metric("R2 (q50)", f"{test_eval.summary['r2_q50']:.3f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("Test Rows", f"{int(test_eval.summary['test_rows']):,}")
    m5.metric("Avg Width (q10-q90)", f"{test_eval.summary['interval_width_q10_q90']:.3f}")
    m6.metric("Coverage (q10-q90)", f"{test_eval.summary['interval_coverage_q10_q90']:.3f}")

    st.markdown("#### Empirical Coverage Calibration")
    calibration_df = test_eval.quantile_metrics[["nominal_quantile", "empirical_coverage"]].copy()
    calibration_df = calibration_df.sort_values("nominal_quantile").reset_index(drop=True)

    fig_calibration = go.Figure()
    fig_calibration.add_trace(
        go.Scatter(
            x=calibration_df["nominal_quantile"],
            y=calibration_df["empirical_coverage"],
            mode="lines+markers",
            name="Empirical",
        )
    )
    fig_calibration.add_trace(
        go.Scatter(
            x=calibration_df["nominal_quantile"],
            y=calibration_df["nominal_quantile"],
            mode="lines",
            name="Ideal",
            line={"dash": "dash"},
        )
    )
    fig_calibration.update_layout(
        xaxis_title="Nominal quantile",
        yaxis_title="Empirical coverage",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    st.plotly_chart(fig_calibration, use_container_width=True)

    st.markdown("#### Interval Width vs Total Games In Dataset")
    player_profile = test_eval.player_interval_profile.copy()
    if not player_profile.empty:
        fig_profile = px.scatter(
            player_profile,
            x="total_games_in_dataset",
            y="mean_interval_width_q10_q90",
            hover_data=["PLAYER_NAME", "outlier_rate", "test_rows"],
            labels={
                "total_games_in_dataset": "Total games in dataset",
                "mean_interval_width_q10_q90": "Mean interval width (q10-q90)",
            },
        )
        fig_profile.update_traces(marker={"size": 10, "opacity": 0.7})
        fig_profile.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
        st.plotly_chart(fig_profile, use_container_width=True)
    else:
        st.info("Player interval profile is unavailable.")

    st.markdown("#### Performance by Player Data Volume")
    st.dataframe(test_eval.games_bucket_metrics, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Predict Matchup")

teams_df = team_lookup(current_teams)
if teams_df.empty:
    st.error("No teams available for selection.")
    st.stop()

teams_df = teams_df.copy()
teams_df["label"] = teams_df.apply(_team_label, axis=1)
team_name_map = _build_team_name_map(teams_df)
team_ids = teams_df["TEAM_ID"].astype(int).tolist()

col_home, col_away, col_game = st.columns([2, 2, 1])
with col_home:
    home_team_id = st.selectbox(
        "Team Name (Home)",
        options=team_ids,
        index=0,
        format_func=lambda tid: team_name_map.get(int(tid), "Unknown Team"),
    )
with col_away:
    away_candidates = [tid for tid in team_ids if tid != int(home_team_id)]
    away_team_id = st.selectbox(
        "Team Name (Away)",
        options=away_candidates,
        index=0,
        format_func=lambda tid: team_name_map.get(int(tid), "Unknown Team"),
    )
with col_game:
    is_playoff = st.toggle("Playoffs", value=False)

current_matchup_key = (int(home_team_id), int(away_team_id), bool(is_playoff))
if st.session_state.last_matchup_key != current_matchup_key:
    # Prevent stale prediction tables from previous team selections.
    st.session_state.latest_predictions = None
    st.session_state.last_matchup_key = current_matchup_key

try:
    home_roster, away_roster = get_matchup_rosters(
        current_players=current_players,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        enforce_official_roster=True,
    )

    st.markdown("#### Official Roster Preview")
    rp1, rp2 = st.columns(2)
    with rp1:
        st.caption(f"{team_name_map.get(int(home_team_id), 'Home Team')} ({len(home_roster)} players)")
        st.dataframe(home_roster[["PLAYER_NAME"]], use_container_width=True, hide_index=True)
    with rp2:
        st.caption(f"{team_name_map.get(int(away_team_id), 'Away Team')} ({len(away_roster)} players)")
        st.dataframe(away_roster[["PLAYER_NAME"]], use_container_width=True, hide_index=True)
except Exception as exc:
    st.warning(f"Could not load roster preview: {exc}")

if st.button("Calculate Predictions", type="primary"):
    try:
        with st.spinner("Calculating player predictions..."):
            output = predict_matchup(
                artifacts=artifacts,
                current_players=current_players,
                current_teams=current_teams,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                is_playoff=is_playoff,
                enforce_official_roster=True,
            )
        st.session_state.latest_predictions = output
    except Exception as exc:
        st.session_state.latest_predictions = None
        st.error(f"Prediction failed: {exc}")

if st.session_state.latest_predictions is not None:
    prediction_df = st.session_state.latest_predictions.copy()

    home_preds = prediction_df[prediction_df["TEAM_ID"] == int(home_team_id)].copy()
    away_preds = prediction_df[prediction_df["TEAM_ID"] == int(away_team_id)].copy()

    quantile_cols = [col for col in prediction_df.columns if col.startswith("q")]
    show_cols = ["PLAYER_NAME", *quantile_cols]

    st.markdown(f"### {team_name_map.get(int(home_team_id), 'Home Team')}")
    st.dataframe(home_preds[show_cols], use_container_width=True, hide_index=True)

    st.markdown(f"### {team_name_map.get(int(away_team_id), 'Away Team')}")
    st.dataframe(away_preds[show_cols], use_container_width=True, hide_index=True)
