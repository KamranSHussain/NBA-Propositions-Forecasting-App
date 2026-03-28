"""Streamlit app for NBA player prop quantile predictions."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data import DEFAULT_END_YEAR, DEFAULT_START_YEAR, get_nba_data
from src.service import (
    evaluate_test_set,
    get_matchup_rosters,
    predict_matchup,
    team_lookup,
    train_model,
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

st.caption("Train by date split, then predict player quantiles for a selected matchup.")

MIN_DATA_YEAR = 2010
MAX_DATA_YEAR = 2026


@st.cache_data(show_spinner=False)
def load_datasets(start_year: int, end_year: int):
    """Load and cache processed training + inference datasets."""
    return get_nba_data(start_year=start_year, end_year=end_year)


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
    st.session_state.setdefault("test_eval", None)
    st.session_state.setdefault("latest_predictions", None)
    st.session_state.setdefault("last_matchup_key", None)


_ensure_state_defaults()

with st.sidebar:
    st.header("Data")
    start_year = st.number_input(
        "Start Year",
        min_value=MIN_DATA_YEAR,
        max_value=MAX_DATA_YEAR - 1,
        value=max(MIN_DATA_YEAR, min(DEFAULT_START_YEAR, MAX_DATA_YEAR - 1)),
        step=1,
    )
    end_year = st.number_input(
        "End Year (exclusive)",
        min_value=int(start_year) + 1,
        max_value=MAX_DATA_YEAR,
        value=min(MAX_DATA_YEAR, max(int(start_year) + 1, DEFAULT_END_YEAR)),
        step=1,
    )

    if int(end_year) <= int(start_year):
        st.error("End year must be greater than start year.")

    if st.button("Load Data", type="primary"):
        if int(end_year) <= int(start_year):
            st.error("Fix year range before loading data.")
        else:
            try:
                with st.spinner("Fetching and processing NBA data..."):
                    train_df, current_players, current_teams = load_datasets(
                        start_year=int(start_year),
                        end_year=int(end_year),
                    )
                st.session_state.train_df = train_df
                st.session_state.current_players = current_players
                st.session_state.current_teams = current_teams
                st.session_state.artifacts = None
                st.session_state.test_eval = None
                st.session_state.latest_predictions = None
                st.session_state.last_matchup_key = None
                st.session_state.data_loaded = True
                st.success("Data loaded.")
            except Exception as exc:
                st.error(f"Data load failed: {exc}")

if not st.session_state.data_loaded:
    st.info("Load data from the sidebar to begin training.")
    st.stop()

train_df: pd.DataFrame = st.session_state.train_df
current_players: pd.DataFrame = st.session_state.current_players
current_teams: pd.DataFrame = st.session_state.current_teams

col_a, col_b, col_c = st.columns(3)
col_a.metric("Training Rows", f"{len(train_df):,}")
col_b.metric("Current Players", f"{len(current_players):,}")
col_c.metric("Current Teams", f"{len(current_teams):,}")

st.divider()
st.subheader("1. Train Model")

sorted_dates = pd.to_datetime(train_df["GAME_DATE"]).sort_values().reset_index(drop=True)
if len(sorted_dates) < 2:
    st.error("Not enough rows to create a train/test split.")
    st.stop()

left, right = st.columns([2, 1])
with left:
    train_split_pct = st.slider("Train Split (%)", min_value=10, max_value=90, value=75, step=1)

    split_idx = int(len(sorted_dates) * (train_split_pct / 100.0)) - 1
    split_idx = max(0, min(len(sorted_dates) - 2, split_idx))
    split_date = pd.Timestamp(sorted_dates.iloc[split_idx]).date()

    est_train_rows = int((pd.to_datetime(train_df["GAME_DATE"]) <= pd.Timestamp(split_date)).sum())
    est_test_rows = int((pd.to_datetime(train_df["GAME_DATE"]) > pd.Timestamp(split_date)).sum())
    st.caption(
        f"Split date: {split_date} | Train rows: {est_train_rows:,} | Test rows: {est_test_rows:,}"
    )
with right:
    with st.expander("Training Params", expanded=False):
        epochs = st.slider("Epochs", min_value=10, max_value=200, value=75, step=5)
        batch_size = st.selectbox("Batch Size", options=[64, 128, 256, 512], index=2)
        learning_rate = st.selectbox("Learning Rate", options=[1e-4, 5e-4, 1e-3, 2e-3], index=2)

if st.button("Train", type="primary"):
    try:
        with st.spinner("Training quantile model..."):
            artifacts = train_model(
                df=train_df,
                split_date=split_date,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
            )
            test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)
        st.session_state.artifacts = artifacts
        st.session_state.test_eval = test_eval
        st.success("Training complete.")
    except Exception as exc:
        st.session_state.artifacts = None
        st.session_state.test_eval = None
        st.error(f"Training failed: {exc}")

if st.session_state.test_eval is not None:
    st.subheader("Test Set Evaluation")
    test_eval = st.session_state.test_eval

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (q50)", f"{test_eval.summary['mae_q50']:.3f}")
    m2.metric("RMSE (q50)", f"{test_eval.summary['rmse_q50']:.3f}")
    m3.metric("R2 (q50)", f"{test_eval.summary['r2_q50']:.3f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("Test Rows", f"{int(test_eval.summary['test_rows']):,}")
    m5.metric("Avg Width (q10-q90)", f"{test_eval.summary['interval_width_q10_q90']:.3f}")
    m6.metric("Coverage (q10-q90)", f"{test_eval.summary['interval_coverage_q10_q90']:.3f}")

    st.markdown("#### Quantile Diagnostics")
    st.dataframe(test_eval.quantile_metrics, use_container_width=True, hide_index=True)

    if {"actual", "q50"}.issubset(test_eval.predictions.columns):
        chart_df = test_eval.predictions[["actual", "q50"]].copy().head(2000)
        chart_df["actual"] = pd.to_numeric(chart_df["actual"], errors="coerce")
        chart_df["q50"] = pd.to_numeric(chart_df["q50"], errors="coerce")
        chart_df = chart_df.dropna(subset=["actual", "q50"]).reset_index(drop=True)

        st.markdown("#### Actual vs q50 (sample)")
        st.line_chart(chart_df, y=["actual", "q50"])

    with st.expander("Show Test Predictions Sample", expanded=False):
        sample_df = test_eval.predictions.copy()
        team_name_map = _build_team_name_map(current_teams)
        sample_df["team_name"] = sample_df["TEAM_ID"].map(team_name_map).fillna("Unknown Team")

        quantile_cols = [col for col in sample_df.columns if col.startswith("q")]
        display_cols = ["GAME_DATE", "PLAYER_NAME", "team_name", "actual", *quantile_cols]
        display_cols = [col for col in display_cols if col in sample_df.columns]
        sample_df = sample_df[display_cols].copy()

        n_rows = min(200, len(sample_df))
        if n_rows > 0:
            sample_df = sample_df.sample(n=n_rows, random_state=42).sort_values(by="GAME_DATE")
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

st.divider()
st.subheader("2. Predict Matchup")

if st.session_state.artifacts is None:
    st.warning("Train the model first to enable matchup predictions.")
    st.stop()

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
                artifacts=st.session_state.artifacts,
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
