"""Streamlit app for NBA player prop quantile predictions."""

from __future__ import annotations

import re
import string
import unicodedata
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from nba_api.stats.endpoints import leaguegamefinder

from src.data import get_nba_data
from src.fanduel_live import fetch_fanduel_live_player_points
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
MODEL_ARTIFACT_PATH = Path("models/player_prop_artifacts_opp28.pt")
BACKTEST_EVAL_PATH = Path("betting data/backtests/partner_odds_backtest.csv")
RECOMMENDER_EDGE_MAX = -4.0
RECOMMENDER_MIN_DECIMAL_ODDS = 1.81
LIVE_REGION = "NY"
LIVE_MAX_EVENTS = 8


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


def _normalize_team_name(value: object) -> str:
    """Normalize team labels for fuzzy team-id resolution."""
    if value is None or pd.isna(value):
        return ""
    cleaned = unicodedata.normalize("NFKD", str(value))
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = cleaned.lower()
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _build_team_resolution_map(teams_df: pd.DataFrame) -> dict[str, int]:
    """Build normalized team label -> TEAM_ID lookup."""
    mapping: dict[str, int] = {}
    for _, row in teams_df.iterrows():
        try:
            team_id = int(row.get("TEAM_ID"))
        except (TypeError, ValueError):
            continue
        for col in ["TEAM_ABBREVIATION", "TEAM_NAME", "NICKNAME", "CITY", "label"]:
            if col not in row or pd.isna(row[col]):
                continue
            key = _normalize_team_name(row[col])
            if key:
                mapping[key] = team_id
    return mapping


def _resolve_team_id(team_name: object, mapping: dict[str, int]) -> int | None:
    """Resolve team id from FanDuel team labels."""
    key = _normalize_team_name(team_name)
    if not key:
        return None
    if key in mapping:
        return mapping[key]
    for known, team_id in mapping.items():
        if key in known or known in key:
            return team_id
    return None


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_betting_lines_csv(csv_path: str) -> pd.DataFrame:
    """Load sampled betting lines CSV."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing betting lines file at '{path}'.")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_backtest_eval_csv(csv_path: str, mtime: float | None = None) -> pd.DataFrame:
    """Load backtest evaluation rows generated by scripts/backtest_partner_odds.py."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing backtest evaluation file at '{path}'.")
    return pd.read_csv(path)


@st.cache_data(show_spinner=False, ttl=90)
def load_fanduel_live_lines(region: str, max_events: int) -> pd.DataFrame:
    """Load live FanDuel NBA player points lines."""
    return fetch_fanduel_live_player_points(region=region, max_events=max_events)


def _build_team_abbreviation_map(teams_df: pd.DataFrame) -> dict[str, int]:
    """Build TEAM_ABBREVIATION -> TEAM_ID map."""
    if "TEAM_ABBREVIATION" not in teams_df.columns:
        return {}
    mapping: dict[str, int] = {}
    rows = teams_df[["TEAM_ID", "TEAM_ABBREVIATION"]].dropna().drop_duplicates()
    for _, row in rows.iterrows():
        try:
            mapping[str(row["TEAM_ABBREVIATION"]).strip().upper()] = int(row["TEAM_ID"])
        except (TypeError, ValueError):
            continue
    return mapping


def _sorted_quantile_columns(df: pd.DataFrame) -> list[str]:
    """Return quantile columns like q10/q50/q90 sorted by numeric suffix."""
    quant_cols = []
    for col in df.columns:
        if not isinstance(col, str) or not col.startswith("q"):
            continue
        try:
            quantile_value = float(col[1:])
        except (TypeError, ValueError):
            continue
        quant_cols.append((quantile_value, col))
    quant_cols.sort(key=lambda x: x[0])
    return [col for _, col in quant_cols]


def _decimal_series_to_american(decimal_odds: pd.Series) -> pd.Series:
    """Convert decimal odds to American odds for display."""
    decimal = pd.to_numeric(decimal_odds, errors="coerce")
    american = pd.Series(pd.NA, index=decimal.index, dtype="Float64")
    valid = decimal > 1
    favored = valid & (decimal < 2)
    dog = valid & (decimal >= 2)
    american.loc[favored] = -100.0 / (decimal.loc[favored] - 1.0)
    american.loc[dog] = (decimal.loc[dog] - 1.0) * 100.0
    return american


def _format_american_odds(value: object) -> str:
    """Format a numeric American-odds value with sign and rounding."""
    if pd.isna(value):
        return "N/A"
    rounded = int(round(float(value)))
    return f"+{rounded}" if rounded > 0 else str(rounded)


@st.cache_data(show_spinner=False, ttl=60 * 30)
def load_completed_player_stats_for_range(date_from_iso: str, date_to_iso: str) -> pd.DataFrame:
    """Fetch completed NBA player stats for a date range via nba_api."""
    empty = pd.DataFrame(columns=["game_day", "team", "player_key", "actual_value", "GAME_ID"])

    finder = leaguegamefinder.LeagueGameFinder(
        date_from_nullable=date_from_iso,
        date_to_nullable=date_to_iso,
        player_or_team_abbreviation="P",
    )
    frames = finder.get_data_frames()
    if not frames:
        return empty

    df = frames[0].copy()
    required = {"GAME_DATE", "TEAM_ABBREVIATION", "PLAYER_NAME", "PTS"}
    if not required.issubset(df.columns):
        return empty

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df["game_day"] = df["GAME_DATE"].dt.normalize()
    df["team"] = df["TEAM_ABBREVIATION"].astype(str).str.strip().str.upper()
    df["player_key"] = df["PLAYER_NAME"].map(_normalize_player_name)
    df["actual_value"] = pd.to_numeric(df["PTS"], errors="coerce")

    result = df[["game_day", "team", "player_key", "actual_value", "GAME_ID"]].copy()
    result = result.dropna(subset=["game_day", "actual_value"])
    result = result[result["player_key"].ne("")]
    result = result.drop_duplicates(subset=["game_day", "team", "player_key"], keep="first")
    return result


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
    st.session_state.setdefault("betting_preds_cache", {})
    st.session_state.setdefault("startup_live_lines", None)
    st.session_state.setdefault("startup_live_lines_error", None)


_ensure_state_defaults()

with st.sidebar:
    st.header("Setup")
    st.caption("Data and model are prepared automatically.")
    st.caption(f"Data window: {DATA_START_YEAR} to present")
    st.caption(f"Fixed split date: {TRAIN_TEST_SPLIT_DATE.date()}")
    page = st.radio("Page", options=["Predict Matchup", "Betting Lines", "Test Stats"], index=0)

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

        # Pull live lines one time at startup; reuse this snapshot across reruns.
        try:
            st.session_state.startup_live_lines = load_fanduel_live_lines(
                region=LIVE_REGION,
                max_events=LIVE_MAX_EVENTS,
            ).copy()
            st.session_state.startup_live_lines_error = None
        except Exception as live_exc:
            st.session_state.startup_live_lines = None
            st.session_state.startup_live_lines_error = str(live_exc)
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

summary = model_summary(artifacts)

if page == "Betting Lines":
    st.divider()
    st.subheader("Betting Lines Command Center")

    if st.session_state.startup_live_lines_error:
        st.error(f"Could not load startup FanDuel lines: {st.session_state.startup_live_lines_error}")
        st.stop()

    live_df = (
        st.session_state.startup_live_lines.copy()
        if isinstance(st.session_state.startup_live_lines, pd.DataFrame)
        else pd.DataFrame()
    )

    if live_df.empty:
        st.warning("No live FanDuel NBA player points lines were found.")
        st.stop()

    live_df["line"] = pd.to_numeric(live_df.get("line"), errors="coerce")
    live_df["over_odds"] = pd.to_numeric(live_df.get("over_odds"), errors="coerce")
    live_df["under_odds"] = pd.to_numeric(live_df.get("under_odds"), errors="coerce")
    live_df["game_date"] = pd.to_datetime(live_df.get("game_date"), errors="coerce")
    live_df["player_name"] = live_df["player_name"].astype(str).str.strip()

    teams_lookup = team_lookup(current_teams).copy()
    if teams_lookup.empty:
        st.error("Unable to resolve current NBA team mappings.")
        st.stop()
    teams_lookup["label"] = teams_lookup.apply(_team_label, axis=1)
    resolution_map = _build_team_resolution_map(teams_lookup)

    live_df["home_team_id"] = live_df["home_team"].map(lambda value: _resolve_team_id(value, resolution_map))
    live_df["away_team_id"] = live_df["away_team"].map(lambda value: _resolve_team_id(value, resolution_map))

    pred_frames: list[pd.DataFrame] = []
    unique_matchups = (
        live_df[["home_team_id", "away_team_id"]]
        .dropna()
        .drop_duplicates()
        .astype(int)
        .itertuples(index=False, name=None)
    )
    for home_team_id, away_team_id in unique_matchups:
        try:
            matchup_pred = predict_matchup(
                artifacts=artifacts,
                current_players=current_players,
                current_teams=current_teams,
                history_df=train_df,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                is_playoff=False,
                enforce_official_roster=True,
            )
        except Exception:
            continue
        if matchup_pred is None or matchup_pred.empty:
            continue
        frame = matchup_pred.copy()
        frame["player_key"] = frame["PLAYER_NAME"].map(_normalize_player_name)
        pred_frames.append(frame)

    if not pred_frames:
        st.error("Model projections could not be generated for live matchups.")
        st.stop()

    preds = pd.concat(pred_frames, ignore_index=True)
    preds = preds.drop_duplicates(subset=["TEAM_ID", "player_key"], keep="first")
    quantile_cols = _sorted_quantile_columns(preds)
    if "q50" not in preds.columns:
        st.error("Model output is missing q50 predictions.")
        st.stop()

    player_team_map: dict[tuple[str, int], str] = {}
    for _, row in preds.iterrows():
        key = (_normalize_player_name(row.get("PLAYER_NAME")), int(row.get("TEAM_ID")))
        player_team_map[key] = str(row.get("PLAYER_NAME"))

    enriched = live_df.copy()
    enriched["player_key"] = enriched["player_name"].map(_normalize_player_name)

    team_assignments: list[int | None] = []
    display_team: list[str] = []
    display_opp: list[str] = []
    team_side: list[str] = []
    for _, row in enriched.iterrows():
        home_id = row.get("home_team_id")
        away_id = row.get("away_team_id")
        home_name = str(row.get("home_team") or "")
        away_name = str(row.get("away_team") or "")
        home_match = pd.notna(home_id) and (row["player_key"], int(home_id)) in player_team_map
        away_match = pd.notna(away_id) and (row["player_key"], int(away_id)) in player_team_map

        if home_match and away_match:
            team_assignments.append(int(home_id))
            display_team.append(home_name)
            display_opp.append(away_name)
            team_side.append("home")
        elif home_match:
            team_assignments.append(int(home_id))
            display_team.append(home_name)
            display_opp.append(away_name)
            team_side.append("home")
        elif away_match:
            team_assignments.append(int(away_id))
            display_team.append(away_name)
            display_opp.append(home_name)
            team_side.append("away")
        else:
            team_assignments.append(None)
            display_team.append("")
            display_opp.append("")
            team_side.append("unmatched")

    enriched["team_id"] = team_assignments
    enriched["team"] = display_team
    enriched["opponent"] = display_opp
    enriched["team_side"] = team_side

    pred_cols = ["TEAM_ID", "player_key", *[col for col in quantile_cols if col in preds.columns]]
    merged = enriched.merge(
        preds[pred_cols],
        left_on=["team_id", "player_key"],
        right_on=["TEAM_ID", "player_key"],
        how="left",
    )

    merged["q50"] = pd.to_numeric(merged["q50"], errors="coerce")
    merged["edge"] = merged["line"] - merged["q50"]
    merged["model_recommendation"] = "pending"
    over_mask = merged["q50"].notna() & merged["line"].notna() & (merged["q50"] > merged["line"])
    under_mask = merged["q50"].notna() & merged["line"].notna() & (merged["q50"] <= merged["line"])
    merged.loc[over_mask, "model_recommendation"] = "over"
    merged.loc[under_mask, "model_recommendation"] = "under"

    merged["pick_odds"] = merged["over_odds"].where(merged["model_recommendation"] == "over", merged["under_odds"])
    merged.loc[merged["model_recommendation"] == "pending", "pick_odds"] = pd.NA
    merged["pick_odds_american"] = _decimal_series_to_american(merged["pick_odds"])
    merged["recommender_pick"] = (
        merged["edge"].le(RECOMMENDER_EDGE_MAX)
        & merged["model_recommendation"].isin(["over", "under"])
        & merged["pick_odds"].ge(RECOMMENDER_MIN_DECIMAL_ODDS)
    )
    merged["recommendation_tier"] = 0
    merged.loc[
        merged["model_recommendation"].isin(["over", "under"]) & merged["pick_odds"].notna(),
        "recommendation_tier",
    ] = 1
    merged.loc[merged["recommender_pick"], "recommendation_tier"] = 2
    # Only strict recommender-qualified bets are considered recommendations.
    merged["is_recommended"] = merged["recommender_pick"]
    merged["status"] = "live"
    merged["confidence"] = merged["edge"].abs()

    valid_rows = merged[
        merged["line"].notna() & merged["over_odds"].notna() & merged["under_odds"].notna() & merged["player_name"].ne("")
    ].copy()
    if valid_rows.empty:
        st.warning("Live rows were fetched, but no valid player points lines were parseable.")
        st.stop()

    projected = valid_rows[valid_rows["q50"].notna()].copy()
    recommended = projected[projected["is_recommended"]].copy()

    m1, m3 = st.columns(2)
    m1.metric("Live FanDuel Lines", f"{len(valid_rows):,}")
    m3.metric("Recommended Bets", f"{len(recommended):,}")

    sort_spec = [
        ("recommendation_tier", False),
        ("recommender_pick", False),
        ("confidence", False),
        ("game_date", True),
    ]
    sort_cols = [col for col, _ in sort_spec if col in projected.columns]
    sort_asc = [asc for col, asc in sort_spec if col in projected.columns]

    card_source = recommended if not recommended.empty else projected
    card_df = card_source.sort_values(sort_cols, ascending=sort_asc).head(12) if sort_cols else card_source.head(12)
    st.markdown("#### Live FanDuel Pick Board")
    if card_df.empty:
        st.info("No live rows could be matched to model projections yet.")
    else:
        quant_cols = _sorted_quantile_columns(projected)
        low_col = quant_cols[0] if len(quant_cols) >= 2 else None
        high_col = quant_cols[-1] if len(quant_cols) >= 2 else None
        cards_html: list[str] = []
        for _, row in card_df.iterrows():
            border = "#16a34a" if bool(row.get("is_recommended")) else "#f97316"
            badge_bg = "rgba(22, 163, 74, 0.2)" if bool(row.get("is_recommended")) else "rgba(249, 115, 22, 0.2)"
            badge_txt = "RECOMMENDED" if bool(row.get("is_recommended")) else "LIVE"
            pick = str(row.get("model_recommendation", "pending")).upper()
            game_label = (
                f"{pd.to_datetime(row.get('game_date')).strftime('%b %d %I:%M %p')} UTC"
                if pd.notna(row.get("game_date"))
                else "TBD"
            )
            interval_txt = "N/A"
            if low_col and high_col and pd.notna(row.get(low_col)) and pd.notna(row.get(high_col)):
                interval_txt = f"{float(row.get(low_col)):.1f} - {float(row.get(high_col)):.1f}"
            cards_html.append(
                f"""
<div style="
    border: 1px solid {border};
    border-left: 5px solid {border};
    border-radius: 12px;
    padding: 0.75rem 0.85rem;
    margin-bottom: 0.8rem;
    background: linear-gradient(180deg, rgba(15,23,42,0.9), rgba(17,27,47,0.92));
        min-height: 210px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-sizing: border-box;
">
    <div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:0.6rem;">
            <div style="font-weight:700;color:#f8fafc;">{row.get("player_name", "Unknown Player")}</div>
        <div style="font-size:0.72rem;padding:0.12rem 0.4rem;border-radius:999px;background:{badge_bg};color:#e2e8f0;">{badge_txt}</div>
        </div>
        <div style="color:#94a3b8;font-size:0.82rem;">{game_label} | {row.get("team", "?")} vs {row.get("opponent", "?")}</div>
        <div style="margin-top:0.4rem;display:grid;grid-template-columns:1fr 1fr 1fr;gap:0.35rem;font-size:0.82rem;">
            <div><span style="color:#94a3b8;">Line</span><br><b style="color:#f1f5f9;">{float(row.get("line")):.1f}</b></div>
            <div><span style="color:#94a3b8;">Model q50</span><br><b style="color:#f1f5f9;">{float(row.get("q50")):.1f}</b></div>
            <div><span style="color:#94a3b8;">Edge</span><br><b style="color:#f1f5f9;">{float(row.get("edge")):+.1f}</b></div>
        </div>
        <div style="margin-top:0.35rem;font-size:0.78rem;color:#cbd5e1;">Interval: <b>{interval_txt}</b></div>
  </div>
  <div style="margin-top:0.45rem;display:flex;justify-content:space-between;align-items:center;">
    <div style="font-size:0.76rem;color:#cbd5e1;">Pick: <b>{pick}</b></div>
    <div style="font-size:0.76rem;color:#cbd5e1;">Odds: <b>{_format_american_odds(row.get("pick_odds_american"))}</b></div>
  </div>
</div>
"""
            )

        for start in range(0, len(cards_html), 3):
            row_cols = st.columns(3)
            row_cards = cards_html[start : start + 3]
            for col_idx, card_html in enumerate(row_cards):
                row_cols[col_idx].markdown(card_html, unsafe_allow_html=True)

    display_cols = [
        "game_date",
        "player_name",
        "team",
        "opponent",
        "line",
        "q10",
        "q50",
        "q90",
        "edge",
        "model_recommendation",
        "pick_odds_american",
        "recommender_pick",
    ]
    with st.expander("Detailed Live Lines Table", expanded=False):
        detailed_df = projected[[col for col in display_cols if col in projected.columns]]
        detailed_sort_cols = [col for col in sort_cols if col in detailed_df.columns]
        detailed_sort_asc = [asc for col, asc in zip(sort_cols, sort_asc) if col in detailed_df.columns]
        if detailed_sort_cols:
            detailed_df = detailed_df.sort_values(detailed_sort_cols, ascending=detailed_sort_asc)
        st.dataframe(
            detailed_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Historical Betting Lines Evaluation")
    st.caption("Historical backtest charts shown below live picks for recommendation context.")

    if not BACKTEST_EVAL_PATH.exists():
        st.info(f"Backtest CSV not found at {BACKTEST_EVAL_PATH}.")
    else:
        try:
            backtest_df = load_backtest_eval_csv(
                str(BACKTEST_EVAL_PATH),
                mtime=BACKTEST_EVAL_PATH.stat().st_mtime,
            ).copy()
        except Exception as exc:
            st.warning(f"Could not load backtest CSV: {exc}")
        else:
            if backtest_df.empty:
                st.info("Backtest CSV is empty.")
            else:
                backtest_df["status"] = backtest_df.get("status", pd.Series(dtype="object")).astype(str).str.lower()
                backtest_df["edge"] = pd.to_numeric(backtest_df.get("edge"), errors="coerce")
                backtest_df["bookmaker"] = backtest_df.get("bookmaker", pd.Series(dtype="object")).astype(str)
                backtest_df["model_recommendation"] = (
                    backtest_df.get("model_recommendation", pd.Series(dtype="object")).astype(str).str.lower()
                )
                backtest_df["over_odds"] = pd.to_numeric(backtest_df.get("over_odds"), errors="coerce")
                backtest_df["under_odds"] = pd.to_numeric(backtest_df.get("under_odds"), errors="coerce")
                backtest_df["is_correct"] = backtest_df["status"].eq("correct")
                backtest_df["game_date"] = pd.to_datetime(backtest_df.get("game_date"), errors="coerce")

                backtest_df["pick_odds"] = backtest_df["over_odds"].where(
                    backtest_df["model_recommendation"].eq("over"),
                    backtest_df["under_odds"],
                )
                backtest_df.loc[
                    ~backtest_df["model_recommendation"].isin(["over", "under"]),
                    "pick_odds",
                ] = pd.NA

                recommender_hist = backtest_df[
                    backtest_df["edge"].le(RECOMMENDER_EDGE_MAX)
                    & backtest_df["model_recommendation"].isin(["over", "under"])
                    & backtest_df["pick_odds"].ge(RECOMMENDER_MIN_DECIMAL_ODDS)
                ].copy()

                if recommender_hist.empty:
                    st.info("No historical rows match the recommender rule yet.")
                else:
                    total_recs = int(len(recommender_hist))
                    correct_recs = int(recommender_hist["is_correct"].sum())
                    rec_accuracy = (correct_recs / total_recs) if total_recs > 0 else 0.0

                    r1, r2, r3 = st.columns(3)
                    r1.metric("Historical Recs", f"{total_recs:,}")
                    r2.metric("Correct Recs", f"{correct_recs:,}")
                    r3.metric("Rec Accuracy", f"{rec_accuracy:.1%}")

                    # Chart 1: Percent of recommendations that were correct.
                    pct_df = pd.DataFrame(
                        {
                            "result": ["Correct", "Incorrect"],
                            "count": [correct_recs, max(total_recs - correct_recs, 0)],
                        }
                    )
                    fig_pct = px.pie(
                        pct_df,
                        values="count",
                        names="result",
                        title="Percent of Recommendations Correct",
                        hole=0.55,
                    )
                    fig_pct.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10})
                    st.plotly_chart(fig_pct, use_container_width=True)

                    # Chart 2: Cumulative recommendation accuracy over time.
                    acc_time = recommender_hist.dropna(subset=["game_date"]).copy()
                    acc_time["week_start"] = acc_time["game_date"].dt.to_period("W-MON").dt.start_time
                    acc_time = (
                        acc_time.groupby("week_start", observed=False)["is_correct"]
                        .agg(correct="sum", picks="count")
                        .reset_index()
                        .sort_values("week_start")
                    )
                    if not acc_time.empty:
                        acc_time["cum_correct"] = acc_time["correct"].cumsum()
                        acc_time["cum_picks"] = acc_time["picks"].cumsum()
                        acc_time["cum_accuracy"] = acc_time["cum_correct"] / acc_time["cum_picks"]

                        fig_time = go.Figure()
                        fig_time.add_trace(
                            go.Scatter(
                                x=acc_time["week_start"],
                                y=acc_time["cum_accuracy"],
                                mode="lines+markers",
                                name="Cumulative accuracy",
                                customdata=acc_time[["cum_picks"]],
                                hovertemplate="Week: %{x|%Y-%m-%d}<br>Cumulative accuracy: %{y:.1%}<br>Cumulative picks: %{customdata[0]}<extra></extra>",
                            )
                        )
                        fig_time.update_layout(
                            title="Cumulative Recommendation Accuracy Over Time",
                            margin={"l": 10, "r": 10, "t": 40, "b": 10},
                            yaxis_title="Accuracy",
                            xaxis_title="Week",
                        )
                        fig_time.update_yaxes(tickformat=".0%")
                        st.plotly_chart(fig_time, use_container_width=True)

                    # Chart 3: Bankroll trajectory with flat staking on each recommendation.
                    bankroll_df = recommender_hist.dropna(subset=["game_date", "pick_odds"]).copy()
                    bankroll_df = bankroll_df[bankroll_df["pick_odds"].gt(1.0)].sort_values("game_date")
                    if not bankroll_df.empty:
                        bankroll_start = 100.0
                        stake_per_bet = 1.0
                        bankroll_df["bet_pnl"] = -stake_per_bet
                        bankroll_df.loc[
                            bankroll_df["is_correct"],
                            "bet_pnl",
                        ] = stake_per_bet * (bankroll_df["pick_odds"] - 1.0)
                        bankroll_df["date"] = bankroll_df["game_date"].dt.normalize()
                        daily_bankroll = (
                            bankroll_df.groupby("date", observed=False)
                            .agg(daily_pnl=("bet_pnl", "sum"), bets=("bet_pnl", "count"))
                            .reset_index()
                            .sort_values("date")
                        )
                        daily_bankroll["running_bankroll"] = bankroll_start + daily_bankroll["daily_pnl"].cumsum()
                        daily_bankroll["prev_bankroll"] = daily_bankroll["running_bankroll"].shift(1).fillna(bankroll_start)
                        daily_bankroll["prev_date"] = daily_bankroll["date"].shift(1)
                        daily_bankroll.loc[0, "prev_date"] = daily_bankroll.loc[0, "date"]

                        fig_bankroll = go.Figure()
                        showed_green_legend = False
                        showed_red_legend = False
                        for row in daily_bankroll.itertuples(index=False):
                            is_up_day = row.daily_pnl >= 0
                            fig_bankroll.add_trace(
                                go.Scatter(
                                    x=[row.prev_date, row.date],
                                    y=[row.prev_bankroll, row.running_bankroll],
                                    mode="lines",
                                    name="Green days" if is_up_day else "Red days",
                                    line={"color": "#22c55e" if is_up_day else "#ef4444", "width": 2},
                                    showlegend=not showed_green_legend if is_up_day else not showed_red_legend,
                                    hovertemplate="Date: %{x|%Y-%m-%d}<br>Bankroll: $%{y:.2f}<extra></extra>",
                                )
                            )
                            if is_up_day:
                                showed_green_legend = True
                            else:
                                showed_red_legend = True

                        marker_colors = ["#22c55e" if pnl >= 0 else "#ef4444" for pnl in daily_bankroll["daily_pnl"]]
                        fig_bankroll.add_trace(
                            go.Scatter(
                                x=daily_bankroll["date"],
                                y=daily_bankroll["running_bankroll"],
                                mode="markers",
                                name="Daily bankroll",
                                marker={"color": marker_colors, "size": 6},
                                customdata=daily_bankroll[["daily_pnl", "bets"]],
                                hovertemplate="Date: %{x|%Y-%m-%d}<br>Bankroll: $%{y:.2f}<br>Day P/L: $%{customdata[0]:.2f}<br>Bets: %{customdata[1]}<extra></extra>",
                            )
                        )
                        fig_bankroll.update_layout(
                            title="Cumulative Bankroll ($100 start, $1 per recommendation)",
                            margin={"l": 10, "r": 10, "t": 40, "b": 10},
                            yaxis_title="Bankroll ($)",
                            xaxis_title="Date",
                        )
                        st.plotly_chart(fig_bankroll, use_container_width=True)

                # Chart 4: Accuracy by edge bucket from <= -4 to >= 4.
                edge_chart_df = backtest_df.dropna(subset=["edge"]).copy()
                edge_bins = [-float("inf"), -4, -3, -2, -1, 0, 1, 2, 3, 4, float("inf")]
                edge_labels = ["<= -4", "-4 to -3", "-3 to -2", "-2 to -1", "-1 to 0", "0 to 1", "1 to 2", "2 to 3", "3 to 4", ">= 4"]
                edge_chart_df["edge_bucket"] = pd.cut(edge_chart_df["edge"], bins=edge_bins, labels=edge_labels)
                edge_acc = (
                    edge_chart_df.groupby("edge_bucket", observed=False)["is_correct"]
                    .agg(accuracy="mean", picks="count")
                    .reset_index()
                )
                fig_edge = px.bar(
                    edge_acc,
                    x="edge_bucket",
                    y="accuracy",
                    title="Accuracy by Edge (<= -4 to >= 4)",
                    hover_data=["picks"],
                    labels={"edge_bucket": "Edge Bucket", "accuracy": "Accuracy"},
                )
                fig_edge.update_yaxes(tickformat=".0%")
                fig_edge.update_layout(margin={"l": 10, "r": 10, "t": 40, "b": 10}, showlegend=False)
                st.plotly_chart(fig_edge, use_container_width=True)
    st.stop()

if page == "Test Stats":
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Training Rows", f"{len(train_df):,}")
    col_b.metric("Current Players", f"{len(current_players):,}")
    col_c.metric("Current Teams", f"{len(current_teams):,}")

    st.divider()
    st.subheader("Model Setup")
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

    if st.session_state.test_eval is None and st.session_state.test_eval_error is None:
        try:
            with st.spinner("Computing test diagnostics..."):
                st.session_state.test_eval = evaluate_test_set(df=train_df, artifacts=artifacts)
        except Exception as eval_exc:
            st.session_state.test_eval_error = str(eval_exc)

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

    st.stop()

st.divider()
st.subheader("Predict Matchup")
st.caption("Pick teams, run the model, and scan projections with visual summaries.")

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

home_label = team_name_map.get(int(home_team_id), "Home Team")
away_label = team_name_map.get(int(away_team_id), "Away Team")
playoff_label = "PLAYOFF GAME" if is_playoff else "REGULAR SEASON"
st.markdown(
    f"""
<div style="
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 0.85rem 1rem;
    margin: 0.35rem 0 0.8rem;
    background: linear-gradient(120deg, rgba(14,23,40,0.95), rgba(15,23,42,0.86));
">
  <div style="font-size:0.76rem;color:#94a3b8;letter-spacing:0.06em;">MATCHUP PREVIEW</div>
  <div style="display:flex;justify-content:space-between;align-items:center;gap:0.8rem;">
    <div style="font-size:1.35rem;font-family:'Teko',sans-serif;color:#f8fafc;">{home_label} vs {away_label}</div>
    <div style="font-size:0.75rem;padding:0.2rem 0.5rem;border-radius:999px;background:rgba(249,115,22,0.2);color:#fdba74;">{playoff_label}</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

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
        st.markdown(
            f"""
<div style="border:1px solid #334155;border-radius:12px;padding:0.7rem;background:rgba(15,23,42,0.65);">
  <div style="font-weight:700;color:#f8fafc;">{home_label}</div>
  <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.4rem;">{len(home_roster)} active players</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.dataframe(home_roster[["PLAYER_NAME"]], use_container_width=True, hide_index=True, height=320)
    with rp2:
        st.markdown(
            f"""
<div style="border:1px solid #334155;border-radius:12px;padding:0.7rem;background:rgba(15,23,42,0.65);">
  <div style="font-weight:700;color:#f8fafc;">{away_label}</div>
  <div style="font-size:0.78rem;color:#94a3b8;margin-bottom:0.4rem;">{len(away_roster)} active players</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.dataframe(away_roster[["PLAYER_NAME"]], use_container_width=True, hide_index=True, height=320)
except Exception as exc:
    st.warning(f"Could not load roster preview: {exc}")

if st.button("Run Matchup Model", type="primary"):
    try:
        with st.spinner("Calculating player predictions..."):
            output = predict_matchup(
                artifacts=artifacts,
                current_players=current_players,
                current_teams=current_teams,
                history_df=train_df,
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
    q50_col = "q50" if "q50" in quantile_cols else (quantile_cols[0] if quantile_cols else None)
    q10_col = "q10" if "q10" in quantile_cols else None
    q90_col = "q90" if "q90" in quantile_cols else None

    if q10_col and q90_col:
        home_preds["interval_width"] = home_preds[q90_col] - home_preds[q10_col]
        away_preds["interval_width"] = away_preds[q90_col] - away_preds[q10_col]
    else:
        home_preds["interval_width"] = pd.NA
        away_preds["interval_width"] = pd.NA

    if q50_col:
        home_preds = home_preds.sort_values(q50_col, ascending=False)
        away_preds = away_preds.sort_values(q50_col, ascending=False)

    show_cols = ["PLAYER_NAME", *quantile_cols, "interval_width"]

    st.markdown("#### Full Team Forecasts")
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(f"##### {home_label}")
        st.dataframe(home_preds[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True, height=520)
    with fc2:
        st.markdown(f"##### {away_label}")
        st.dataframe(away_preds[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True, height=520)
