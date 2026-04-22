"""FanDuel live NBA player points market loader."""

from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ODDS_API_EVENTS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
ODDS_API_EVENT_ODDS_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIVE_CACHE_DIR = PROJECT_ROOT / ".cache" / "live_odds"
LIVE_CACHE_TTL_SECONDS = 120


def _as_decimal(runner: dict) -> float | None:
    for key in ("price", "decimalDisplayOdds", "trueOdds", "oddsDecimal"):
        value = runner.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    odds = runner.get("winRunnerOdds") or runner.get("odds")
    if isinstance(odds, dict):
        for key in ("decimalDisplayOdds", "trueOdds", "oddsDecimal"):
            value = odds.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _coerce_iso(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        return None


def _cache_file_path(api_key: str, max_events: int) -> Path:
    key_hash = hashlib.sha1(api_key.encode("utf-8")).hexdigest()[:12]
    LIVE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return LIVE_CACHE_DIR / f"fanduel_player_points_{key_hash}_{max_events}.pkl"


def _read_cached_df(cache_path: Path, max_age_seconds: int | None) -> pd.DataFrame | None:
    if not cache_path.exists():
        return None
    try:
        payload = pd.read_pickle(cache_path)
    except Exception:
        return None
    if not isinstance(payload, dict) or "df" not in payload:
        return None

    fetched_at = payload.get("fetched_at")
    if max_age_seconds is not None:
        try:
            age = time.time() - float(fetched_at)
            if age > max_age_seconds:
                return None
        except (TypeError, ValueError):
            return None

    df = payload.get("df")
    if not isinstance(df, pd.DataFrame):
        return None
    return df.copy()


def _write_cached_df(cache_path: Path, df: pd.DataFrame) -> None:
    try:
        pd.to_pickle({"fetched_at": time.time(), "df": df.copy()}, cache_path)
    except Exception:
        return


def fetch_fanduel_live_player_points(region: str = "NY", max_events: int = 8) -> pd.DataFrame:
    """Fetch live/upcoming FanDuel NBA player points markets.

    Returns one row per player line with decimal over/under odds.
    """
    del region  # Region is unused with this provider; kept for app compatibility.

    api_key = os.getenv("ODDS_API_KEY") or os.getenv("THE_ODDS_API_KEY")
    if not api_key:
        return pd.DataFrame()

    event_limit = max(0, int(max_events))
    cache_path = _cache_file_path(api_key=api_key, max_events=event_limit)
    cached_df = _read_cached_df(cache_path, max_age_seconds=LIVE_CACHE_TTL_SECONDS)
    if cached_df is not None:
        return cached_df

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/json,text/plain,*/*",
        }
    )

    params = {
        "apiKey": api_key,
        "dateFormat": "iso",
    }
    resp = session.get(ODDS_API_EVENTS_URL, params=params, timeout=20)
    if resp.status_code != 200:
        stale_df = _read_cached_df(cache_path, max_age_seconds=None)
        return stale_df if stale_df is not None else pd.DataFrame()

    payload = resp.json()
    if not isinstance(payload, list):
        stale_df = _read_cached_df(cache_path, max_age_seconds=None)
        return stale_df if stale_df is not None else pd.DataFrame()

    rows: list[dict] = []

    event_candidates: list[dict] = []
    for event in payload:
        if not isinstance(event, dict):
            continue
        if not str(event.get("id") or ""):
            continue
        event_candidates.append(event)
        if event_limit > 0 and len(event_candidates) >= event_limit:
            break

    for event in event_candidates:
        if not isinstance(event, dict):
            continue

        event_id = str(event.get("id") or "")
        if not event_id:
            continue

        event_odds_params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "player_points",
            "bookmakers": "fanduel",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        event_resp = session.get(
            ODDS_API_EVENT_ODDS_URL.format(event_id=event_id),
            params=event_odds_params,
            timeout=20,
        )
        if event_resp.status_code != 200:
            continue

        event_payload = event_resp.json()
        if not isinstance(event_payload, dict):
            continue

        home_team = str(event_payload.get("home_team") or event.get("home_team") or "")
        away_team = str(event_payload.get("away_team") or event.get("away_team") or "")
        event_name = f"{away_team} @ {home_team}".strip(" @")
        open_date = _coerce_iso(event_payload.get("commence_time") or event.get("commence_time"))

        bookmakers = event_payload.get("bookmakers") or []
        if not isinstance(bookmakers, list):
            continue

        for bookmaker in bookmakers:
            if not isinstance(bookmaker, dict):
                continue

            if str(bookmaker.get("key") or "").lower() != "fanduel":
                continue

            markets = bookmaker.get("markets") or []
            if not isinstance(markets, list):
                continue

            for market in markets:
                if not isinstance(market, dict):
                    continue
                if str(market.get("key") or "").lower() != "player_points":
                    continue

                outcomes = market.get("outcomes") or []
                if not isinstance(outcomes, list) or not outcomes:
                    continue

                grouped: dict[tuple[str, float], dict[str, float | str]] = {}
                for outcome in outcomes:
                    if not isinstance(outcome, dict):
                        continue
                    player_name = str(outcome.get("description") or "").strip()
                    side = str(outcome.get("name") or "").strip().lower()
                    if side not in ("over", "under"):
                        continue
                    if not player_name:
                        continue

                    point = outcome.get("point")
                    try:
                        line_value = float(point)
                    except (TypeError, ValueError):
                        continue

                    odds = _as_decimal(outcome)
                    if odds is None:
                        continue

                    key = (player_name, line_value)
                    if key not in grouped:
                        grouped[key] = {
                            "player_name": player_name,
                            "line": line_value,
                        }
                    grouped[key][side] = float(odds)

                for value in grouped.values():
                    over_odds = value.get("over")
                    under_odds = value.get("under")
                    if over_odds is None or under_odds is None:
                        continue

                    rows.append(
                        {
                            "event_id": event_id,
                            "event_name": event_name,
                            "game_date": open_date.tz_convert(None) if open_date is not None else pd.NaT,
                            "home_team": home_team,
                            "away_team": away_team,
                            "team": "",
                            "opponent": "",
                            "player_name": str(value["player_name"]),
                            "line": float(value["line"]),
                            "over_odds": float(over_odds),
                            "under_odds": float(under_odds),
                            "bookmaker": "fanduel",
                            "market_name": "player_points",
                        }
                    )

    df = pd.DataFrame(rows)
    if df.empty:
        _write_cached_df(cache_path, df)
        return df

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df = df[df["player_name"].ne("")].copy()
    df = df.drop_duplicates(subset=["event_id", "player_name", "line"], keep="first")

    # Infer team labels from event title where available.
    at_pattern = re.compile(r"^\s*(.*?)\s*@\s*(.*?)\s*$")
    for idx, name in df["event_name"].items():
        match = at_pattern.match(str(name))
        if not match:
            continue
        away_name = match.group(1).strip()
        home_name = match.group(2).strip()
        if not df.at[idx, "home_team"]:
            df.at[idx, "home_team"] = home_name
        if not df.at[idx, "away_team"]:
            df.at[idx, "away_team"] = away_name

    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    df = df.sort_values(["game_date", "event_id", "player_name"]).reset_index(drop=True)
    _write_cached_df(cache_path, df)
    return df
