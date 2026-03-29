from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _placeholder(title: str, message: str):
    fig = go.Figure()
    fig.update_layout(title=title)
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return fig


def _apply_time_axis(df: pd.DataFrame, base_time: float | None) -> pd.DataFrame:
    df = df.copy()
    # Use a synthetic anchor so timeline plots always receive datetime axes.
    anchor = pd.Timestamp("2026-01-01 00:00:00")
    if base_time is None:
        df["start"] = anchor + pd.to_timedelta(df["scheduled_time"], unit="s")
        # Wider bars improve readability for point schedules.
        df["finish"] = df["start"] + pd.to_timedelta(30, unit="s")
        return df
    base = pd.to_datetime(base_time, unit="s")
    df["start"] = base + pd.to_timedelta(df["scheduled_time"], unit="s")
    df["finish"] = base + pd.to_timedelta(df["scheduled_time"] + 30.0, unit="s")
    return df


def _add_flight_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].fillna("").astype(str).str.strip()
    if "aircraft" in df.columns:
        df["aircraft"] = df["aircraft"].fillna("").astype(str).str.strip()
    if "icao24" in df.columns:
        df["icao24"] = df["icao24"].fillna("").astype(str).str.strip()
    callsign = df.get("callsign", pd.Series([""] * len(df)))
    aircraft = df.get("aircraft", pd.Series([""] * len(df)))
    icao24 = df.get("icao24", pd.Series([""] * len(df)))
    df["flight_label"] = callsign.where(callsign != "", aircraft)
    df["flight_label"] = df["flight_label"].where(df["flight_label"] != "", icao24)
    df["flight_label"] = df["flight_label"].where(df["flight_label"] != "", "UNKNOWN")
    return df


def runway_timeline(schedule: pd.DataFrame, base_time: float | None = None):
    if schedule.empty:
        return _placeholder("Runway Utilization", "No schedule data available")
    df = _add_flight_label(_apply_time_axis(schedule, base_time))
    hover_cols = [col for col in ["flight_label", "callsign", "aircraft", "icao24", "delay"] if col in df.columns]
    fig = px.timeline(
        df,
        x_start="start",
        x_end="finish",
        y="assigned_runway",
        color="wake_class",
        hover_data=hover_cols,
        title="Runway Utilization",
    )
    fig.update_traces(text=df["flight_label"], textposition="inside")
    fig.update_yaxes(autorange="reversed")
    return fig


def arrivals_departures_timeline(schedule: pd.DataFrame, base_time: float | None = None):
    if schedule.empty:
        return _placeholder("Arrivals vs Departures by Runway", "No schedule data available")
    df = _add_flight_label(_apply_time_axis(schedule, base_time))
    hover_cols = [col for col in ["flight_label", "callsign", "aircraft", "icao24", "delay", "priority_score"] if col in df.columns]
    fig = px.timeline(
        df,
        x_start="start",
        x_end="finish",
        y="assigned_runway",
        color="event_type",
        hover_data=hover_cols,
        title="Arrivals vs Departures by Runway",
    )
    fig.update_traces(text=df["flight_label"], textposition="inside")
    fig.update_yaxes(autorange="reversed")
    return fig


def runway_free_intervals(schedule: pd.DataFrame, base_time: float | None = None):
    if schedule.empty:
        return _placeholder("Runway Free-Time Intervals", "No schedule data available")
    rows = []
    for runway, group in schedule.groupby("assigned_runway"):
        ordered = group.sort_values("scheduled_time")
        times = ordered["scheduled_time"].values
        for i in range(1, len(times)):
            gap = times[i] - times[i - 1]
            rows.append({"runway": runway, "start": times[i - 1], "finish": times[i], "gap": gap})
    if not rows:
        return _placeholder("Runway Free-Time Intervals", "Not enough points to compute gaps")
    df = pd.DataFrame(rows)
    anchor = pd.Timestamp("2026-01-01 00:00:00")
    if base_time is not None:
        anchor = pd.to_datetime(base_time, unit="s")
    df["start"] = anchor + pd.to_timedelta(df["start"], unit="s")
    df["finish"] = anchor + pd.to_timedelta(df["finish"], unit="s")
    fig = px.timeline(
        df,
        x_start="start",
        x_end="finish",
        y="runway",
        color="gap",
        title="Runway Free-Time Intervals",
    )
    fig.update_yaxes(autorange="reversed")
    return fig
