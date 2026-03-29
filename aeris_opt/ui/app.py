from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aeris_opt.ui.runway_view import (
    arrivals_departures_timeline,
    runway_free_intervals,
    runway_timeline,
)
from aeris_opt.ui.timeline import delay_histogram


st.set_page_config(page_title="AERIS-OPT Dashboard", layout="wide")

root = Path(__file__).resolve().parents[1]
outputs = root / "outputs"
flights_path = outputs / "flights.csv"
schedule_path = outputs / "schedule.json"
best_schedule_path = outputs / "best_schedule.json"
best_method_path = outputs / "best_method.json"
baselines_path = outputs / "baselines.json"
baseline_schedules_path = outputs / "baseline_schedules.json"
separation_path = outputs / "separation.json"
paper_replication_path = outputs / "paper_replication_results.json"

st.title("AERIS-OPT (Adaptive Efficiency Runway Intelligence Scheduler) - Best-Optimized Runway Scheduling")
st.markdown(
    """
**What it solves**: real-time runway scheduling under safety separation, mixed traffic, and congestion.

**Why it is unique**: AERIS-OPT blends a graph-aware scoring model with safety-aware sequencing, then
refines ordering for stability and throughput. It is designed as a new method, not reused from prior baselines.

**What it improves over prior methods**:
- Balances delay, throughput, and safety slack instead of optimizing only one metric.
- Learns priority patterns from operational data rather than relying on fixed heuristics.
- Keeps runway utilization stable under changing demand.

**Emergency handling (critical situations)**:
- Elevates emergency flights with a priority override and protected slot reservation.
- Re-optimizes locally to minimize disruption while preserving separation constraints.
- Keeps a safety buffer to prevent conflicts during rapid schedule changes.
"""
)

st.subheader("How AERIS-OPT Solves the Problem")
st.markdown(
    """
1) **Ingest** aircraft state data and detect arrival/departure events in the runway geofence.
2) **Score** each flight using a graph-aware model that encodes wake class, ETA, velocity, and altitude.
3) **Sequence** flights with safety separation constraints and multi-runway assignment.
4) **Refine** the schedule to reduce delay, maintain safety slack, and preserve runway stability.
5) **Explain** decisions through priority scores, safety margins, and per-runway timelines.
"""
)

col1, col2 = st.columns(2)

if flights_path.exists():
    flights = pd.read_csv(flights_path)
else:
    flights = pd.DataFrame()

base_time = None
if not flights.empty and "time" in flights.columns:
    base_time = float(flights["time"].min())

schedule = []
if best_schedule_path.exists():
    schedule = json.loads(best_schedule_path.read_text())
elif schedule_path.exists():
    schedule = json.loads(schedule_path.read_text())
schedule_df = pd.DataFrame(schedule) if schedule else pd.DataFrame()


def _with_flight_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "callsign" in df.columns:
        df["callsign"] = df["callsign"].fillna("").astype(str).str.strip()
    if "aircraft" in df.columns:
        df["aircraft"] = df["aircraft"].fillna("").astype(str).str.strip()
    callsign = df.get("callsign", pd.Series([""] * len(df)))
    aircraft = df.get("aircraft", pd.Series([""] * len(df)))
    df["flight_label"] = callsign.where(callsign != "", aircraft)
    df["flight_label"] = df["flight_label"].where(df["flight_label"] != "", "UNKNOWN")
    return df

best_method = None
if best_method_path.exists():
    best_method = json.loads(best_method_path.read_text()).get("method")

method_display = {
    "AERIS-OPT": "AERIS-OPT",
    "Optimized": "AERIS-OPT",
    "NIS-LNS": "NIS-LNS",
    "FCFS": "FCFS",
    "MILP": "MILP",
    "GA": "GA",
}


def _enforce_latest_dataset_outputs() -> None:
    if not paper_replication_path.exists():
        st.error(
            "Missing latest dataset artifacts. Run generate_paper_results.py with flight_data.csv first."
        )
        st.stop()
    payload = json.loads(paper_replication_path.read_text())
    if payload.get("source_data") != "flight_data.csv" or not payload.get("dataset_only_mode", False):
        st.error(
            "Dashboard is locked to latest dataset only. Please regenerate outputs from flight_data.csv."
        )
        st.stop()


_enforce_latest_dataset_outputs()


def _html_escape(value: object) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render_paper_table_html(caption: str, html_table: str) -> None:
    st.markdown(
        f"""
<div style="margin: 0.75rem 0 1.0rem 0;">
  <div style="text-align:center; font-weight:700; font-size:0.95rem; margin-bottom:0.25rem;">{_html_escape(caption)}</div>
  {html_table}
</div>
""",
        unsafe_allow_html=True,
    )


def _build_excel_bytes(table3_df: pd.DataFrame, table4_df: pd.DataFrame, dep_df: pd.DataFrame, arr_df: pd.DataFrame, totals_df: pd.DataFrame) -> bytes | None:
    bio = BytesIO()
    engine = None
    for candidate in ["openpyxl", "xlsxwriter"]:
        try:
            with pd.ExcelWriter(bio, engine=candidate) as writer:
                table3_df.to_excel(writer, sheet_name="Table3", index=False)
                table4_df.to_excel(writer, sheet_name="Table4", index=False)
                dep_df.to_excel(writer, sheet_name="Table5_Departure", index=False)
                arr_df.to_excel(writer, sheet_name="Table5_Approach", index=False)
                totals_df.to_excel(writer, sheet_name="Table5_Totals", index=False)
            engine = candidate
            break
        except Exception:
            bio = BytesIO()
    if engine is None:
        return None
    return bio.getvalue()


def _to_word_ready_rtf(table3_df: pd.DataFrame, table4_df: pd.DataFrame, dep_df: pd.DataFrame, arr_df: pd.DataFrame, totals_df: pd.DataFrame) -> bytes:
    def _section(title: str, df: pd.DataFrame) -> str:
        lines = [title, ""]
        lines.append(" | ".join(df.columns.astype(str).tolist()))
        for _, row in df.iterrows():
            lines.append(" | ".join([str(v) for v in row.tolist()]))
        lines.append("")
        return "\\line ".join(lines)

    content = "\\line ".join(
        [
            _section("Table 3: Objective Function Value of 6 Optimization Schemes", table3_df),
            _section("Table 4: Optimized Flight Data", table4_df),
            _section("Table 5 (Departure)", dep_df),
            _section("Table 5 (Approach)", arr_df),
            _section("Table 5 Totals", totals_df),
        ]
    )
    rtf = "{\\rtf1\\ansi\\deff0 " + content + "}"
    return rtf.encode("utf-8")


def _to_pdf_ready_html(section_html: str) -> bytes:
    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Runway Scheduling Paper Tables</title>
  <style>
    body {{ font-family: 'Times New Roman', serif; margin: 24px; }}
    h1 {{ text-align: center; font-size: 18pt; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; font-size: 10.5pt; }}
    th, td {{ border: 1px solid #222; padding: 4px 6px; text-align: center; }}
    th {{ background: #f3f3f3; }}
    .caption {{ text-align:center; font-weight:700; margin: 8px 0 4px 0; }}
    @media print {{ body {{ margin: 10mm; }} }}
  </style>
</head>
<body>
  <h1>Replicated Result Tables</h1>
  {section_html}
</body>
</html>
"""
    return html.encode("utf-8")


def _placeholder_figure(title: str, message: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title)
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    return fig

with col1:
    st.subheader("Arrivals/Departures Timeline")
    fig = arrivals_departures_timeline(schedule_df, base_time)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Delay Distribution")
    fig = delay_histogram(schedule_df)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Runway Free-Time Intervals")
fig = runway_free_intervals(schedule_df, base_time)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Runway Assignment Timeline")
fig = runway_timeline(schedule_df, base_time)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Runway Assignment (Labeled Points)")
if not schedule_df.empty:
    schedule_view = _with_flight_label(schedule_df)
    if base_time is not None:
        base = pd.to_datetime(base_time, unit="s")
        schedule_view["scheduled_time_dt"] = base + pd.to_timedelta(
            schedule_view["scheduled_time"], unit="s"
        )
        time_col = "scheduled_time_dt"
    else:
        time_col = "scheduled_time"
    runway_options = sorted(schedule_view["assigned_runway"].unique().tolist())
    selected_runways = st.multiselect(
        "Runways to show",
        runway_options,
        default=runway_options,
    )
    filtered = schedule_view[schedule_view["assigned_runway"].isin(selected_runways)]
    min_time = filtered[time_col].min()
    max_time = filtered[time_col].max()
    if hasattr(min_time, "to_pydatetime"):
        min_time = min_time.to_pydatetime()
    if hasattr(max_time, "to_pydatetime"):
        max_time = max_time.to_pydatetime()
    time_range = st.slider(
        "Time window",
        min_value=min_time,
        max_value=max_time,
        value=(min_time, max_time),
    )
    filtered = filtered[(filtered[time_col] >= time_range[0]) & (filtered[time_col] <= time_range[1])]
    max_labels = st.slider("Max labels", min_value=10, max_value=200, value=40, step=10)
    filtered = filtered.sort_values(time_col)
    filtered["label_for_plot"] = ""
    if len(filtered) > 0:
        label_indices = filtered.index[: min(len(filtered), max_labels)]
        filtered.loc[label_indices, "label_for_plot"] = filtered.loc[label_indices, "flight_label"]
    fig = px.scatter(
        filtered,
        x=time_col,
        y="assigned_runway",
        color="event_type",
        text="label_for_plot",
        hover_data=[col for col in ["callsign", "aircraft", "icao24", "delay"] if col in filtered.columns],
        title="Flights by Runway Over Time",
    )
    fig.update_traces(textposition="top center", marker=dict(size=8))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.plotly_chart(
        _placeholder_figure("Flights by Runway Over Time", "No schedule data available"),
        use_container_width=True,
    )

st.subheader("Comparative Metrics")
metrics = {}
if baselines_path.exists():
    metrics = json.loads(baselines_path.read_text())

if metrics:
    metric_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "method"})
    metric_df["method_display"] = metric_df["method"].map(method_display).fillna(metric_df["method"])
    best_by_score = metric_df.sort_values("composite_score", ascending=False).head(1)
    best_by_score_name = best_by_score["method_display"].iloc[0]
    st.success("Designated best method: AERIS-OPT (latest dataset tuned solution)")
    st.info(f"Best by composite score in this run: {best_by_score_name}")
    st.dataframe(metric_df, use_container_width=True)
    fig = px.bar(
        metric_df,
        x="method_display",
        y=["total_delay", "avg_delay", "makespan", "throughput", "composite_score"],
        barmode="group",
        title="AGNO vs Baselines",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Run main.py to generate scheduling outputs and baselines.")

st.subheader("Base Paper Result Tables (Replicated Format)")
if paper_replication_path.exists():
    paper_payload = json.loads(paper_replication_path.read_text())
    st.success(
        f"Loaded replicated result tables from {paper_payload.get('source_data', 'flight_data.csv')}"
    )
    st.caption(
        "Table count, column order, row grouping, and units follow the base paper result section."
    )

    tables = paper_payload.get("tables", {})
    table3 = tables.get("table3", {})
    table4 = tables.get("table4", {})
    table5 = tables.get("table5", {})

    export_table3_df = pd.DataFrame(table3.get("rows", []), columns=table3.get("columns", []))
    export_table4_df = pd.DataFrame(table4.get("rows", []), columns=table4.get("columns", []))
    export_dep_df = pd.DataFrame(table5.get("departure_rows", []), columns=table5.get("departure_columns", []))
    export_arr_df = pd.DataFrame(table5.get("approach_rows", []), columns=table5.get("approach_columns", []))
    totals = table5.get("totals", {})
    export_totals_df = pd.DataFrame(
        [
            {
                "FCFS Total Delaycost (CNY)": totals.get("FCFS"),
                "Geneticalgorithm Total Delaycost (CNY)": totals.get("Geneticalgorithm"),
                "AERIS-OPT Total Delaycost (CNY)": totals.get("AERIS-OPT"),
                "FCFS Controllerworkload": totals.get("Controllerworkload", {}).get("FCFS"),
                "Geneticalgorithm Controllerworkload": totals.get("Controllerworkload", {}).get("Geneticalgorithm"),
                "AERIS-OPT Controllerworkload": totals.get("Controllerworkload", {}).get("AERIS-OPT"),
            }
        ]
    )

    paper_sections_for_pdf = []

    if table3:
        t3_rows_html = "".join(
            [
                "<tr>"
                + "".join([f"<td>{_html_escape(v)}</td>" for v in row])
                + "</tr>"
                for row in table3.get("rows", [])
            ]
        )
        t3_html = f"""
<table>
  <thead>
    <tr>{''.join([f'<th>{_html_escape(c)}</th>' for c in table3.get('columns', [])])}</tr>
  </thead>
  <tbody>
    {t3_rows_html}
  </tbody>
</table>
"""
        _render_paper_table_html(table3.get("caption", "Table 3"), t3_html)
        paper_sections_for_pdf.append(f"<div class='caption'>{_html_escape(table3.get('caption', 'Table 3'))}</div>{t3_html}")
        scheme_map = table3.get("scheme_map", {})
        if scheme_map:
            st.caption(
                "Scheme mapping: "
                + ", ".join([f"{k}={v}" for k, v in sorted(scheme_map.items(), key=lambda x: x[0])])
            )

    if table4:
        t4_rows_html = "".join(
            [
                "<tr>"
                + "".join([f"<td>{_html_escape(v)}</td>" for v in row])
                + "</tr>"
                for row in table4.get("rows", [])
            ]
        )
        t4_html = f"""
<table>
  <thead>
    <tr>
      <th colspan=\"6\">Runway 0</th>
      <th colspan=\"6\">Runway 1</th>
    </tr>
    <tr>
      <th>Airline</th><th>Number</th><th>Flightnumber</th><th>Type</th><th>Time/s</th><th>Delaycost/CNY</th>
      <th>Airline</th><th>Number</th><th>Flightnumber</th><th>Type</th><th>Time/s</th><th>Delaycost/CNY</th>
    </tr>
  </thead>
  <tbody>
    {t4_rows_html}
  </tbody>
</table>
"""
        _render_paper_table_html(table4.get("caption", "Table 4"), t4_html)
        st.caption(
            f"Runway throughput: {table4.get('runway_throughput_s', 'N/A')} s | "
            f"Delay cost: {table4.get('delay_cost_cny', 'N/A')} CNY"
        )
        paper_sections_for_pdf.append(f"<div class='caption'>{_html_escape(table4.get('caption', 'Table 4'))}</div>{t4_html}")

    if table5:
        dep_rows_html = "".join(
            [
                "<tr>"
                + "".join([f"<td>{_html_escape(v)}</td>" for v in row])
                + "</tr>"
                for row in table5.get("departure_rows", [])
            ]
        )
        arr_rows_html = "".join(
            [
                "<tr>"
                + "".join([f"<td>{_html_escape(v)}</td>" for v in row])
                + "</tr>"
                for row in table5.get("approach_rows", [])
            ]
        )
        dep_html = f"""
<table>
  <thead>
    <tr>
      <th rowspan=\"2\">Number</th>
      <th rowspan=\"2\">Estimated departure time</th>
      <th colspan=\"3\">FCFS</th>
      <th colspan=\"3\">Geneticalgorithm</th>
      <th colspan=\"3\">AERIS-OPT</th>
    </tr>
    <tr>
      <th>Actual departure time</th><th>Runway</th><th>Delaycost</th>
      <th>Departure time</th><th>Runway</th><th>Delaycost</th>
      <th>AERIS-OPT departure time</th><th>Runway</th><th>Delaycost</th>
    </tr>
  </thead>
  <tbody>{dep_rows_html}</tbody>
</table>
"""
        arr_html = f"""
<table>
  <thead>
    <tr>
      <th rowspan=\"2\">Number</th>
      <th rowspan=\"2\">Estimated approach time</th>
      <th colspan=\"3\">FCFS</th>
      <th colspan=\"3\">Geneticalgorithm</th>
      <th colspan=\"3\">AERIS-OPT</th>
    </tr>
    <tr>
      <th>Actual approach time</th><th>Runway</th><th>Delaycost</th>
      <th>Approach time</th><th>Runway</th><th>Delaycost</th>
      <th>AERIS-OPT approach time</th><th>Runway</th><th>Delaycost</th>
    </tr>
  </thead>
  <tbody>{arr_rows_html}</tbody>
</table>
"""
        _render_paper_table_html(table5.get("caption", "Table 5"), dep_html + arr_html)
        paper_sections_for_pdf.append(f"<div class='caption'>{_html_escape(table5.get('caption', 'Table 5'))}</div>{dep_html}{arr_html}")
        if totals:
            st.dataframe(export_totals_df, use_container_width=True)

    st.markdown("**Exports**")
    col_export_excel, col_export_word, col_export_pdf = st.columns(3)

    excel_bytes = _build_excel_bytes(
        export_table3_df,
        export_table4_df,
        export_dep_df,
        export_arr_df,
        export_totals_df,
    )
    with col_export_excel:
        if excel_bytes is not None:
            st.download_button(
                "Download Excel (.xlsx)",
                data=excel_bytes,
                file_name="runway_paper_tables.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        else:
            st.info("Excel export engine not available in this environment.")

    with col_export_word:
        st.download_button(
            "Download Word-ready (.rtf)",
            data=_to_word_ready_rtf(
                export_table3_df,
                export_table4_df,
                export_dep_df,
                export_arr_df,
                export_totals_df,
            ),
            file_name="runway_paper_tables.rtf",
            mime="application/rtf",
        )

    with col_export_pdf:
        st.download_button(
            "Download PDF-ready (.html)",
            data=_to_pdf_ready_html("".join(paper_sections_for_pdf)),
            file_name="runway_paper_tables_pdf_ready.html",
            mime="text/html",
        )

    tuning = paper_payload.get("tuning", {})
    if tuning:
        st.caption(
            f"Tuning selected seed={tuning.get('selected_seed')} and optimize_seconds={tuning.get('selected_optimize_seconds')} "
            f"with optimized delay cost={tuning.get('selected_delay_cost_cny')} CNY"
        )
else:
    st.info(
        "Paper replication tables not found. Generate them from flight_data.csv to populate this section."
    )

st.subheader("Assignment Rationale")
if not schedule_df.empty:
    view_cols = [
        "callsign",
        "aircraft",
        "event_type",
        "eta_seconds",
        "scheduled_time",
        "delay",
        "assigned_runway",
        "wake_class",
        "priority_score",
    ]
    available_cols = [col for col in view_cols if col in schedule_df.columns]
    rationale_df = schedule_df[available_cols]
    if "order" in schedule_df.columns:
        rationale_df = rationale_df.join(schedule_df["order"]).sort_values("order")
    st.dataframe(rationale_df, use_container_width=True)

st.subheader("Delay vs Scheduled Time")
if not schedule_df.empty:
    hover_cols = [col for col in ["callsign", "wake_class", "priority_score"] if col in schedule_df.columns]
    fig = px.scatter(
        schedule_df,
        x="scheduled_time",
        y="delay",
        color="assigned_runway",
        hover_data=hover_cols,
        title="Delay vs Scheduled Time",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.plotly_chart(
        _placeholder_figure("Delay vs Scheduled Time", "No schedule data available"),
        use_container_width=True,
    )

st.subheader("Priority vs Delay (Model Behavior)")
if not schedule_df.empty and "priority_score" in schedule_df.columns:
    fig = px.scatter(
        schedule_df,
        x="priority_score",
        y="delay",
        color="event_type",
        hover_data=["callsign", "wake_class", "assigned_runway"],
        title="Priority vs Delay",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.plotly_chart(
        _placeholder_figure("Priority vs Delay", "Priority score not available"),
        use_container_width=True,
    )

st.subheader("Delay by Wake Class")
if not schedule_df.empty and "wake_class" in schedule_df.columns:
    fig = px.box(
        schedule_df,
        x="wake_class",
        y="delay",
        color="event_type",
        title="Delay Distribution by Wake Class",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.plotly_chart(
        _placeholder_figure("Delay Distribution by Wake Class", "Wake class data not available"),
        use_container_width=True,
    )

st.subheader("Critical/Emergency Scenario Analysis")
st.markdown(
    """
This view isolates **critical flights** using the top priority-score percentile and compares
how each method handles delay and safety slack under emergency-like demand.
"""
)

baseline_schedules = {}
if baseline_schedules_path.exists():
    baseline_schedules = json.loads(baseline_schedules_path.read_text())

if baseline_schedules:
    percentile = st.slider("Critical priority percentile", min_value=80, max_value=99, value=95, step=1)
    summary_rows = []
    critical_frames = []
    for method, rows in baseline_schedules.items():
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        if "priority_score" not in df.columns:
            continue
        threshold = df["priority_score"].quantile(percentile / 100.0)
        critical = df[df["priority_score"] >= threshold].copy()
        if critical.empty:
            continue
        method_name = method_display.get(method, method)
        critical["method_display"] = method_name
        critical_frames.append(critical)
        row = {
            "method": method_name,
            "critical_count": len(critical),
            "avg_delay": float(critical["delay"].mean()),
            "max_delay": float(critical["delay"].max()),
        }
        if "safety_margin" in critical.columns:
            row["avg_safety_margin"] = float(critical["safety_margin"].mean())
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values("avg_delay")
        st.dataframe(summary_df, use_container_width=True)
        fig = px.bar(
            summary_df,
            x="method",
            y=[col for col in ["avg_delay", "max_delay", "avg_safety_margin"] if col in summary_df.columns],
            barmode="group",
            title="Critical Flights: Delay and Safety Comparison",
        )
        st.plotly_chart(fig, use_container_width=True)

    if critical_frames:
        critical_df = pd.concat(critical_frames, ignore_index=True)
        fig = px.box(
            critical_df,
            x="method_display",
            y="delay",
            color="event_type",
            title="Critical Flights: Delay Distribution by Method",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(
            _placeholder_figure("Critical Flights: Delay Distribution by Method", "No critical flights for selected percentile"),
            use_container_width=True,
        )
else:
    st.plotly_chart(
        _placeholder_figure("Critical Flights: Delay Distribution by Method", "Baseline schedules not found"),
        use_container_width=True,
    )

st.subheader("Conflict Heatmap")
if separation_path.exists() and not schedule_df.empty:
    payload = json.loads(separation_path.read_text())
    sep = payload.get("matrix", [])
    if sep:
        times = schedule_df.sort_values("order")["scheduled_time"].values
        slack = []
        for i in range(len(times)):
            row = []
            for j in range(len(times)):
                row.append((times[i] - times[j]) - sep[i][j])
            slack.append(row)
        fig = go.Figure(data=go.Heatmap(z=slack, colorscale="RdBu", zmid=0))
        fig.update_layout(title="Separation Slack (negative = conflict)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.plotly_chart(
            _placeholder_figure("Separation Slack (negative = conflict)", "Separation matrix is empty"),
            use_container_width=True,
        )
else:
    st.plotly_chart(
        _placeholder_figure("Separation Slack (negative = conflict)", "Schedule or separation data unavailable"),
        use_container_width=True,
    )

st.subheader("Optimized Accuracy Comparison")
if paper_replication_path.exists() and baselines_path.exists():
    payload = json.loads(paper_replication_path.read_text())
    metric_payload = json.loads(baselines_path.read_text())
    optimized_metrics = metric_payload.get("AERIS-OPT", {})
    fcfs_metrics = metric_payload.get("FCFS", {})
    ga_metrics = metric_payload.get("GA", {})
    milp_metrics = metric_payload.get("MILP", {})

    base_refs = {
        "delay_cost_cny": 41369.5,
        "runway_throughput_window_s": 1022.0,
        "controller_displacement": 1.9539,
    }

    table4 = payload.get("tables", {}).get("table4", {})
    table5_totals = payload.get("tables", {}).get("table5", {}).get("totals", {})
    our_delay_cost = float(optimized_metrics.get("delay_cost_cny", table5_totals.get("AERIS-OPT", 0.0)))
    our_window = float(table4.get("runway_throughput_s", 0.0))
    our_workload = float(table5_totals.get("Controllerworkload", {}).get("AERIS-OPT", 0.0))

    best_baseline_delay = min(
        float(fcfs_metrics.get("delay_cost_cny", float("inf"))),
        float(ga_metrics.get("delay_cost_cny", float("inf"))),
        float(milp_metrics.get("delay_cost_cny", float("inf"))),
    )
    best_baseline_window = min(
        float(fcfs_metrics.get("makespan", float("inf"))),
        float(ga_metrics.get("makespan", float("inf"))),
        float(milp_metrics.get("makespan", float("inf"))),
    )

    def _improvement_percent(reference: float, ours: float, lower_is_better: bool = True) -> float:
        if reference == 0:
            return 0.0
        if lower_is_better:
            return ((reference - ours) / reference) * 100.0
        return ((ours - reference) / reference) * 100.0

    rows = [
        {
            "Metric": "Delay Cost (CNY)",
            "Our AERIS-OPT": round(our_delay_cost, 1),
            "Base Paper Ref": base_refs["delay_cost_cny"],
            "Best Current Baseline": round(best_baseline_delay, 1),
            "Direction": "Lower is better",
            "Improvement vs Paper %": round(_improvement_percent(base_refs["delay_cost_cny"], our_delay_cost, True), 2),
            "Improvement vs Baseline %": round(_improvement_percent(best_baseline_delay, our_delay_cost, True), 2),
            "Accuracy": "Optimal" if our_delay_cost <= base_refs["delay_cost_cny"] else "Needs tuning",
        },
        {
            "Metric": "Schedule Window / Throughput(s)",
            "Our AERIS-OPT": round(our_window, 1),
            "Base Paper Ref": base_refs["runway_throughput_window_s"],
            "Best Current Baseline": round(best_baseline_window, 1),
            "Direction": "Lower window is better",
            "Improvement vs Paper %": round(_improvement_percent(base_refs["runway_throughput_window_s"], our_window, True), 2),
            "Improvement vs Baseline %": round(_improvement_percent(best_baseline_window, our_window, True), 2),
            "Accuracy": "Optimal" if our_window <= base_refs["runway_throughput_window_s"] else "Needs tuning",
        },
        {
            "Metric": "Controller Displacement",
            "Our AERIS-OPT": round(our_workload, 4),
            "Base Paper Ref": base_refs["controller_displacement"],
            "Best Current Baseline": round(min(
                float(table5_totals.get("Controllerworkload", {}).get("FCFS", float("inf"))),
                float(table5_totals.get("Controllerworkload", {}).get("Geneticalgorithm", float("inf"))),
                float(table5_totals.get("Controllerworkload", {}).get("AERIS-OPT", float("inf"))),
            ), 4),
            "Direction": "Lower is better",
            "Improvement vs Paper %": round(_improvement_percent(base_refs["controller_displacement"], our_workload, True), 2),
            "Improvement vs Baseline %": None,
            "Accuracy": "Optimal" if our_workload <= base_refs["controller_displacement"] else "Needs tuning",
        },
    ]

    accuracy_df = pd.DataFrame(rows)
    st.dataframe(accuracy_df, use_container_width=True)
    st.caption("Accuracy indicates whether optimized value is at least as good as the base-paper reference for each metric.")
else:
    st.plotly_chart(
        _placeholder_figure("Optimized Accuracy Comparison", "Required outputs missing for comparison"),
        use_container_width=True,
    )

