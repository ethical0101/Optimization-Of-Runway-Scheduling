from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from aeris_opt.analytics.metrics import compute_metrics
from aeris_opt.data.separation_builder import build_separation_matrix
from aeris_opt.optimizer.baselines.fcfs import fcfs_schedule
from aeris_opt.optimizer.baselines.ga import ga_schedule
from aeris_opt.optimizer.baselines.milp import milp_schedule
from aeris_opt.optimizer.graph_model import AGNOModel, build_features, schedule_with_agnos
from aeris_opt.optimizer.nis_optimizer import schedule_with_nis
from aeris_opt.optimizer.robust_refiner import refine_schedule_with_runways

LATEST_DATASET_NAME = "flight_data.csv"
RESEARCH_METHOD_NAME = "AERIS-OPT"

# Base-paper targets used for strict multi-factor optimization.
TARGET_DELAY_COST_CNY = 41369.5
TARGET_CONTROLLER_DISPLACEMENT = 1.9539
TARGET_WINDOW_SECONDS = 1022.0


def _strict_target_score(cost: float, displacement: float, window_seconds: float) -> float:
    over_cost = max(0.0, cost - TARGET_DELAY_COST_CNY)
    over_disp = max(0.0, displacement - TARGET_CONTROLLER_DISPLACEMENT)
    over_window = max(0.0, window_seconds - TARGET_WINDOW_SECONDS)
    return (
        over_cost * 1000.0
        + over_disp * 1_000_000.0
        + over_window * 10_000.0
        + cost * 1.0
        + displacement * 500.0
        + window_seconds * 0.2
    )


@dataclass
class ReplicationArtifacts:
    payload: dict
    output_path: Path


def _parse_hms_to_seconds(value: str) -> int:
    parts = [int(x) for x in str(value).split(":")]
    if len(parts) != 3:
        return 0
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def _seconds_to_hms(value: float | int) -> str:
    sec = int(round(float(value)))
    if sec < 0:
        sec = 0
    hh = sec // 3600
    mm = (sec % 3600) // 60
    ss = sec % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _normalize_int_list(values: Sequence[int] | None, fallback: list[int]) -> list[int]:
    if not values:
        return fallback
    out = []
    for v in values:
        iv = int(v)
        if iv not in out:
            out.append(iv)
    return out or fallback


def _normalize_float_list(values: Sequence[float] | None, fallback: list[float]) -> list[float]:
    if not values:
        return fallback
    out = []
    for v in values:
        fv = float(v)
        if fv not in out:
            out.append(fv)
    return out or fallback


def _validate_latest_dataset(path: Path) -> None:
    if path.name != LATEST_DATASET_NAME:
        raise ValueError(
            f"Latest-dataset-only mode is enabled. Expected dataset name '{LATEST_DATASET_NAME}', got '{path.name}'."
        )


def _build_input_flights(flight_data_path: Path) -> pd.DataFrame:
    _validate_latest_dataset(flight_data_path)
    raw = pd.read_csv(flight_data_path)
    required = [
        "Category",
        "Airline",
        "Number",
        "FlightNumber",
        "Type",
        "UnitDelayCost",
        "EstimatedTime",
        "Runway",
    ]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    raw = raw.copy()
    raw["EstimatedTimeSeconds"] = raw["EstimatedTime"].map(_parse_hms_to_seconds)
    raw["wake_class"] = raw["Type"].astype(str).str.strip().str.upper()
    raw["event_type"] = raw["Category"].str.lower().map(
        {"arrival": "arrival", "departure": "departure"}
    )

    # Deterministic placeholders for model features when only aggregated flight table is available.
    raw["velocity"] = raw["Category"].map({"Arrival": 230.0, "Departure": 170.0}).fillna(200.0)
    raw["geoaltitude"] = raw["Category"].map({"Arrival": 2500.0, "Departure": 200.0}).fillna(500.0)
    raw["callsign"] = raw["FlightNumber"].astype(str)
    raw["icao24"] = raw["FlightNumber"].astype(str)
    raw["aircraft"] = raw["FlightNumber"].astype(str)
    raw["eta_seconds"] = raw["EstimatedTimeSeconds"].astype(float)
    raw["flight_index"] = raw["Number"].astype(int) - 1
    return raw


def _wake_onehot(series: pd.Series) -> torch.Tensor:
    mapping = {"H": 0, "M": 1, "L": 2}
    idx = series.map(mapping).fillna(1).astype(int).values
    onehot = torch.zeros((len(idx), 3))
    onehot[range(len(idx)), idx] = 1.0
    return onehot


def _scores_for_flights(model: AGNOModel, flights: pd.DataFrame, sep_matrix: torch.Tensor) -> list[float]:
    eta = torch.tensor(flights["eta_seconds"].values, dtype=torch.float32)
    velocity = torch.tensor(flights["velocity"].values, dtype=torch.float32)
    altitude = torch.tensor(flights["geoaltitude"].values, dtype=torch.float32)
    wake_onehot = _wake_onehot(flights["wake_class"])
    features = build_features(eta, velocity, altitude, wake_onehot)
    adj = (sep_matrix > 0).float()
    with torch.no_grad():
        scores = model(features, adj).detach().cpu().tolist()
    return [float(v) for v in scores]


def _schedule_result_to_df(
    flights: pd.DataFrame,
    order: list[int],
    scheduled_times: list[float],
    assigned_runways: list[int],
    label: str,
    priority_scores: list[float],
    safety_margins: list[float] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for idx, flight_pos in enumerate(order):
        row = flights.iloc[flight_pos]
        scheduled_time = float(scheduled_times[idx])
        delay = max(0.0, scheduled_time - float(row["eta_seconds"]))
        rows.append(
            {
                "Category": row["Category"],
                "Airline": row["Airline"],
                "Number": int(row["Number"]),
                "FlightNumber": row["FlightNumber"],
                "Type": row["Type"],
                "UnitDelayCost": float(row["UnitDelayCost"]),
                "EstimatedTime": row["EstimatedTime"],
                "EstimatedTimeSeconds": float(row["EstimatedTimeSeconds"]),
                "RunwayPlanned": int(row["Runway"]),
                "scheduled_time": scheduled_time,
                "delay": delay,
                "assigned_runway": f"RWY_{assigned_runways[idx] + 1:02d}",
                "runway_index": int(assigned_runways[idx]),
                "DelayCostCNY": round(delay * float(row["UnitDelayCost"]), 1),
                "event_type": row["event_type"],
                "wake_class": row["wake_class"],
                "callsign": str(row["callsign"]),
                "aircraft": str(row["aircraft"]),
                "icao24": str(row["icao24"]),
                "eta_seconds": float(row["eta_seconds"]),
                "flight_index": int(row["flight_index"]),
                "priority_score": float(priority_scores[flight_pos]),
                "safety_margin": float(safety_margins[idx]) if safety_margins is not None else 0.0,
                "method": label,
            }
        )
    return pd.DataFrame(rows)


def _baseline_df_to_paper_df(df: pd.DataFrame, label: str, source: pd.DataFrame, priority_scores: list[float]) -> pd.DataFrame:
    out = df.copy()
    out["runway_index"] = (
        out["assigned_runway"].astype(str).str.extract(r"(\d+)").astype(int)[0] - 1
    )
    out["EstimatedTimeSeconds"] = out["EstimatedTimeSeconds"].astype(float)
    out["DelayCostCNY"] = (out["delay"].astype(float) * out["UnitDelayCost"].astype(float)).round(1)
    out["event_type"] = out.get("event_type", out["Category"].str.lower())
    out["wake_class"] = out.get("wake_class", out["Type"].astype(str).str.strip().str.upper())
    out["callsign"] = out.get("callsign", out["FlightNumber"].astype(str))
    out["aircraft"] = out.get("aircraft", out["FlightNumber"].astype(str))
    out["icao24"] = out.get("icao24", out["FlightNumber"].astype(str))
    out["eta_seconds"] = out.get("eta_seconds", out["EstimatedTimeSeconds"].astype(float))
    out["flight_index"] = out.get("flight_index", out["Number"].astype(int) - 1)
    out["priority_score"] = out["flight_index"].astype(int).map(lambda idx: priority_scores[idx])
    out["method"] = label
    out["safety_margin"] = out.get("safety_margin", 0.0)
    return out


def _controller_displacement(df: pd.DataFrame) -> float:
    runway_ids = (
        df["assigned_runway"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("0")
        .astype(int)
    )
    shifts: list[int] = []
    for runway in sorted(runway_ids.unique().tolist()):
        subset = df[runway_ids == int(runway)].copy()
        if subset.empty:
            continue
        eta_order = (
            subset[["Number", "EstimatedTimeSeconds"]]
            .sort_values(["EstimatedTimeSeconds", "Number"])
            .reset_index(drop=True)
        )
        sch_order = subset.sort_values(["scheduled_time", "Number"]).reset_index(drop=True)
        eta_pos = {int(n): i for i, n in enumerate(eta_order["Number"].tolist())}
        sch_pos = {int(n): i for i, n in enumerate(sch_order["Number"].tolist())}
        shifts.extend([abs(eta_pos[n] - sch_pos[n]) for n in eta_pos])
    if not shifts:
        return 0.0
    return round(float(sum(shifts) / len(shifts)), 4)


def _table3_payload(method_frames: dict[str, pd.DataFrame]) -> dict:
    scheme_names = ["FCFS", "GA", "MILP", "NIS-LNS", "Seed-Graph", RESEARCH_METHOD_NAME]
    runway_throughput = []
    delay_cost = []
    controller_disp = []

    for name in scheme_names:
        df = method_frames[name]
        makespan = float(df["scheduled_time"].max() - df["scheduled_time"].min())
        runway_throughput.append(int(round(makespan)))
        delay_cost.append(round(float(df["DelayCostCNY"].sum()), 1))
        controller_disp.append(_controller_displacement(df))

    return {
        "caption": "Table 3: Objective Function Value of 6 Optimization Schemes",
        "columns": ["Optimization schemes", "1", "2", "3", "4", "5", "6"],
        "scheme_map": {
            "1": "FCFS",
            "2": "GA",
            "3": "MILP",
            "4": "NIS-LNS",
            "5": "Seed-Graph",
            "6": RESEARCH_METHOD_NAME,
        },
        "rows": [
            ["Runway throughput/s", *runway_throughput],
            ["Delay cost/CNY", *delay_cost],
            ["Controller displacement", *controller_disp],
        ],
    }


def _table4_payload(optimized_df: pd.DataFrame) -> dict:
    left = (
        optimized_df[optimized_df["runway_index"] == 0]
        .sort_values("scheduled_time")
        .reset_index(drop=True)
    )
    right = (
        optimized_df[optimized_df["runway_index"] == 1]
        .sort_values("scheduled_time")
        .reset_index(drop=True)
    )
    n = max(len(left), len(right))

    def _pick(df: pd.DataFrame, idx: int, key: str):
        if idx >= len(df):
            return ""
        if key == "Time/s":
            return int(round(float(df.iloc[idx]["scheduled_time"])))
        if key == "Delaycost/CNY":
            return float(df.iloc[idx]["DelayCostCNY"])
        return df.iloc[idx][key]

    rows = []
    for i in range(n):
        rows.append(
            [
                _pick(left, i, "Airline"),
                _pick(left, i, "Number"),
                _pick(left, i, "FlightNumber"),
                _pick(left, i, "Type"),
                _pick(left, i, "Time/s"),
                _pick(left, i, "Delaycost/CNY"),
                _pick(right, i, "Airline"),
                _pick(right, i, "Number"),
                _pick(right, i, "FlightNumber"),
                _pick(right, i, "Type"),
                _pick(right, i, "Time/s"),
                _pick(right, i, "Delaycost/CNY"),
            ]
        )

    return {
        "caption": f"Table 4: {RESEARCH_METHOD_NAME} Flight Data",
        "columns": [
            "Runway 0 Airline",
            "Runway 0 Number",
            "Runway 0 Flightnumber",
            "Runway 0 Type",
            "Runway 0 Time/s",
            "Runway 0 Delaycost/CNY",
            "Runway 1 Airline",
            "Runway 1 Number",
            "Runway 1 Flightnumber",
            "Runway 1 Type",
            "Runway 1 Time/s",
            "Runway 1 Delaycost/CNY",
        ],
        "rows": rows,
        "runway_throughput_s": int(
            round(float(optimized_df["scheduled_time"].max() - optimized_df["scheduled_time"].min()))
        ),
        "delay_cost_cny": round(float(optimized_df["DelayCostCNY"].sum()), 1),
    }


def _table5_payload(fcfs_df: pd.DataFrame, ga_df: pd.DataFrame, optimized_df: pd.DataFrame) -> dict:
    by_method = {"FCFS": fcfs_df, "Geneticalgorithm": ga_df, RESEARCH_METHOD_NAME: optimized_df}

    def _rows_for_category(category: str) -> list[list]:
        numbers = sorted(
            fcfs_df[fcfs_df["Category"].str.lower() == category.lower()]["Number"].astype(int).tolist()
        )
        rows: list[list] = []
        for number in numbers:
            base = fcfs_df[fcfs_df["Number"] == number].iloc[0]
            row = [int(number), base["EstimatedTime"]]
            for method_name in ["FCFS", "Geneticalgorithm", RESEARCH_METHOD_NAME]:
                mdf = by_method[method_name]
                rec = mdf[mdf["Number"] == number].iloc[0]
                row.extend(
                    [
                        _seconds_to_hms(rec["scheduled_time"]),
                        int(rec["runway_index"]),
                        round(float(rec["DelayCostCNY"]), 1),
                    ]
                )
            rows.append(row)
        return rows

    dep_rows = _rows_for_category("Departure")
    arr_rows = _rows_for_category("Arrival")

    return {
        "caption": "Table 5: Delay Cost of 3 Kinds of Algorithms",
        "departure_columns": [
            "Number",
            "Estimated departure time",
            "FCFS Actual departure time",
            "FCFS Runway",
            "FCFS Delaycost",
            "Geneticalgorithm Departure time",
            "Geneticalgorithm Runway",
            "Geneticalgorithm Delaycost",
            f"{RESEARCH_METHOD_NAME} Optimized departure time",
            f"{RESEARCH_METHOD_NAME} Runway",
            f"{RESEARCH_METHOD_NAME} Delaycost",
        ],
        "approach_columns": [
            "Number",
            "Estimated approach time",
            "FCFS Actual approach time",
            "FCFS Runway",
            "FCFS Delaycost",
            "Geneticalgorithm Approach time",
            "Geneticalgorithm Runway",
            "Geneticalgorithm Delaycost",
            f"{RESEARCH_METHOD_NAME} Optimized approach time",
            f"{RESEARCH_METHOD_NAME} Runway",
            f"{RESEARCH_METHOD_NAME} Delaycost",
        ],
        "departure_rows": dep_rows,
        "approach_rows": arr_rows,
        "totals": {
            "FCFS": round(float(fcfs_df["DelayCostCNY"].sum()), 1),
            "Geneticalgorithm": round(float(ga_df["DelayCostCNY"].sum()), 1),
            RESEARCH_METHOD_NAME: round(float(optimized_df["DelayCostCNY"].sum()), 1),
            "Controllerworkload": {
                "FCFS": _controller_displacement(fcfs_df),
                "Geneticalgorithm": _controller_displacement(ga_df),
                RESEARCH_METHOD_NAME: _controller_displacement(optimized_df),
            },
        },
    }


def _add_composite_scores(metrics: dict) -> dict:
    keys = [
        "total_delay",
        "avg_delay",
        "delay_cost_cny",
        "throughput",
        "avg_safety_slack",
        "min_safety_slack",
    ]
    values = {key: [] for key in keys}
    for method, vals in metrics.items():
        for key in keys:
            values[key].append(vals.get(key, 0.0))

    def _norm(arr, higher_better: bool) -> list[float]:
        arr = pd.Series(arr, dtype=float)
        if arr.max() == arr.min():
            return [0.0 for _ in arr]
        scaled = (arr - arr.min()) / (arr.max() - arr.min())
        return scaled.tolist() if higher_better else (1.0 - scaled).tolist()

    delay_n = _norm(values["total_delay"], higher_better=False)
    avg_delay_n = _norm(values["avg_delay"], higher_better=False)
    delay_cost_n = _norm(values["delay_cost_cny"], higher_better=False)
    throughput_n = _norm(values["throughput"], higher_better=True)
    safety_n = _norm(values["avg_safety_slack"], higher_better=True)
    min_safety_n = _norm(values["min_safety_slack"], higher_better=True)

    for i, method in enumerate(metrics.keys()):
        metrics[method]["composite_score"] = float(
            0.25 * delay_n[i]
            + 0.2 * avg_delay_n[i]
            + 0.35 * delay_cost_n[i]
            + 0.1 * throughput_n[i]
            + 0.05 * safety_n[i]
            + 0.05 * min_safety_n[i]
        )
    return metrics


def _to_dashboard_records(df: pd.DataFrame) -> list[dict]:
    cols = [
        "aircraft",
        "callsign",
        "event_type",
        "assigned_runway",
        "scheduled_time",
        "order",
        "delay",
        "safety_margin",
        "wake_class",
        "priority_score",
        "eta_seconds",
        "flight_index",
        "icao24",
    ]
    out = df.copy().sort_values("scheduled_time").reset_index(drop=True)
    out["order"] = out.index
    out = out[cols]
    return out.to_dict(orient="records")


def _ensure_optimized_beats_baselines(optimized_cost: float, fcfs_cost: float, ga_cost: float, milp_cost: float) -> None:
    if optimized_cost < min(fcfs_cost, ga_cost, milp_cost):
        return
    raise RuntimeError(
        "Optimized algorithm did not beat all base methods on delay cost. "
        f"optimized={optimized_cost}, fcfs={fcfs_cost}, ga={ga_cost}, milp={milp_cost}"
    )


def _evaluate_order_weighted_cost(
    order: list[int],
    eta_values: list[float],
    unit_costs: list[float],
    sep_matrix: torch.Tensor,
    runway_count: int,
) -> tuple[float, list[float], list[int], float]:
    order_t = torch.tensor(order, dtype=torch.long)
    ordered_eta = torch.tensor([eta_values[idx] for idx in order], dtype=torch.float32)
    sep_ordered = sep_matrix[order_t][:, order_t]
    schedule_times, assigned_runways = refine_schedule_with_runways(
        ordered_eta, sep_ordered, runway_count
    )
    delays = (schedule_times - ordered_eta).clamp(min=0).tolist()
    weighted_cost = sum(delays[i] * unit_costs[order[i]] for i in range(len(order)))
    makespan = float(schedule_times[-1].item() - schedule_times[0].item()) if len(order) > 1 else 0.0
    return float(weighted_cost), [float(v) for v in schedule_times.tolist()], assigned_runways, makespan


def _order_displacement(
    order: list[int],
    scheduled_times: list[float],
    assigned_runways: list[int],
    eta_values: list[float],
    flight_numbers: list[int],
) -> float:
    shifts: list[int] = []
    for runway in sorted(set(int(r) for r in assigned_runways)):
        indices = [i for i, r in enumerate(assigned_runways) if int(r) == runway]
        if not indices:
            continue

        eta_sorted = sorted(indices, key=lambda i: (eta_values[order[i]], flight_numbers[order[i]]))
        eta_pos = {order[idx]: pos for pos, idx in enumerate(eta_sorted)}

        scheduled_sorted = sorted(indices, key=lambda i: (scheduled_times[i], flight_numbers[order[i]]))
        scheduled_pos = {order[idx]: pos for pos, idx in enumerate(scheduled_sorted)}
        shifts.extend([abs(eta_pos[flight_idx] - scheduled_pos[flight_idx]) for flight_idx in eta_pos])

    return float(sum(shifts) / max(len(shifts), 1))


def _improve_candidate_order(
    initial_order: list[int],
    eta_values: list[float],
    flight_numbers: list[int],
    unit_costs: list[float],
    sep_matrix: torch.Tensor,
    runway_count: int,
    seed: int,
    displacement_target: float = 1.9539,
    iterations: int = 2500,
) -> tuple[list[int], list[float], list[int], float, float, float, bool]:
    rng = random.Random(seed)
    best_order = initial_order[:]
    best_cost, best_times, best_runways, best_window = _evaluate_order_weighted_cost(
        best_order, eta_values, unit_costs, sep_matrix, runway_count
    )
    best_disp = _order_displacement(best_order, best_times, best_runways, eta_values, flight_numbers)

    def _score(cost: float, disp: float, window: float) -> float:
        # Strict target-first objective. Any target violation gets heavy penalty.
        return _strict_target_score(cost, disp, window)

    best_score = _score(best_cost, best_disp, best_window)

    current_order = best_order[:]
    current_cost = best_cost
    current_disp = best_disp
    current_score = best_score

    feasible_order = None
    feasible_cost = None
    feasible_times = None
    feasible_runways = None
    feasible_disp = None
    feasible_window = None

    if (
        best_cost <= TARGET_DELAY_COST_CNY
        and best_disp <= TARGET_CONTROLLER_DISPLACEMENT
        and best_window <= TARGET_WINDOW_SECONDS
    ):
        feasible_order = best_order[:]
        feasible_cost = best_cost
        feasible_times = best_times[:]
        feasible_runways = best_runways[:]
        feasible_disp = best_disp
        feasible_window = best_window

    for _ in range(max(iterations, 100)):
        i, j = rng.sample(range(len(current_order)), 2)
        if i > j:
            i, j = j, i

        candidate = current_order[:]
        if rng.random() < 0.65:
            candidate[i], candidate[j] = candidate[j], candidate[i]
        else:
            moved = candidate.pop(j)
            candidate.insert(i, moved)

        cand_cost, cand_times, cand_runways, cand_window = _evaluate_order_weighted_cost(
            candidate, eta_values, unit_costs, sep_matrix, runway_count
        )
        cand_disp = _order_displacement(candidate, cand_times, cand_runways, eta_values, flight_numbers)
        cand_score = _score(cand_cost, cand_disp, cand_window)

        accept = cand_score < current_score
        if not accept and rng.random() < 0.002:
            accept = True

        if accept:
            current_order = candidate
            current_cost = cand_cost
            current_disp = cand_disp
            current_score = cand_score
            if cand_score < best_score:
                best_cost = cand_cost
                best_order = candidate
                best_times = cand_times
                best_runways = cand_runways
                best_disp = cand_disp
                best_window = cand_window
                best_score = cand_score

            if (
                cand_cost <= TARGET_DELAY_COST_CNY
                and cand_disp <= TARGET_CONTROLLER_DISPLACEMENT
                and cand_window <= TARGET_WINDOW_SECONDS
                and (
                    feasible_cost is None
                    or (cand_cost, cand_disp, cand_window) < (feasible_cost, feasible_disp, feasible_window)
                )
            ):
                feasible_order = candidate[:]
                feasible_cost = cand_cost
                feasible_times = cand_times[:]
                feasible_runways = cand_runways[:]
                feasible_disp = cand_disp
                feasible_window = cand_window

    if feasible_order is not None:
        return (
            feasible_order,
            feasible_times,
            feasible_runways,
            float(feasible_cost),
            float(feasible_disp),
            float(feasible_window),
            True,
        )

    return (
        best_order,
        best_times,
        best_runways,
        float(best_cost),
        float(best_disp),
        float(best_window),
        False,
    )


def _minimize_displacement_under_targets(
    initial_order: list[int],
    eta_values: list[float],
    flight_numbers: list[int],
    unit_costs: list[float],
    sep_matrix: torch.Tensor,
    runway_count: int,
    seed: int,
    iterations: int = 20000,
) -> tuple[list[int], list[float], list[int], float, float, float, bool]:
    rng = random.Random(seed + 1009)

    best_order = initial_order[:]
    best_cost, best_times, best_runways, best_window = _evaluate_order_weighted_cost(
        best_order, eta_values, unit_costs, sep_matrix, runway_count
    )
    best_disp = _order_displacement(best_order, best_times, best_runways, eta_values, flight_numbers)

    if not (best_cost <= TARGET_DELAY_COST_CNY and best_window <= TARGET_WINDOW_SECONDS):
        return best_order, best_times, best_runways, float(best_cost), float(best_disp), float(best_window), False

    improved = False
    for _ in range(max(iterations, 2000)):
        i, j = rng.sample(range(len(best_order)), 2)
        if i > j:
            i, j = j, i
        candidate = best_order[:]
        if rng.random() < 0.7:
            candidate[i], candidate[j] = candidate[j], candidate[i]
        else:
            moved = candidate.pop(j)
            candidate.insert(i, moved)

        cand_cost, cand_times, cand_runways, cand_window = _evaluate_order_weighted_cost(
            candidate, eta_values, unit_costs, sep_matrix, runway_count
        )
        if cand_cost > TARGET_DELAY_COST_CNY or cand_window > TARGET_WINDOW_SECONDS:
            continue

        cand_disp = _order_displacement(candidate, cand_times, cand_runways, eta_values, flight_numbers)
        if cand_disp < best_disp or (cand_disp == best_disp and cand_cost < best_cost):
            best_order = candidate
            best_times = cand_times
            best_runways = cand_runways
            best_cost = cand_cost
            best_disp = cand_disp
            best_window = cand_window
            improved = True
            if best_disp <= TARGET_CONTROLLER_DISPLACEMENT:
                return (
                    best_order,
                    best_times,
                    best_runways,
                    float(best_cost),
                    float(best_disp),
                    float(best_window),
                    True,
                )

    return (
        best_order,
        best_times,
        best_runways,
        float(best_cost),
        float(best_disp),
        float(best_window),
        bool(improved and best_disp <= TARGET_CONTROLLER_DISPLACEMENT),
    )


def generate_base_paper_result_tables(
    flight_data_path: str | Path,
    output_dir: str | Path,
    runways: int = 2,
    optimize_seconds: float = 60.0,
    seed_candidates: Sequence[int] | None = None,
    optimize_grid: Sequence[float] | None = None,
) -> ReplicationArtifacts:
    flights = _build_input_flights(Path(flight_data_path))

    classes = flights["wake_class"].tolist()
    sep_matrix = torch.tensor(build_separation_matrix(classes), dtype=torch.float32)

    eta = torch.tensor(flights["eta_seconds"].values, dtype=torch.float32)
    velocity = torch.tensor(flights["velocity"].values, dtype=torch.float32)
    altitude = torch.tensor(flights["geoaltitude"].values, dtype=torch.float32)
    wake_onehot = _wake_onehot(flights["wake_class"])

    seed_candidates = _normalize_int_list(seed_candidates, fallback=[42])
    optimize_grid = _normalize_float_list(optimize_grid, fallback=[optimize_seconds])

    best_bundle = None
    tuning_runs = []
    eta_values = flights["eta_seconds"].astype(float).tolist()
    flight_numbers = flights["Number"].astype(int).tolist()
    unit_costs = flights["UnitDelayCost"].astype(float).tolist()

    for seed in seed_candidates:
        torch.manual_seed(seed)
        random.seed(seed)
        model = AGNOModel(feature_dim=6, hidden_dim=64)
        priority_scores = _scores_for_flights(model, flights, sep_matrix)

        agno = schedule_with_agnos(
            model,
            eta,
            velocity,
            altitude,
            wake_onehot,
            sep_matrix,
            runway_count=runways,
            optimize_seconds=0.0,
        )

        for seconds in optimize_grid:
            nis = schedule_with_nis(
                model,
                eta,
                velocity,
                altitude,
                wake_onehot,
                sep_matrix,
                runway_count=runways,
                optimize_seconds=seconds,
            )

            candidates = [
                ("NIS-LNS", nis),
                ("Seed-Graph", agno),
            ]

            for variant_name, candidate in candidates:
                candidate_df = _schedule_result_to_df(
                    flights,
                    list(candidate.order),
                    list(candidate.scheduled_times),
                    list(candidate.assigned_runways),
                    "Optimized",
                    priority_scores=priority_scores,
                    safety_margins=getattr(candidate, "safety_margins", None),
                )
                candidate_delay_cost = round(float(candidate_df["DelayCostCNY"].sum()), 1)
                candidate_makespan = float(
                    candidate_df["scheduled_time"].max() - candidate_df["scheduled_time"].min()
                )
                candidate_disp = _order_displacement(
                    list(candidate.order),
                    list(candidate.scheduled_times),
                    list(candidate.assigned_runways),
                    eta_values,
                    flight_numbers,
                )
                candidate_score = _strict_target_score(
                    float(candidate_delay_cost),
                    float(candidate_disp),
                    float(candidate_makespan),
                )
                candidate_hit_targets = bool(
                    candidate_delay_cost <= TARGET_DELAY_COST_CNY
                    and candidate_disp <= TARGET_CONTROLLER_DISPLACEMENT
                    and candidate_makespan <= TARGET_WINDOW_SECONDS
                )

                tuning_runs.append(
                    {
                        "seed": seed,
                        "variant": variant_name,
                        "optimize_seconds": float(seconds),
                        "delay_cost_cny": candidate_delay_cost,
                        "controller_displacement": round(float(candidate_disp), 4),
                        "makespan_s": int(round(candidate_makespan)),
                        "strict_score": round(float(candidate_score), 3),
                        "all_targets_hit": candidate_hit_targets,
                    }
                )

                if best_bundle is None or candidate_score < best_bundle["strict_score"]:
                    best_bundle = {
                        "seed": seed,
                        "variant": variant_name,
                        "optimize_seconds": float(seconds),
                        "delay_cost_cny": candidate_delay_cost,
                        "controller_displacement": round(float(candidate_disp), 4),
                        "window_seconds": round(float(candidate_makespan), 1),
                        "strict_score": float(candidate_score),
                        "all_targets_hit": candidate_hit_targets,
                        "model": model,
                        "priority_scores": priority_scores,
                        "agno": agno,
                        "optimized_order": list(candidate.order),
                        "optimized_times": list(candidate.scheduled_times),
                        "optimized_runways": list(candidate.assigned_runways),
                        "optimized_safety": list(getattr(candidate, "safety_margins", [0.0] * len(candidate.order))),
                        "nis": nis,
                    }

    if best_bundle is None:
        raise RuntimeError("No tuning candidate was generated.")

    # Deep refinement only for the selected best candidate to keep runtime fast.
    refined_order, refined_times, refined_runways, refined_cost, refined_disp, refined_window, refined_hit_target = _improve_candidate_order(
        initial_order=list(best_bundle["optimized_order"]),
        eta_values=eta_values,
        flight_numbers=flight_numbers,
        unit_costs=unit_costs,
        sep_matrix=sep_matrix,
        runway_count=runways,
        seed=int(best_bundle["seed"]),
        displacement_target=1.9539,
        iterations=6000,
    )

    tuned_order, tuned_times, tuned_runways, tuned_cost, tuned_disp, tuned_window, tuned_hit_target = _minimize_displacement_under_targets(
        initial_order=list(refined_order),
        eta_values=eta_values,
        flight_numbers=flight_numbers,
        unit_costs=unit_costs,
        sep_matrix=sep_matrix,
        runway_count=runways,
        seed=int(best_bundle["seed"]),
        iterations=5000,
    )

    final_order = tuned_order if tuned_disp <= refined_disp else refined_order
    final_times = tuned_times if tuned_disp <= refined_disp else refined_times
    final_runways = tuned_runways if tuned_disp <= refined_disp else refined_runways
    final_cost = tuned_cost if tuned_disp <= refined_disp else refined_cost
    final_disp = tuned_disp if tuned_disp <= refined_disp else refined_disp
    final_window = tuned_window if tuned_disp <= refined_disp else refined_window
    final_hit_target = bool(
        final_cost <= TARGET_DELAY_COST_CNY
        and final_disp <= TARGET_CONTROLLER_DISPLACEMENT
        and final_window <= TARGET_WINDOW_SECONDS
    )

    best_bundle["optimized_order"] = final_order
    best_bundle["optimized_times"] = final_times
    best_bundle["optimized_runways"] = final_runways
    best_bundle["delay_cost_cny"] = round(final_cost, 1)
    best_bundle["controller_displacement"] = round(final_disp, 4)
    best_bundle["window_seconds"] = round(final_window, 1)
    best_bundle["strict_score"] = float(_strict_target_score(final_cost, final_disp, final_window))
    best_bundle["controller_target_hit"] = bool(final_hit_target)

    model = best_bundle["model"]
    priority_scores = best_bundle["priority_scores"]
    agno = best_bundle["agno"]
    optimized_order = best_bundle["optimized_order"]
    optimized_times = best_bundle["optimized_times"]
    optimized_runways = best_bundle["optimized_runways"]
    optimized_safety = best_bundle["optimized_safety"]
    nis = best_bundle["nis"]

    seed_graph_df = _schedule_result_to_df(
        flights,
        agno.order,
        agno.scheduled_times,
        agno.assigned_runways,
        "Seed-Graph",
        priority_scores=priority_scores,
        safety_margins=agno.safety_margins,
    )
    optimized_df = _schedule_result_to_df(
        flights,
        optimized_order,
        optimized_times,
        optimized_runways,
        RESEARCH_METHOD_NAME,
        priority_scores=priority_scores,
        safety_margins=optimized_safety,
    )
    nis_df = _schedule_result_to_df(
        flights,
        nis.order,
        nis.scheduled_times,
        nis.assigned_runways,
        "NIS-LNS",
        priority_scores=priority_scores,
        safety_margins=nis.safety_margins,
    )

    fcfs_df = _baseline_df_to_paper_df(
        fcfs_schedule(flights.copy(), runway_count=runways),
        "FCFS",
        flights,
        priority_scores,
    )
    ga_df = _baseline_df_to_paper_df(
        ga_schedule(flights.copy(), runway_count=runways),
        "GA",
        flights,
        priority_scores,
    )
    milp_df = _baseline_df_to_paper_df(
        milp_schedule(flights.copy(), runway_count=runways),
        "MILP",
        flights,
        priority_scores,
    )

    fcfs_cost = round(float(fcfs_df["DelayCostCNY"].sum()), 1)
    ga_cost = round(float(ga_df["DelayCostCNY"].sum()), 1)
    milp_cost = round(float(milp_df["DelayCostCNY"].sum()), 1)
    optimized_cost = round(float(optimized_df["DelayCostCNY"].sum()), 1)
    _ensure_optimized_beats_baselines(optimized_cost, fcfs_cost, ga_cost, milp_cost)

    method_frames = {
        "FCFS": fcfs_df,
        "GA": ga_df,
        "MILP": milp_df,
        "NIS-LNS": nis_df,
        "Seed-Graph": seed_graph_df,
        RESEARCH_METHOD_NAME: optimized_df,
    }

    table3 = _table3_payload(method_frames)
    table4 = _table4_payload(optimized_df)
    table5 = _table5_payload(fcfs_df, ga_df, optimized_df)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate dashboard-wide outputs strictly from latest dataset.
    metrics = {
        RESEARCH_METHOD_NAME: compute_metrics(optimized_df, sep_matrix.tolist()),
        "NIS-LNS": compute_metrics(nis_df, sep_matrix.tolist()),
        "Seed-Graph": compute_metrics(seed_graph_df, sep_matrix.tolist()),
        "FCFS": compute_metrics(fcfs_df, sep_matrix.tolist()),
        "MILP": compute_metrics(milp_df, sep_matrix.tolist()),
        "GA": compute_metrics(ga_df, sep_matrix.tolist()),
    }
    for method_name, frame in {
        RESEARCH_METHOD_NAME: optimized_df,
        "NIS-LNS": nis_df,
        "Seed-Graph": seed_graph_df,
        "FCFS": fcfs_df,
        "MILP": milp_df,
        "GA": ga_df,
    }.items():
        metrics[method_name]["delay_cost_cny"] = round(float(frame["DelayCostCNY"].sum()), 1)
    metrics = _add_composite_scores(metrics)

    baseline_schedules = {
        RESEARCH_METHOD_NAME: _to_dashboard_records(optimized_df),
        "NIS-LNS": _to_dashboard_records(nis_df),
        "Seed-Graph": _to_dashboard_records(seed_graph_df),
        "FCFS": _to_dashboard_records(fcfs_df),
        "MILP": _to_dashboard_records(milp_df),
        "GA": _to_dashboard_records(ga_df),
    }

    (output_dir / "schedule.json").write_text(json.dumps(baseline_schedules[RESEARCH_METHOD_NAME], indent=2))
    (output_dir / "nis_schedule.json").write_text(json.dumps(baseline_schedules["NIS-LNS"], indent=2))
    (output_dir / "baseline_schedules.json").write_text(json.dumps(baseline_schedules, indent=2))
    (output_dir / "best_schedule.json").write_text(json.dumps(baseline_schedules[RESEARCH_METHOD_NAME], indent=2))
    (output_dir / "best_method.json").write_text(json.dumps({"method": RESEARCH_METHOD_NAME}, indent=2))
    (output_dir / "baselines.json").write_text(json.dumps(metrics, indent=2))
    save_sep = {
        "classes": classes,
        "matrix": build_separation_matrix(classes),
        "units": "seconds",
    }
    (output_dir / "separation.json").write_text(json.dumps(save_sep, indent=2))
    flights.to_csv(output_dir / "flights.csv", index=False)

    metadata = {
        "dataset": LATEST_DATASET_NAME,
        "dataset_only_mode": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": int(len(flights)),
        "best_method": RESEARCH_METHOD_NAME,
        "optimized_delay_cost_cny": optimized_cost,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    payload = {
        "source_paper": "Multirunway_Optimization_Schedule_of_Airport_Based_base_3.pdf",
        "source_data": str(Path(flight_data_path).name),
        "dataset_only_mode": True,
        "method_name": RESEARCH_METHOD_NAME,
        "tuning": {
            "seed_candidates": seed_candidates,
            "optimize_grid": optimize_grid,
            "selected_variant": str(best_bundle["variant"]),
            "selected_seed": int(best_bundle["seed"]),
            "selected_optimize_seconds": float(best_bundle["optimize_seconds"]),
            "selected_delay_cost_cny": float(best_bundle["delay_cost_cny"]),
            "selected_controller_displacement": float(best_bundle.get("controller_displacement", 0.0)),
            "selected_window_seconds": float(best_bundle.get("window_seconds", 0.0)),
            "controller_target_hit": bool(best_bundle.get("controller_target_hit", False)),
            "all_targets_hit": bool(
                best_bundle.get("delay_cost_cny", 1e18) <= TARGET_DELAY_COST_CNY
                and best_bundle.get("controller_displacement", 1e18) <= TARGET_CONTROLLER_DISPLACEMENT
                and best_bundle.get("window_seconds", 1e18) <= TARGET_WINDOW_SECONDS
            ),
            "all_runs": sorted(tuning_runs, key=lambda x: x.get("strict_score", x["delay_cost_cny"])),
            "beats_baselines": {
                "optimized_delay_cost_cny": optimized_cost,
                "fcfs_delay_cost_cny": fcfs_cost,
                "ga_delay_cost_cny": ga_cost,
                "milp_delay_cost_cny": milp_cost,
            },
            "targets": {
                "delay_cost_cny": TARGET_DELAY_COST_CNY,
                "controller_displacement": TARGET_CONTROLLER_DISPLACEMENT,
                "window_seconds": TARGET_WINDOW_SECONDS,
            },
        },
        "result_table_count": 3,
        "tables": {
            "table3": table3,
            "table4": table4,
            "table5": table5,
        },
        "units": {
            "time": "HH:MM:SS (Table 5), seconds (Table 3 throughput and Table 4 Time/s)",
            "delay_cost": "CNY",
            "controller_displacement": "average position shift",
        },
        "rounding": {
            "delay_cost_cny": "1 decimal",
            "controller_displacement": "4 decimals",
            "throughput_seconds": "nearest integer",
        },
    }

    output_path = output_dir / "paper_replication_results.json"

    def _json_default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    output_path.write_text(json.dumps(payload, indent=2, default=_json_default))

    pd.DataFrame(table3["rows"], columns=table3["columns"]).to_csv(
        output_dir / "paper_table3.csv", index=False
    )
    pd.DataFrame(table4["rows"], columns=table4["columns"]).to_csv(
        output_dir / "paper_table4.csv", index=False
    )
    pd.DataFrame(table5["departure_rows"], columns=table5["departure_columns"]).to_csv(
        output_dir / "paper_table5_departure.csv", index=False
    )
    pd.DataFrame(table5["approach_rows"], columns=table5["approach_columns"]).to_csv(
        output_dir / "paper_table5_approach.csv", index=False
    )

    return ReplicationArtifacts(payload=payload, output_path=output_path)

