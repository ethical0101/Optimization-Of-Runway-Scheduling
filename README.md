# Optimization Of Runway Scheduling

This repository delivers a full runway scheduling pipeline centered on the final optimized method **AERIS-OPT** (Adaptive Efficiency Runway Intelligence Scheduler).

It includes:
- strict latest-dataset workflow,
- multi-method benchmarking (FCFS, GA, MILP, NIS-LNS, Seed-Graph),
- paper-style replicated result tables,
- and a Streamlit dashboard with analytics and export support.

## Optimized Solution Summary

The optimized solution is built as a **target-first multi-factor scheduler**.

Primary objective:
- minimize delay cost (CNY).

Hard/strict optimization targets:
- delay cost <= `41369.5` CNY,
- controller displacement <= `1.9539`,
- schedule window <= `1022` seconds.

Latest generated result (from `flight_data.csv`):
- method: `AERIS-OPT`
- delay cost: `37472.9` CNY
- controller displacement: `1.8333`
- schedule window: `900` seconds
- all strict targets hit: `true`

## Current Benchmark Snapshot

From `aeris_opt/outputs/baselines.json`:

| Method | Delay Cost (CNY) | Total Delay | Avg Delay | Makespan (s) | Composite Score |
|---|---:|---:|---:|---:|---:|
| AERIS-OPT | 37472.9 | 5081.0 | 211.71 | 900.0 | 0.9277 |
| NIS-LNS | 74519.2 | 5840.0 | 243.33 | 960.0 | 0.8567 |
| FCFS | 121868.2 | 5840.0 | 243.33 | 990.0 | 0.7723 |
| MILP | 121868.2 | 5840.0 | 243.33 | 990.0 | 0.7723 |
| GA | 228560.1 | 13189.0 | 549.54 | 1308.0 | 0.1268 |
| Seed-Graph | 297446.2 | 8510.0 | 354.58 | 1140.0 | 0.3922 |

## Architecture And Flow

1. Load validated flight dataset.
2. Build wake-class separation matrix.
3. Generate candidate schedules from NIS-LNS and Seed-Graph over seed/time grid.
4. Score candidates with strict target-penalty function.
5. Apply deep local refinement for weighted delay cost.
6. Apply displacement-focused refinement under delay/window constraints.
7. Select final `AERIS-OPT` schedule.
8. Validate baseline dominance (`AERIS-OPT` beats FCFS, GA, MILP on delay cost).
9. Emit all dashboard and paper artifacts.

## Dataset Policy (Strict)

The pipeline is locked to:
- `flight_data.csv`

Enforcement behavior:
- generation rejects any other dataset file name,
- outputs carry dataset metadata lock fields,
- dashboard refuses to load stale or non-compliant artifacts.

## Installation

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install pandas numpy torch plotly streamlit
```

## Run Commands

Full pipeline (generate + tune + dashboard):

```powershell
python -m aeris_opt.main
```

Generate only:

```powershell
python -m aeris_opt.main --no-ui
```

Direct generator with explicit tuning grid:

```powershell
python -m aeris_opt.generate_paper_results --data flight_data.csv --output aeris_opt/outputs --runways 2 --optimize-seconds 120 --seed-candidates 7,21,42,84,126 --optimize-grid 30,60,90,120
```

Dashboard only:

```powershell
python -m streamlit run aeris_opt/ui/app.py
```

## Key Output Artifacts

Under `aeris_opt/outputs/`:

- `metadata.json`: dataset lock + generation metadata.
- `best_method.json`: final selected method (`AERIS-OPT`).
- `schedule.json`, `best_schedule.json`: final optimized schedule.
- `baselines.json`: full metrics for all methods.
- `baseline_schedules.json`: schedule rows per method.
- `separation.json`: wake separation matrix.
- `paper_replication_results.json`: full paper-table payload + tuning evidence.
- `paper_table3.csv`, `paper_table4.csv`, `paper_table5_departure.csv`, `paper_table5_approach.csv`.

## Dashboard Capabilities

`aeris_opt/ui/app.py` provides:

- arrivals/departures timeline,
- runway utilization and free-interval visualizations,
- delay analytics and wake-class distribution,
- conflict heatmap,
- critical/emergency scenario analysis,
- paper-style tables (Table 3/4/5 layout),
- export buttons: Excel, Word-ready RTF, PDF-ready HTML,
- optimized accuracy comparison against paper reference targets.

## Validation Checklist

After generation, validate:

1. Dataset lock:
- `metadata.json` has `"dataset": "flight_data.csv"` and `"dataset_only_mode": true`.

2. Paper lock:
- `paper_replication_results.json` has `"source_data": "flight_data.csv"` and `"dataset_only_mode": true`.

3. Strict-target success:
- `paper_replication_results.json -> tuning -> all_targets_hit` is `true`.

4. Baseline dominance:
- `paper_replication_results.json -> tuning -> beats_baselines` confirms optimized delay cost is lower than FCFS, GA, MILP.

## Troubleshooting

Generation command exits with code 1:
- check that input file is exactly `flight_data.csv`.
- confirm required columns exist in dataset.

Generation is slow:
- reduce tuning grid, for example:
  - `--seed-candidates 42,84`
  - `--optimize-grid 30,60`

Dashboard refuses to load:
- regenerate outputs using the direct generator command.

Streamlit port conflict:

```powershell
python -m streamlit run aeris_opt/ui/app.py --server.port 8503
```

