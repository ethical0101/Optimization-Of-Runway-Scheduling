# AERIS-OPT Runway Module Guide

This module implements the complete optimization engine and visualization layer for the runway scheduling system.

Final optimized method name:
- `AERIS-OPT`

## Core Design

`AERIS-OPT` is not a single heuristic pass. It is a staged optimization process:

1. Candidate generation across seeds and optimization windows.
2. Strict target-first scoring (delay, displacement, window).
3. Local deep refinement for weighted delay-cost minimization.
4. Secondary refinement focused on controller displacement under hard constraints.
5. Final benchmark validation against FCFS, GA, MILP.

## Strict Target Configuration

Configured in `paper/base_results.py`:

- `TARGET_DELAY_COST_CNY = 41369.5`
- `TARGET_CONTROLLER_DISPLACEMENT = 1.9539`
- `TARGET_WINDOW_SECONDS = 1022.0`

Candidate ranking uses a strict penalty score to discourage any target violation before fine-grained cost optimization.

## Latest Run Outcome

From generated artifacts:

- selected variant: `NIS-LNS`
- selected seed: `126`
- selected optimize seconds: `30.0`
- delay cost: `37472.9`
- controller displacement: `1.8333`
- schedule window: `900.0`
- all targets hit: `true`

## Dataset Enforcement

This module only accepts:
- `flight_data.csv`

Any other data filename is rejected by validation logic.

## Commands

Run full pipeline:

```powershell
python -m aeris_opt.main
```

Run generation without UI:

```powershell
python -m aeris_opt.main --no-ui
```

Run direct tuned generation:

```powershell
python -m aeris_opt.generate_paper_results --data flight_data.csv --output aeris_opt/outputs --runways 2 --optimize-seconds 120 --seed-candidates 7,21,42,84,126 --optimize-grid 30,60,90,120
```

Run dashboard only:

```powershell
python -m streamlit run aeris_opt/ui/app.py
```

## Implementation Map

- `paper/base_results.py`
	- dataset validation
	- candidate generation/tuning
	- strict score computation
	- displacement-aware refinement
	- benchmark checks
	- paper table payload generation

- `optimizer/robust_refiner.py`
	- per-runway feasible scheduling with separation enforcement

- `ui/app.py`
	- dataset-lock checks before render
	- paper-format result tables
	- analytics panels and export actions

- `generate_paper_results.py`
	- CLI wrapper for controlled batch generation

- `main.py`
	- orchestration entrypoint for generation + optional dashboard launch

## Output Contract

Required output files in `outputs/`:

- `metadata.json`
- `best_method.json`
- `best_schedule.json`
- `schedule.json`
- `baselines.json`
- `baseline_schedules.json`
- `separation.json`
- `paper_replication_results.json`
- `paper_table3.csv`
- `paper_table4.csv`
- `paper_table5_departure.csv`
- `paper_table5_approach.csv`

## Validation Steps

1. Verify dataset lock in `metadata.json`:
- `dataset == flight_data.csv`
- `dataset_only_mode == true`

2. Verify paper payload integrity in `paper_replication_results.json`:
- `source_data == flight_data.csv`
- `dataset_only_mode == true`
- `method_name == AERIS-OPT`

3. Verify strict optimization success:
- `tuning.all_targets_hit == true`

4. Verify baseline superiority:
- `tuning.beats_baselines.optimized_delay_cost_cny` is lower than FCFS, GA, and MILP values.

## Troubleshooting

Generation failure:
- confirm `flight_data.csv` exists at repo root.
- confirm file columns match expected schema.

Long runtime:
- reduce search space:
	- `--seed-candidates 42,84`
	- `--optimize-grid 30,60`

UI not loading results:
- regenerate outputs and ensure latest files are present in `outputs/`.

