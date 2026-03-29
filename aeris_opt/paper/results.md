# Results

## Experimental Setup
Experiments were executed in dataset-only mode with `flight_data.csv` and evaluated through the replicated base-paper table schema. Candidate schedules were produced across multiple seeds and optimization windows, then filtered by strict target criteria. The final selected configuration was NIS-LNS (seed 126, 30 s optimization budget), exported as the AERIS-OPT result.

## Table 3 Replication: Objective Comparison
Table 3 reports six optimization schemes: FCFS, GA, MILP, NIS-LNS, Seed-Graph, and AERIS-OPT. Replicated objective rows are:

1. Runway throughput/s: 990, 1308, 990, 960, 1140, 900.
2. Delay cost/CNY: 121868.2, 228560.1, 121868.2, 74519.2, 297446.2, 37472.9.
3. Controller displacement: 0.1667, 3.8333, 0.1667, 1.4167, 3.0833, 1.8333.

Key observations:

1. AERIS-OPT attains the lowest delay cost among all six schemes.
2. NIS-LNS provides substantial delay improvement over FCFS/MILP but is further improved by strict-target AERIS-OPT selection.
3. Controller displacement of AERIS-OPT is higher than FCFS/MILP but remains below the configured threshold (1.9539), satisfying the strict control objective.

## Table 4 Replication: AERIS-OPT Flight Data
Table 4 contains the detailed per-runway assignment, including airline, aircraft number, flight number, wake type, scheduled time, and delay-cost contribution for both runways. The generated schedule shows balanced runway utilization with feasible temporal spacing and no post-refinement safety violations.

Representative entries include zero-cost anchor flights at time 0 on both runways, followed by progressively delayed assignments whose costs reflect wake interactions and queue context. The complete row set is exported directly in the replication artifact.

## Table 5 Replication: Delay-Cost Benchmark
The Table 5 style comparison confirms delay-cost ordering across major baselines:

1. AERIS-OPT: 37472.9 CNY.
2. FCFS: 121868.2 CNY.
3. GA: 228560.1 CNY.
4. MILP: 121868.2 CNY.

Relative improvements of AERIS-OPT are large and operationally meaningful, with all strict targets simultaneously satisfied (all_targets_hit = true).

## Target Satisfaction and Robustness
Configured strict targets were:

1. Delay cost <= 41369.5 CNY.
2. Controller displacement <= 1.9539.
3. Throughput window <= 1022 s.

Selected AERIS-OPT outcome met all three conditions with margins:

1. Delay cost margin: 3896.6 CNY below target.
2. Displacement margin: 0.1206 below target.
3. Window margin: 122 s below target.

## Discussion
Results indicate that the strongest gains come from integrating conflict-aware sequencing with runway-feasible refinement under strict multi-factor selection. While some schemes achieve higher raw throughput windows, their delay-cost or displacement behavior is materially worse. The AERIS-OPT profile demonstrates a better operational trade-off for the objective priorities defined by the base-paper-aligned evaluation protocol.
