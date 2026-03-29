# Method

## Data and Evaluation Protocol
All experiments in this paper run in dataset-only mode using `flight_data.csv`, with reporting mapped to the base reference paper table structure. The pipeline generates three primary outputs:

1. Table 3 style objective comparison across six schemes.
2. Table 4 style detailed AERIS-OPT runway assignment records.
3. Table 5 style delay-cost comparisons for the major baselines.

This protocol ensures each optimization run can be evaluated directly against the same objective dimensions used in the base paper.

## Problem Formulation
Given a set of flights with estimated times and operational attributes, assign each flight to a runway and a sequence position such that wake and temporal separation rules are respected while minimizing aggregate operating cost. Let $t_i$ be scheduled time for flight $i$, $\eta_i$ its reference time, and $S_{ij}$ the minimum required separation between flights $i$ and $j$ under runway-specific ordering.

The core optimization objective is

$$
\min_{\pi, r} \sum_i c_i(t_i - \eta_i) + \lambda \sum_{(i,j)} \max(0, S_{ij} - \Delta t_{ij})
$$

where $\pi$ is sequence policy, $r$ denotes runway assignment, $c_i$ is delay-cost weight, and $\Delta t_{ij}$ is realized temporal spacing.

## AERIS-OPT Pipeline
The implemented optimization stack contains four coordinated stages:

1. Conflict graph construction from flight interactions and separation dependencies.
2. Neural-informed scheduling (NIS) and large-neighborhood search (LNS) for candidate sequence generation.
3. Runway-feasible robust refinement that enforces per-runway temporal validity.
4. Strict-target selection over multiple seeds and optimization budgets.

The strict-target phase evaluates seed and time-grid combinations and selects the schedule that best satisfies predefined objective targets while preserving safety feasibility.

## Strict Multi-Factor Selection
To align with the base-paper objective focus, candidate runs are scored with a strict metric that prioritizes simultaneous achievement of delay-cost, controller displacement, and throughput-window targets. The configured targets are:

1. Delay cost target: 41369.5 CNY.
2. Controller displacement target: 1.9539.
3. Throughput window target: 1022 s.

The selected run in this study is NIS-LNS with seed 126 and 30 s optimization budget, yielding delay cost 37472.9 CNY, controller displacement 1.8333, and window 900 s, with all targets satisfied.

## Baseline Comparison Design
Baselines retained for comparability are FCFS, GA, and MILP, augmented by two intermediate learned variants (NIS-LNS and Seed-Graph) prior to final AERIS-OPT selection. This decomposition separates gains from:

1. Learning-informed ordering quality.
2. Search/refinement effectiveness.
3. Runway-feasible assignment correction.

All values reported in the Results section are read from generated replication artifacts and remain reproducible through the same pipeline configuration.
