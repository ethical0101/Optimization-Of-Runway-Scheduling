# Introduction

Runway scheduling is one of the most consequential tactical decisions in airport operations because it directly influences departure queues, approach holding, fuel burn, and downstream network reliability. The decision space is constrained by wake-separation rules, runway occupancy interactions, and mixed traffic priorities, making the problem inherently combinatorial and highly dynamic. Classical optimization and heuristic methods remain widely used, but they often expose a trade-off between computational tractability and responsiveness under changing traffic states.

The base reference paper used in this project reports performance through three core result tables: objective values across multiple schemes, a detailed optimized runway allocation table, and per-flight delay-cost comparisons for major baselines. To provide a methodologically consistent comparison, this work reproduces that reporting structure and keeps the same practical focus on delay cost, throughput window, and controller workload.

This paper introduces AERIS-OPT, a graph-informed multi-runway scheduler designed to improve delay-cost performance while preserving safety feasibility. Instead of relying solely on fixed-priority heuristics, AERIS-OPT uses conflict-aware scoring and constrained refinement to produce schedules that remain operationally valid and quantitatively superior under the selected metrics.

Contributions of this study are as follows:

1. A base-paper-aligned evaluation pipeline that reproduces Tables 3-5 with consistent grouping, units, and objective interpretation.
2. A strict-target optimization strategy combining candidate search, weighted refinement, and displacement-aware post-optimization.
3. A runway-specific feasibility mechanism that enforces separation constraints at assignment time, improving both correctness and optimization quality.
4. A comparative benchmark against FCFS, GA, MILP, and intermediate learned variants (NIS-LNS, Seed-Graph), with reproducible outputs and dashboard integration.

The remainder of this document reviews related work, presents the method, and reports quantitative findings with reference-consistent table outputs.
