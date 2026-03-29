# Literature Review

Runway scheduling research generally follows three methodological streams: rule-based sequencing, optimization-based search, and learning-assisted decision models. These streams are frequently compared along delay reduction, computational tractability, and ability to maintain safety constraints under traffic variability.

Rule-based dispatching, especially FCFS, remains a foundational operational baseline because it is interpretable, easy to implement, and stable in low-to-moderate demand conditions. Its drawback is objective blindness: FCFS does not explicitly optimize system-wide delay cost or runway resource balance, which can amplify inefficiency during congestion.

Metaheuristic solvers such as GA broaden the search space and often improve over deterministic dispatching by exploring nonlocal sequencing alternatives. Prior work on multi-runway genetic optimization demonstrates competitive objective values, but outcome quality strongly depends on configuration choices such as population design, mutation pressure, and stop conditions. This sensitivity can complicate robust transfer across datasets and operating regimes.

Exact mathematical programming, including MILP families, provides stronger optimization structure and interpretable constraints. These methods are valuable for principled schedule construction but often require simplifications for real-time use, especially when high-resolution interaction effects are included. In dense traffic windows, computational scaling and model fidelity remain an active tension.

Recent learning-driven approaches, including graph-informed policies, aim to encode aircraft interaction structure directly and reduce repeated combinatorial solving. This direction is particularly relevant for runway scheduling because conflict relationships are naturally relational and evolve with assignment and timing decisions. However, learned rankings alone are insufficient unless paired with explicit feasibility mechanisms that respect wake separation and runway occupancy constraints.

Position of this work in the literature:

1. It keeps FCFS, GA, and MILP as mandatory baselines to preserve interpretability and continuity with standard evaluation practice.
2. It evaluates intermediate learned strategies (NIS-LNS and Seed-Graph) to expose the contribution of progressively richer optimization components.
3. It introduces AERIS-OPT as a strict-target, displacement-aware variant that integrates conflict-aware ordering with runway-feasible refinement.

This comparative framing enables a clearer attribution of performance gains and supports direct alignment with the base-paper objective tables.
