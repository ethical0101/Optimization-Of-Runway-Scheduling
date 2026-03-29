from __future__ import annotations

import torch


def refine_schedule_with_runways(
    schedule_times: torch.Tensor, sep_matrix: torch.Tensor, runway_count: int
) -> tuple[torch.Tensor, list[int]]:
    times = schedule_times.clone()
    n = times.numel()
    runway_available = [0.0 for _ in range(runway_count)]
    runway_flights: list[list[int]] = [[] for _ in range(runway_count)]
    runways = [0 for _ in range(n)]

    for i in range(n):
        candidate_times = []
        for runway_idx in range(runway_count):
            t = max(float(times[i].item() if isinstance(times[i], torch.Tensor) else times[i]), runway_available[runway_idx])
            if runway_flights[runway_idx]:
                required = max(
                    float(
                        (times[j].item() if isinstance(times[j], torch.Tensor) else times[j])
                        + float(sep_matrix[i, j].item() if isinstance(sep_matrix[i, j], torch.Tensor) else sep_matrix[i, j])
                    )
                    for j in runway_flights[runway_idx]
                )
                t = max(t, required)
            candidate_times.append(t)

        runway_idx = min(range(runway_count), key=lambda r: candidate_times[r])
        chosen_time = candidate_times[runway_idx]
        times[i] = chosen_time
        runway_available[runway_idx] = chosen_time
        runway_flights[runway_idx].append(i)
        runways[i] = runway_idx

    return times, runways


def refine_schedule(
    schedule_times: torch.Tensor, sep_matrix: torch.Tensor, runway_count: int
) -> torch.Tensor:
    times, _ = refine_schedule_with_runways(schedule_times, sep_matrix, runway_count)
    return times
