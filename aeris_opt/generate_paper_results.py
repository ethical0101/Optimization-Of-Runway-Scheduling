from __future__ import annotations

import argparse
from pathlib import Path

from aeris_opt.paper.base_results import generate_base_paper_result_tables


def _parse_int_csv(values: str) -> list[int]:
    return [int(v.strip()) for v in values.split(",") if v.strip()]


def _parse_float_csv(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="flight_data.csv")
    parser.add_argument("--output", type=str, default="aeris_opt/outputs")
    parser.add_argument("--runways", type=int, default=2)
    parser.add_argument("--optimize-seconds", type=float, default=60.0)
    parser.add_argument("--seed-candidates", type=str, default="42")
    parser.add_argument("--optimize-grid", type=str, default="60")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = root / data_path

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    artifacts = generate_base_paper_result_tables(
        flight_data_path=data_path,
        output_dir=output_path,
        runways=args.runways,
        optimize_seconds=args.optimize_seconds,
        seed_candidates=_parse_int_csv(args.seed_candidates),
        optimize_grid=_parse_float_csv(args.optimize_grid),
    )
    print(f"Generated replicated paper tables at: {artifacts.output_path}")
    tuning = artifacts.payload.get("tuning", {})
    if tuning:
        print(
            "Best tuned optimizer: "
            f"variant={tuning.get('selected_variant')} "
            f"seed={tuning.get('selected_seed')} "
            f"optimize_seconds={tuning.get('selected_optimize_seconds')} "
            f"delay_cost_cny={tuning.get('selected_delay_cost_cny')}"
        )


if __name__ == "__main__":
    main()

