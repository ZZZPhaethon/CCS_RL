from __future__ import annotations

import argparse
from pathlib import Path

from .preprocessing import RouteWaveDatasetConfig, write_phase1_leg_wave_dataset
from .routes import RouteWaveConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Phase 1 leg-level wave-height dataset.")
    parser.add_argument("--wave-dir", type=Path, default=Path("D:/wave Height"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/wave_height/phase1_leg_wave_2010_2014.csv"),
    )
    parser.add_argument("--start-record", type=int, default=0)
    parser.add_argument("--hours", type=int, default=None)
    parser.add_argument("--sample-spacing-km", type=float, default=75.0)
    parser.add_argument("--aggregations", nargs="+", default=["mean", "p75", "p90", "max"])
    parser.add_argument("--no-emitter-to-emitter", action="store_true")
    parser.add_argument("--no-terminal-to-emitter", action="store_true")
    args = parser.parse_args()

    config = RouteWaveDatasetConfig(
        wave_config=RouteWaveConfig(sample_spacing_km=args.sample_spacing_km, aggregation="p75"),
        aggregations=tuple(args.aggregations),
        start_record=args.start_record,
        hours=args.hours,
    )
    output = write_phase1_leg_wave_dataset(
        args.wave_dir,
        args.output,
        config=config,
        progress=lambda message: print(message, flush=True),
        include_emitter_to_emitter=not args.no_emitter_to_emitter,
        include_terminal_to_emitter=not args.no_terminal_to_emitter,
    )
    print(f"saved leg-level wave dataset: {output}", flush=True)


if __name__ == "__main__":
    main()
