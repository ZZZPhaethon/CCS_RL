from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path

from sim.ship_speed import NORTHERN_LIGHTS_SHIP, speed_factor


WAVE_COLUMNS = ("hs_mean_m", "hs_p75_m", "hs_p90_m", "hs_max_m")
OUTPUT_FIELDS = (
    "timestamp_utc",
    "year",
    "month",
    "day",
    "hour_utc",
    "global_record",
    "source_file",
    "source_record",
    "aggregation_scope",
    "unique_undirected_leg_count",
    "total_unique_leg_distance_km",
    "nominal_speed_knots",
    *WAVE_COLUMNS,
    "speed_factor_mean",
    "speed_factor_p75",
    "speed_factor_p90",
    "speed_factor_max",
)


def _parse_year(source_file: str) -> int:
    stem = Path(source_file).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Cannot determine year from source file {source_file!r}") from exc


def _aggregate_hour(rows: list[dict[str, str]]) -> dict[str, object]:
    first = rows[0]
    undirected: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = tuple(sorted((row["origin"], row["destination"])))
        undirected.setdefault(key, []).append(row)

    legs: list[dict[str, float]] = []
    for pair_rows in undirected.values():
        count = len(pair_rows)
        leg = {
            "distance_km": sum(float(row["distance_km"]) for row in pair_rows) / count,
        }
        for column in WAVE_COLUMNS:
            leg[column] = sum(float(row[column]) for row in pair_rows) / count
        legs.append(leg)

    total_distance = sum(leg["distance_km"] for leg in legs)
    if total_distance <= 0:
        raise ValueError(f"Non-positive total distance at global record {first['global_record']}")

    wave = {
        column: sum(leg[column] * leg["distance_km"] for leg in legs) / total_distance
        for column in WAVE_COLUMNS[:-1]
    }
    wave["hs_max_m"] = max(leg["hs_max_m"] for leg in legs)

    year = _parse_year(first["source_file"])
    source_record = int(first["source_record"])
    timestamp = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(hours=source_record)
    nominal_speed_knots = float(first["speed_knots"])

    result: dict[str, object] = {
        "timestamp_utc": timestamp.isoformat().replace("+00:00", "Z"),
        "year": timestamp.year,
        "month": timestamp.month,
        "day": timestamp.day,
        "hour_utc": timestamp.hour,
        "global_record": int(first["global_record"]),
        "source_file": first["source_file"],
        "source_record": source_record,
        "aggregation_scope": "unique_undirected_leg_coverage_proxy",
        "unique_undirected_leg_count": len(legs),
        "total_unique_leg_distance_km": total_distance,
        "nominal_speed_knots": nominal_speed_knots,
        **wave,
    }
    for name in ("mean", "p75", "p90", "max"):
        result[f"speed_factor_{name}"] = speed_factor(
            wave[f"hs_{name}_m"],
            NORTHERN_LIGHTS_SHIP,
            nominal_speed_knots=nominal_speed_knots,
        )
    return result


def build(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with input_path.open(newline="", encoding="utf-8") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        missing = set(("global_record", "source_file", "source_record", "origin", "destination", "distance_km", "speed_knots", *WAVE_COLUMNS)) - set(
            reader.fieldnames or ()
        )
        if missing:
            raise ValueError(f"Input is missing columns: {sorted(missing)}")
        writer = csv.DictWriter(target, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for _, group in groupby(reader, key=lambda row: int(row["global_record"])):
            writer.writerow(_aggregate_hour(list(group)))
            row_count += 1
    return row_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one leg-coverage proxy row per hour.")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    rows = build(args.input_csv, args.output_csv)
    print(f"Wrote {rows} hourly rows to {args.output_csv}")


if __name__ == "__main__":
    main()
