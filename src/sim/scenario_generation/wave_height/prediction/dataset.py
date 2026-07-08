from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_FEATURE_COLUMNS = (
    "hs_mean_m",
    "hs_p75_m",
    "hs_p90_m",
    "hs_max_m",
    "speed_factor_p75",
)
DEFAULT_TARGET_COLUMN = "hs_p75_m"


@dataclass(frozen=True)
class WaveRouteRecord:
    """One vessel-route wave-height row from the exported CSV."""

    global_record: int
    source_file: str
    source_record: int
    vessel_id: str
    origin: str
    destination: str
    values: dict[str, float]

    @property
    def year(self) -> int | None:
        match = re.search(r"(20\d{2})", self.source_file)
        return int(match.group(1)) if match else None

    @property
    def hour_of_day(self) -> int:
        return self.source_record % 24

    @property
    def hour_of_year(self) -> int:
        return self.source_record


@dataclass(frozen=True)
class ForecastSample:
    """A supervised sample: historical features and future target trajectory."""

    vessel_id: str
    start_global_record: int
    history: list[list[float]]
    target: list[float]
    future_hour_of_year: list[int]
    future_global_record: list[int]


class WaveRouteDataset:
    """Route-level wave-height dataset prepared from ``phase1_route_wave_*.csv``."""

    def __init__(self, records: list[WaveRouteRecord]) -> None:
        self.records = records

    @classmethod
    def from_csv(cls, path: str | Path) -> "WaveRouteDataset":
        records: list[WaveRouteRecord] = []
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                values = {
                    key: float(value)
                    for key, value in row.items()
                    if key.startswith("hs_") or key.startswith("speed_factor_")
                }
                records.append(
                    WaveRouteRecord(
                        global_record=int(row["global_record"]),
                        source_file=row["source_file"],
                        source_record=int(row["source_record"]),
                        vessel_id=row["vessel_id"],
                        origin=row.get("origin", ""),
                        destination=row.get("destination", ""),
                        values=values,
                    )
                )
        return cls(records)

    def by_vessel(self) -> dict[str, list[WaveRouteRecord]]:
        grouped: dict[str, list[WaveRouteRecord]] = {}
        for record in self.records:
            grouped.setdefault(record.vessel_id, []).append(record)
        for vessel_records in grouped.values():
            vessel_records.sort(key=lambda record: record.global_record)
        return grouped

    def split_by_years(
        self,
        *,
        train_years: Iterable[int],
        validation_years: Iterable[int] = (),
        test_years: Iterable[int] = (),
    ) -> dict[str, "WaveRouteDataset"]:
        train = set(train_years)
        validation = set(validation_years)
        test = set(test_years)
        splits = {"train": [], "validation": [], "test": []}
        for record in self.records:
            year = record.year
            if year in train:
                splits["train"].append(record)
            elif year in validation:
                splits["validation"].append(record)
            elif year in test:
                splits["test"].append(record)
        return {name: WaveRouteDataset(records) for name, records in splits.items()}

    def make_samples(
        self,
        *,
        lookback_hours: int = 72,
        horizon_hours: int = 24,
        feature_columns: Iterable[str] = DEFAULT_FEATURE_COLUMNS,
        target_column: str = DEFAULT_TARGET_COLUMN,
        include_time_features: bool = True,
    ) -> list[ForecastSample]:
        if lookback_hours <= 0:
            raise ValueError("lookback_hours must be positive")
        if horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        columns = tuple(feature_columns)
        samples: list[ForecastSample] = []
        for vessel_id, records in self.by_vessel().items():
            for index in range(lookback_hours, len(records) - horizon_hours + 1):
                history_records = records[index - lookback_hours:index]
                future_records = records[index:index + horizon_hours]
                if not _is_contiguous([*history_records, *future_records]):
                    continue
                history = [
                    _feature_vector(record, columns, include_time_features=include_time_features)
                    for record in history_records
                ]
                target = [record.values[target_column] for record in future_records]
                samples.append(
                    ForecastSample(
                        vessel_id=vessel_id,
                        start_global_record=future_records[0].global_record,
                        history=history,
                        target=target,
                        future_hour_of_year=[record.hour_of_year for record in future_records],
                        future_global_record=[record.global_record for record in future_records],
                    )
                )
        return samples


def _feature_vector(
    record: WaveRouteRecord,
    columns: tuple[str, ...],
    *,
    include_time_features: bool,
) -> list[float]:
    features = [record.values[column] for column in columns]
    if include_time_features:
        features.extend(_cyclical_features(record.hour_of_day, 24))
        # 8784 handles leap years; non-leap years simply never occupy the final day.
        features.extend(_cyclical_features(record.hour_of_year, 8784))
    return features


def _cyclical_features(value: int, period: int) -> list[float]:
    angle = 2.0 * math.pi * (value % period) / period
    return [math.sin(angle), math.cos(angle)]


def _is_contiguous(records: list[WaveRouteRecord]) -> bool:
    return all(
        b.global_record == a.global_record + 1
        for a, b in zip(records, records[1:])
    )
