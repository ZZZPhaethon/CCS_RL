from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

from .dataset import DEFAULT_TARGET_COLUMN, ForecastSample, WaveRouteDataset, WaveRouteRecord


@dataclass(frozen=True)
class ForecastMetrics:
    mae: float
    rmse: float
    count: int

    def as_dict(self) -> dict[str, float | int]:
        return {"mae": self.mae, "rmse": self.rmse, "count": self.count}


class SeasonalClimatology:
    """Mean historical wave height by vessel and hour-of-year."""

    def __init__(self, target_column: str = DEFAULT_TARGET_COLUMN) -> None:
        self.target_column = target_column
        self._mean_by_key: dict[tuple[str, int], float] = {}
        self._mean_by_vessel: dict[str, float] = {}
        self._global_mean = 0.0

    def fit(self, dataset: WaveRouteDataset) -> "SeasonalClimatology":
        sums: dict[tuple[str, int], float] = defaultdict(float)
        counts: dict[tuple[str, int], int] = defaultdict(int)
        vessel_sums: dict[str, float] = defaultdict(float)
        vessel_counts: dict[str, int] = defaultdict(int)
        total = 0.0
        total_count = 0
        for record in dataset.records:
            value = record.values[self.target_column]
            key = (record.vessel_id, record.hour_of_year)
            sums[key] += value
            counts[key] += 1
            vessel_sums[record.vessel_id] += value
            vessel_counts[record.vessel_id] += 1
            total += value
            total_count += 1
        self._mean_by_key = {key: sums[key] / counts[key] for key in sums}
        self._mean_by_vessel = {
            vessel_id: vessel_sums[vessel_id] / vessel_counts[vessel_id]
            for vessel_id in vessel_sums
        }
        self._global_mean = total / total_count if total_count else 0.0
        return self

    def predict_sample(self, sample: ForecastSample) -> list[float]:
        vessel_mean = self._mean_by_vessel.get(sample.vessel_id, self._global_mean)
        return [
            self._mean_by_key.get((sample.vessel_id, hour_of_year), vessel_mean)
            for hour_of_year in sample.future_hour_of_year
        ]


def persistence_forecast(sample: ForecastSample, target_feature_index: int = 1) -> list[float]:
    """Repeat the most recent observed target value over the whole horizon.

    With the default feature columns, index 1 is ``hs_p75_m``.
    """
    last_value = sample.history[-1][target_feature_index]
    return [last_value] * len(sample.target)


def evaluate_predictions(samples: list[ForecastSample], predictions: list[list[float]]) -> ForecastMetrics:
    if len(samples) != len(predictions):
        raise ValueError("samples and predictions must have the same length")
    absolute_error = 0.0
    squared_error = 0.0
    count = 0
    for sample, predicted in zip(samples, predictions):
        if len(sample.target) != len(predicted):
            raise ValueError("Each prediction must match the sample horizon length")
        for actual, forecast in zip(sample.target, predicted):
            error = forecast - actual
            absolute_error += abs(error)
            squared_error += error * error
            count += 1
    if count == 0:
        return ForecastMetrics(mae=0.0, rmse=0.0, count=0)
    return ForecastMetrics(mae=absolute_error / count, rmse=math.sqrt(squared_error / count), count=count)


def evaluate_persistence(samples: list[ForecastSample], target_feature_index: int = 1) -> ForecastMetrics:
    predictions = [persistence_forecast(sample, target_feature_index=target_feature_index) for sample in samples]
    return evaluate_predictions(samples, predictions)


def evaluate_seasonal_climatology(
    train_dataset: WaveRouteDataset,
    samples: list[ForecastSample],
    *,
    target_column: str = DEFAULT_TARGET_COLUMN,
) -> ForecastMetrics:
    model = SeasonalClimatology(target_column=target_column).fit(train_dataset)
    predictions = [model.predict_sample(sample) for sample in samples]
    return evaluate_predictions(samples, predictions)


def record_summary(records: list[WaveRouteRecord], target_column: str = DEFAULT_TARGET_COLUMN) -> dict[str, float]:
    values = [record.values[target_column] for record in records]
    if not values:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
