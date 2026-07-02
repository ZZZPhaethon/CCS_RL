from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from .dataset import DEFAULT_TARGET_COLUMN, ForecastSample, WaveRouteDataset
from .lstm import LSTMForecaster, SequenceStandardizer, predict_lstm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling LSTM wave-height forecasts.")
    parser.add_argument("--csv", type=Path, default=Path("output/wave_height/phase1_route_wave_2010_2014.csv"))
    parser.add_argument("--model", type=Path, default=Path("output/wave_height/wave_height_lstm_168h.pt"))
    parser.add_argument("--target-year", type=int, default=2014)
    parser.add_argument("--lookback-hours", type=int, default=None)
    parser.add_argument("--horizon-hours", type=int, default=None)
    parser.add_argument("--replan-every-hours", type=int, default=24)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/wave_height/wave_height_lstm_2014_rolling24_predictions.csv"),
    )
    args = parser.parse_args()

    import torch
    from torch import nn

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    lookback_hours = int(args.lookback_hours or checkpoint["lookback_hours"])
    horizon_hours = int(args.horizon_hours or checkpoint["horizon_hours"])
    input_size = int(checkpoint["input_size"])
    config = checkpoint["config"]
    scaler = SequenceStandardizer.from_state_dict(checkpoint["scaler"])
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        horizon=horizon_hours,
        nn=nn,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    dataset = WaveRouteDataset.from_csv(args.csv)
    samples = _rolling_target_year_samples(
        dataset,
        target_year=args.target_year,
        lookback_hours=lookback_hours,
        horizon_hours=horizon_hours,
        replan_every_hours=args.replan_every_hours,
        target_column=args.target_column,
    )
    print(
        f"predicting {len(samples)} rolling samples: "
        f"year={args.target_year}, lookback={lookback_hours}, horizon={horizon_hours}, "
        f"replan_every={args.replan_every_hours}",
        flush=True,
    )
    predictions = predict_lstm(
        model,
        samples,
        scaler=scaler,
        batch_size=args.batch_size,
        device=args.device,
    )
    metrics = _write_predictions(args.output, samples, predictions)
    print(
        f"saved predictions: {args.output}\n"
        f"rolling forecast MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, count={metrics['count']}",
        flush=True,
    )


def _rolling_target_year_samples(
    dataset: WaveRouteDataset,
    *,
    target_year: int,
    lookback_hours: int,
    horizon_hours: int,
    replan_every_hours: int,
    target_column: str,
) -> list[ForecastSample]:
    target_records = [record for record in dataset.records if record.year == target_year]
    if not target_records:
        raise ValueError(f"No rows found for target_year={target_year}.")
    first_target_record = min(record.global_record for record in target_records)
    last_target_record = max(record.global_record for record in target_records)
    all_samples = dataset.make_samples(
        lookback_hours=lookback_hours,
        horizon_hours=horizon_hours,
        target_column=target_column,
    )
    samples = [
        sample
        for sample in all_samples
        if (
            first_target_record <= sample.start_global_record
            and sample.future_global_record[-1] <= last_target_record
            and (sample.start_global_record - first_target_record) % replan_every_hours == 0
        )
    ]
    if not samples:
        raise ValueError("No rolling samples were generated. Check lookback/horizon/replan settings.")
    return samples


def _write_predictions(path: Path, samples: list[ForecastSample], predictions: list[list[float]]) -> dict[str, float | int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    abs_error = 0.0
    sq_error = 0.0
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_index",
                "vessel_id",
                "horizon_index",
                "global_record",
                "actual",
                "predicted",
                "error",
            ],
        )
        writer.writeheader()
        for sample_index, (sample, prediction) in enumerate(zip(samples, predictions)):
            for horizon_index, (actual, predicted) in enumerate(zip(sample.target, prediction)):
                error = predicted - actual
                abs_error += abs(error)
                sq_error += error * error
                count += 1
                writer.writerow(
                    {
                        "sample_index": sample_index,
                        "vessel_id": sample.vessel_id,
                        "horizon_index": horizon_index,
                        "global_record": sample.future_global_record[horizon_index],
                        "actual": actual,
                        "predicted": predicted,
                        "error": error,
                    }
                )
    return {
        "mae": abs_error / count if count else 0.0,
        "rmse": math.sqrt(sq_error / count) if count else 0.0,
        "count": count,
    }


if __name__ == "__main__":
    main()
