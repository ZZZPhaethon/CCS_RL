from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .baselines import evaluate_persistence, evaluate_seasonal_climatology
from .dataset import DEFAULT_TARGET_COLUMN, WaveRouteDataset
from .lstm import (
    LSTMTrainingConfig,
    evaluate_lstm_samples,
    predict_lstm,
    save_training_history,
    train_lstm_forecaster,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an LSTM wave-height forecaster with early stopping.")
    parser.add_argument("--csv", type=Path, default=Path("output/wave_height/phase1_route_wave_2010_2014.csv"))
    parser.add_argument("--lookback-hours", type=int, default=168)
    parser.add_argument("--horizon-hours", type=int, default=168)
    parser.add_argument("--target-column", default=DEFAULT_TARGET_COLUMN)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--model-output", type=Path, default=Path("output/wave_height/wave_height_lstm.pt"))
    parser.add_argument("--history-output", type=Path, default=Path("output/wave_height/wave_height_lstm_history.csv"))
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=Path("output/wave_height/wave_height_lstm_test_predictions.csv"),
    )
    args = parser.parse_args()

    dataset = WaveRouteDataset.from_csv(args.csv)
    splits = dataset.split_by_years(
        train_years=(2010, 2011, 2012),
        validation_years=(2013,),
        test_years=(2014,),
    )
    print("building supervised windows ...", flush=True)
    train_samples = splits["train"].make_samples(
        lookback_hours=args.lookback_hours,
        horizon_hours=args.horizon_hours,
        target_column=args.target_column,
    )
    validation_samples = splits["validation"].make_samples(
        lookback_hours=args.lookback_hours,
        horizon_hours=args.horizon_hours,
        target_column=args.target_column,
    )
    test_samples = splits["test"].make_samples(
        lookback_hours=args.lookback_hours,
        horizon_hours=args.horizon_hours,
        target_column=args.target_column,
    )
    print(
        f"samples: train={len(train_samples)}, validation={len(validation_samples)}, test={len(test_samples)}; "
        f"lookback={args.lookback_hours}, horizon={args.horizon_hours}",
        flush=True,
    )

    print("baselines:", flush=True)
    validation_persistence = evaluate_persistence(validation_samples)
    validation_seasonal = evaluate_seasonal_climatology(splits["train"], validation_samples)
    test_persistence = evaluate_persistence(test_samples)
    test_seasonal = evaluate_seasonal_climatology(splits["train"], test_samples)
    print(f"  validation persistence: mae={validation_persistence.mae:.4f}, rmse={validation_persistence.rmse:.4f}")
    print(f"  validation seasonal   : mae={validation_seasonal.mae:.4f}, rmse={validation_seasonal.rmse:.4f}")
    print(f"  test persistence      : mae={test_persistence.mae:.4f}, rmse={test_persistence.rmse:.4f}")
    print(f"  test seasonal         : mae={test_seasonal.mae:.4f}, rmse={test_seasonal.rmse:.4f}")

    config = LSTMTrainingConfig(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        device=args.device,
        progress=not args.no_progress,
    )
    model, history, scaler = train_lstm_forecaster(train_samples, validation_samples, config=config)

    import torch

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "lookback_hours": args.lookback_hours,
            "horizon_hours": args.horizon_hours,
            "target_column": args.target_column,
            "input_size": len(train_samples[0].history[0]),
        },
        args.model_output,
    )
    save_training_history(history, args.history_output)
    print(f"saved best model: {args.model_output}", flush=True)
    print(f"saved history: {args.history_output}", flush=True)

    test_metrics = evaluate_lstm_samples(
        model,
        test_samples,
        scaler=scaler,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        f"test LSTM best model: mae={test_metrics.mae:.4f}, rmse={test_metrics.rmse:.4f}, mse={test_metrics.mse:.4f}",
        flush=True,
    )
    _write_prediction_preview(
        args.predictions_output,
        test_samples,
        predict_lstm(model, test_samples, scaler=scaler, batch_size=args.batch_size, device=args.device),
    )
    print(f"saved test prediction preview: {args.predictions_output}", flush=True)


def _write_prediction_preview(path: Path, samples, predictions, max_samples: int = 1000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_index", "vessel_id", "horizon_index", "global_record", "actual", "predicted"],
        )
        writer.writeheader()
        for sample_index, (sample, prediction) in enumerate(zip(samples[:max_samples], predictions[:max_samples])):
            for horizon_index, (actual, predicted) in enumerate(zip(sample.target, prediction)):
                writer.writerow(
                    {
                        "sample_index": sample_index,
                        "vessel_id": sample.vessel_id,
                        "horizon_index": horizon_index,
                        "global_record": sample.future_global_record[horizon_index],
                        "actual": actual,
                        "predicted": predicted,
                    }
                )


if __name__ == "__main__":
    main()
