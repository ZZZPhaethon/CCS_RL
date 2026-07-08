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
from .train_lstm import _write_prediction_preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-year-out LSTM cross validation.")
    parser.add_argument("--csv", type=Path, default=Path("output/wave_height/phase1_route_wave_2010_2014.csv"))
    parser.add_argument("--years", type=int, nargs="+", default=[2010, 2011, 2012, 2013, 2014])
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("output/wave_height/cv_lstm_168h"))
    args = parser.parse_args()

    dataset = WaveRouteDataset.from_csv(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for test_year in args.years:
        validation_year = _previous_year(args.years, test_year)
        train_years = [year for year in args.years if year not in {test_year, validation_year}]
        print(
            f"\n=== fold test={test_year} validation={validation_year} train={train_years} ===",
            flush=True,
        )
        splits = dataset.split_by_years(
            train_years=train_years,
            validation_years=(validation_year,),
            test_years=(test_year,),
        )
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
            f"samples: train={len(train_samples)}, validation={len(validation_samples)}, test={len(test_samples)}",
            flush=True,
        )
        test_persistence = evaluate_persistence(test_samples)
        test_seasonal = evaluate_seasonal_climatology(splits["train"], test_samples)
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
        test_metrics = evaluate_lstm_samples(
            model,
            test_samples,
            scaler=scaler,
            batch_size=args.batch_size,
            device=args.device,
        )
        fold_prefix = args.output_dir / f"test_{test_year}_val_{validation_year}"
        _save_fold_model(
            fold_prefix.with_suffix(".pt"),
            model=model,
            scaler=scaler,
            config=config,
            lookback_hours=args.lookback_hours,
            horizon_hours=args.horizon_hours,
            target_column=args.target_column,
            input_size=len(train_samples[0].history[0]),
        )
        save_training_history(history, fold_prefix.with_name(f"{fold_prefix.name}_history.csv"))
        _write_prediction_preview(
            fold_prefix.with_name(f"{fold_prefix.name}_predictions.csv"),
            test_samples,
            predict_lstm(model, test_samples, scaler=scaler, batch_size=args.batch_size, device=args.device),
        )
        row = {
            "test_year": test_year,
            "validation_year": validation_year,
            "train_years": " ".join(str(year) for year in train_years),
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "test_samples": len(test_samples),
            "lstm_mae": test_metrics.mae,
            "lstm_rmse": test_metrics.rmse,
            "persistence_mae": test_persistence.mae,
            "persistence_rmse": test_persistence.rmse,
            "seasonal_mae": test_seasonal.mae,
            "seasonal_rmse": test_seasonal.rmse,
            "best_validation_mae": min(metric.validation_mae for metric in history),
            "epochs": len(history),
        }
        summary_rows.append(row)
        _write_summary(args.output_dir / "summary.csv", summary_rows)
        print(
            f"fold {test_year}: LSTM mae={test_metrics.mae:.4f}, rmse={test_metrics.rmse:.4f}; "
            f"persistence mae={test_persistence.mae:.4f}, seasonal mae={test_seasonal.mae:.4f}",
            flush=True,
        )

    _write_summary(args.output_dir / "summary.csv", summary_rows)
    print(f"\nwrote summary: {args.output_dir / 'summary.csv'}", flush=True)


def _previous_year(years: list[int], test_year: int) -> int:
    ordered = sorted(years)
    index = ordered.index(test_year)
    return ordered[index - 1]


def _save_fold_model(path: Path, **payload) -> None:
    import torch

    model = payload.pop("model")
    scaler = payload.pop("scaler")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler.state_dict(),
            **payload,
        },
        path,
    )


def _write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
