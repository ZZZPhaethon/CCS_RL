from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .dataset import ForecastSample


@dataclass(frozen=True)
class LSTMTrainingConfig:
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 128
    max_epochs: int = 100
    patience: int = 8
    min_delta: float = 1e-4
    device: str = "auto"
    progress: bool = True


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_mae: float
    train_rmse: float
    validation_loss: float
    validation_mae: float
    validation_rmse: float
    best_validation_mae: float
    epochs_without_improvement: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_mae": self.train_mae,
            "train_rmse": self.train_rmse,
            "validation_loss": self.validation_loss,
            "validation_mae": self.validation_mae,
            "validation_rmse": self.validation_rmse,
            "best_validation_mae": self.best_validation_mae,
            "epochs_without_improvement": self.epochs_without_improvement,
        }


@dataclass
class SequenceStandardizer:
    x_mean: object
    x_std: object
    y_mean: object
    y_std: object

    @classmethod
    def fit(cls, samples: list[ForecastSample], torch):
        x = torch.tensor([sample.history for sample in samples], dtype=torch.float32)
        y = torch.tensor([sample.target for sample in samples], dtype=torch.float32)
        x_mean = x.reshape(-1, x.shape[-1]).mean(dim=0)
        x_std = x.reshape(-1, x.shape[-1]).std(dim=0).clamp_min(1e-6)
        y_mean = y.mean()
        y_std = y.std().clamp_min(1e-6)
        return cls(x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)

    @classmethod
    def from_state_dict(cls, state: dict[str, object]) -> "SequenceStandardizer":
        return cls(
            x_mean=state["x_mean"],
            x_std=state["x_std"],
            y_mean=state["y_mean"],
            y_std=state["y_std"],
        )

    def transform_x(self, x):
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def transform_y(self, y):
        return (y - self.y_mean.to(y.device)) / self.y_std.to(y.device)

    def inverse_y(self, y):
        return y * self.y_std.to(y.device) + self.y_mean.to(y.device)

    def state_dict(self) -> dict[str, object]:
        return {
            "x_mean": self.x_mean,
            "x_std": self.x_std,
            "y_mean": self.y_mean,
            "y_std": self.y_std,
        }


def train_lstm_forecaster(
    train_samples: list[ForecastSample],
    validation_samples: list[ForecastSample],
    *,
    config: LSTMTrainingConfig | None = None,
):
    """Train an LSTM with normalization, dropout, weight decay, and early stopping."""
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    if not train_samples:
        raise ValueError("train_samples must not be empty")
    if not validation_samples:
        raise ValueError("validation_samples must not be empty for early stopping")
    config = config or LSTMTrainingConfig()
    device = _resolve_device(config.device, torch)
    scaler = SequenceStandardizer.fit(train_samples, torch)

    train_loader = DataLoader(
        _tensor_dataset(train_samples, torch, TensorDataset, scaler=scaler),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        _tensor_dataset(validation_samples, torch, TensorDataset, scaler=scaler),
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    input_size = len(train_samples[0].history[0])
    horizon = len(train_samples[0].target)
    model = LSTMForecaster(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        horizon=horizon,
        nn=nn,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    history: list[EpochMetrics] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation_mae = float("inf")
    epochs_without_improvement = 0

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_meter = _MetricMeter()
        train_loss_sum = 0.0
        train_batches = 0
        iterator = train_loader
        if config.progress and tqdm is not None:
            iterator = tqdm(train_loader, desc=f"epoch {epoch}/{config.max_epochs}", unit="batch")
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted_norm = model(batch_x)
            loss = loss_fn(predicted_norm, batch_y)
            loss.backward()
            optimizer.step()

            predicted = scaler.inverse_y(predicted_norm.detach())
            actual = scaler.inverse_y(batch_y.detach())
            train_meter.update(predicted, actual)
            train_loss_sum += float(loss.detach().cpu())
            train_batches += 1
            if config.progress and tqdm is not None:
                iterator.set_postfix(
                    loss=f"{train_loss_sum / max(1, train_batches):.4f}",
                    mae=f"{train_meter.mae:.4f}",
                    rmse=f"{train_meter.rmse:.4f}",
                )

        validation_meter, validation_loss = evaluate_lstm_loader(
            model,
            validation_loader,
            scaler=scaler,
            device=device,
            loss_fn=loss_fn,
        )
        improved = validation_meter.mae + config.min_delta < best_validation_mae
        if improved:
            best_validation_mae = validation_meter.mae
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss_sum / max(1, train_batches),
            train_mae=train_meter.mae,
            train_rmse=train_meter.rmse,
            validation_loss=validation_loss,
            validation_mae=validation_meter.mae,
            validation_rmse=validation_meter.rmse,
            best_validation_mae=best_validation_mae,
            epochs_without_improvement=epochs_without_improvement,
        )
        history.append(metrics)
        print(
            f"epoch {epoch:03d} | "
            f"train loss={metrics.train_loss:.4f} mae={metrics.train_mae:.4f} rmse={metrics.train_rmse:.4f} | "
            f"val loss={metrics.validation_loss:.4f} mae={metrics.validation_mae:.4f} "
            f"rmse={metrics.validation_rmse:.4f} | "
            f"best_val_mae={metrics.best_validation_mae:.4f} "
            f"patience={metrics.epochs_without_improvement}/{config.patience}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            print(f"early stopping at epoch {epoch}; restoring best validation model", flush=True)
            break

    model.load_state_dict(best_state)
    return model, history, scaler


def evaluate_lstm_samples(
    model,
    samples: list[ForecastSample],
    *,
    scaler: SequenceStandardizer,
    batch_size: int = 256,
    device: str = "auto",
):
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    resolved_device = _resolve_device(device, torch)
    model.to(resolved_device)
    loader = DataLoader(
        _tensor_dataset(samples, torch, TensorDataset, scaler=scaler),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=resolved_device.type == "cuda",
    )
    meter, _loss = evaluate_lstm_loader(
        model,
        loader,
        scaler=scaler,
        device=resolved_device,
        loss_fn=nn.MSELoss(),
    )
    return meter


def evaluate_lstm_loader(model, loader, *, scaler: SequenceStandardizer, device, loss_fn):
    model.eval()
    meter = _MetricMeter()
    loss_sum = 0.0
    batches = 0
    torch = _torch()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            predicted_norm = model(batch_x)
            loss_sum += float(loss_fn(predicted_norm, batch_y).detach().cpu())
            batches += 1
            meter.update(scaler.inverse_y(predicted_norm), scaler.inverse_y(batch_y))
    return meter, loss_sum / max(1, batches)


def predict_lstm(
    model,
    samples: list[ForecastSample],
    *,
    scaler: SequenceStandardizer,
    batch_size: int = 256,
    device: str = "auto",
):
    torch, _nn, DataLoader, TensorDataset = _torch_modules()
    resolved_device = _resolve_device(device, torch)
    model.to(resolved_device)
    loader = DataLoader(
        _tensor_dataset(samples, torch, TensorDataset, scaler=scaler),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=resolved_device.type == "cuda",
    )
    model.eval()
    predictions: list[list[float]] = []
    with torch.no_grad():
        for batch_x, _batch_y in loader:
            batch_x = batch_x.to(resolved_device, non_blocking=True)
            predictions.extend(scaler.inverse_y(model(batch_x)).cpu().tolist())
    return predictions


def save_training_history(history: list[EpochMetrics], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].as_dict()) if history else ["epoch"])
        writer.writeheader()
        for metrics in history:
            writer.writerow(metrics.as_dict())
    return output


def LSTMForecaster(
    *,
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    horizon: int,
    nn,
):
    class _LSTMForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_size, horizon)

        def forward(self, x):
            _, (hidden, _cell) = self.lstm(x)
            return self.head(self.dropout(hidden[-1]))

    return _LSTMForecaster()


class _MetricMeter:
    def __init__(self) -> None:
        self.absolute_error = 0.0
        self.squared_error = 0.0
        self.count = 0

    def update(self, predicted, actual) -> None:
        error = predicted - actual
        self.absolute_error += float(error.abs().sum().detach().cpu())
        self.squared_error += float((error * error).sum().detach().cpu())
        self.count += int(error.numel())

    @property
    def mae(self) -> float:
        return self.absolute_error / self.count if self.count else 0.0

    @property
    def mse(self) -> float:
        return self.squared_error / self.count if self.count else 0.0

    @property
    def rmse(self) -> float:
        return math.sqrt(self.mse)


def _tensor_dataset(samples: list[ForecastSample], torch, TensorDataset, *, scaler: SequenceStandardizer):
    x, y = _tensors(samples, torch)
    return TensorDataset(scaler.transform_x(x), scaler.transform_y(y))


def _tensors(samples: list[ForecastSample], torch):
    x = torch.tensor([sample.history for sample in samples], dtype=torch.float32)
    y = torch.tensor([sample.target for sample in samples], dtype=torch.float32)
    return x, y


def _resolve_device(device: str, torch):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return resolved


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LSTM forecasting requires PyTorch (`pip install torch`).") from exc
    return torch


def _torch_modules():
    torch = _torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset
