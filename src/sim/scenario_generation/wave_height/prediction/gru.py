from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .dataset import ForecastSample


@dataclass(frozen=True)
class GRUTrainingConfig:
    hidden_size: int = 64
    num_layers: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 20
    device: str = "auto"
    progress: bool = True


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_mse: float
    train_mae: float
    train_rmse: float
    validation_mse: float
    validation_mae: float
    validation_rmse: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "epoch": self.epoch,
            "train_mse": self.train_mse,
            "train_mae": self.train_mae,
            "train_rmse": self.train_rmse,
            "validation_mse": self.validation_mse,
            "validation_mae": self.validation_mae,
            "validation_rmse": self.validation_rmse,
        }


def train_gru_forecaster(
    train_samples: list[ForecastSample],
    validation_samples: list[ForecastSample],
    *,
    config: GRUTrainingConfig | None = None,
):
    """Train a GRU forecaster with progress bars and per-epoch metrics."""
    torch, nn, DataLoader, TensorDataset = _torch_modules()
    if not train_samples:
        raise ValueError("train_samples must not be empty")
    config = config or GRUTrainingConfig()
    device = _resolve_device(config.device, torch)

    train_loader = DataLoader(
        _tensor_dataset(train_samples, torch, TensorDataset),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        _tensor_dataset(validation_samples, torch, TensorDataset),
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    input_size = len(train_samples[0].history[0])
    horizon = len(train_samples[0].target)
    model = GRUForecaster(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        horizon=horizon,
        nn=nn,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    history: list[EpochMetrics] = []

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional display nicety
        tqdm = None

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_meter = _MetricMeter()
        iterator = train_loader
        if config.progress and tqdm is not None:
            iterator = tqdm(train_loader, desc=f"epoch {epoch}/{config.epochs}", unit="batch")
        for batch_x, batch_y in iterator:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch_x)
            loss = loss_fn(predicted, batch_y)
            loss.backward()
            optimizer.step()
            train_meter.update(predicted.detach(), batch_y.detach())
            if config.progress and tqdm is not None:
                iterator.set_postfix(
                    mse=f"{train_meter.mse:.4f}",
                    mae=f"{train_meter.mae:.4f}",
                    rmse=f"{train_meter.rmse:.4f}",
                )

        validation_meter = evaluate_gru_loader(model, validation_loader, device=device)
        metrics = EpochMetrics(
            epoch=epoch,
            train_mse=train_meter.mse,
            train_mae=train_meter.mae,
            train_rmse=train_meter.rmse,
            validation_mse=validation_meter.mse,
            validation_mae=validation_meter.mae,
            validation_rmse=validation_meter.rmse,
        )
        history.append(metrics)
        print(
            f"epoch {epoch:03d} | "
            f"train mae={metrics.train_mae:.4f} rmse={metrics.train_rmse:.4f} mse={metrics.train_mse:.4f} | "
            f"val mae={metrics.validation_mae:.4f} rmse={metrics.validation_rmse:.4f} mse={metrics.validation_mse:.4f}",
            flush=True,
        )
    return model, history


def evaluate_gru_samples(
    model,
    samples: list[ForecastSample],
    *,
    batch_size: int = 256,
    device: str = "auto",
):
    torch, _nn, DataLoader, TensorDataset = _torch_modules()
    resolved_device = _resolve_device(device, torch)
    model.to(resolved_device)
    loader = DataLoader(
        _tensor_dataset(samples, torch, TensorDataset),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=resolved_device.type == "cuda",
    )
    return evaluate_gru_loader(model, loader, device=resolved_device)


def evaluate_gru_loader(model, loader, *, device):
    model.eval()
    meter = _MetricMeter()
    torch = _torch()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            meter.update(model(batch_x), batch_y)
    return meter


def predict_gru(model, samples: list[ForecastSample], *, batch_size: int = 256, device: str = "auto"):
    torch, _nn, DataLoader, TensorDataset = _torch_modules()
    resolved_device = _resolve_device(device, torch)
    model.to(resolved_device)
    loader = DataLoader(
        _tensor_dataset(samples, torch, TensorDataset),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=resolved_device.type == "cuda",
    )
    model.eval()
    predictions: list[list[float]] = []
    with torch.no_grad():
        for batch_x, _batch_y in loader:
            batch_x = batch_x.to(resolved_device, non_blocking=True)
            predictions.extend(model(batch_x).cpu().tolist())
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


def GRUForecaster(*, input_size: int, hidden_size: int, num_layers: int, horizon: int, nn):
    class _GRUForecaster(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, horizon)

        def forward(self, x):
            _, hidden = self.gru(x)
            return self.head(hidden[-1])

    return _GRUForecaster()


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

    def as_dict(self) -> dict[str, float | int]:
        return {"mae": self.mae, "mse": self.mse, "rmse": self.rmse, "count": self.count}


def _tensor_dataset(samples: list[ForecastSample], torch, TensorDataset):
    x, y = _tensors(samples, torch)
    return TensorDataset(x, y)


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
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("GRU forecasting requires PyTorch (`pip install torch`).") from exc
    return torch


def _torch_modules():
    torch = _torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, DataLoader, TensorDataset
