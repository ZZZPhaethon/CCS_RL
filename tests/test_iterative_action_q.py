import numpy as np
import torch

from sim.control.iterative_action_q import (
    IterativeActionQuantileQ,
    IterativeForecastActionQuantileQ,
    IterativeFutureActionQuantileQ,
    quantile_huber_loss,
)


def _features():
    names = ["global.fill"]
    for vessel in ("a", "b", "c"):
        names.extend(
            [
                f"{vessel}.cargo",
                f"{vessel}.mode_loading",
                f"greedy_proposal.{vessel}.native_action_0",
            ]
        )
    return names


def _model():
    names = _features()
    joint_actions = np.asarray(
        [
            (left, right, third)
            for left in range(3)
            for right in range(3)
            for third in range(2)
        ]
    )
    return IterativeActionQuantileQ(
        names,
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        return_scale=4.0,
        heads=3,
        quantiles=7,
    )


def test_iterative_action_q_shape_and_frozen_prior():
    model = _model()
    q = model(torch.randn(2, 4, len(_features())))
    assert q.shape == (2, 4, 3, 18, 7)
    prior_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("structured_prior_")
    ]
    assert prior_parameters
    assert all(not parameter.requires_grad for parameter in prior_parameters)


def test_iterative_action_q_backpropagates_into_state_encoder():
    model = _model()
    loss = model(torch.randn(2, 1, len(_features()))).mean()
    loss.backward()
    assert model.state_encoder.vessel_encoder[0].weight.grad is not None


def test_iterative_future_q_uses_v4_summary_shape():
    names = _features()
    joint_actions = np.asarray([[0, 0, 0], [1, 0, 0]])
    model = IterativeFutureActionQuantileQ(
        names,
        [f"future_{index}" for index in range(14)],
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        future_mean=np.zeros(14),
        future_std=np.ones(14),
        return_scale=1.0,
        heads=2,
        quantiles=3,
    )
    q = model(
        torch.randn(2, 1, len(names)),
        torch.randn(2, 1, 14),
    )
    assert q.shape == (2, 1, 2, 2, 3)


def test_iterative_forecast_q_encoders_use_masked_168h_shape():
    names = _features()
    joint_actions = np.asarray([[0, 0, 0], [1, 0, 0]])
    forecast_names = [*[f"forecast_{index}" for index in range(9)], "valid_horizon"]
    for encoder in ("small_mlp", "tcn", "gru"):
        model = IterativeForecastActionQuantileQ(
            names,
            forecast_names,
            joint_actions,
            state_mean=np.zeros(len(names)),
            state_std=np.ones(len(names)),
            forecast_mean=np.zeros(9),
            forecast_std=np.ones(9),
            return_scale=1.0,
            forecast_encoder=encoder,
            heads=2,
            quantiles=3,
        )
        forecast = torch.randn(2, 1, 168, 10)
        forecast[..., -1] = 1.0
        forecast[1, :, 80:, :-1] = 0.0
        forecast[1, :, 80:, -1] = 0.0
        q = model(torch.randn(2, 1, len(names)), forecast)
        assert q.shape == (2, 1, 2, 2, 3)


def test_quantile_huber_loss_is_zero_for_identical_point_targets():
    predicted = torch.zeros(4, 5, 17)
    targets = torch.zeros(4, 5, 1)
    loss = quantile_huber_loss(predicted, targets)
    assert loss.shape == (4, 5)
    assert torch.all(loss == 0)


def test_quantile_huber_loss_backpropagates():
    predicted = torch.randn(3, 11, requires_grad=True)
    targets = torch.randn(3, 11)
    quantile_huber_loss(predicted, targets).mean().backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()
