import numpy as np
import torch

from sim.control.recurrent_distributional_q import (
    RecurrentBootstrappedQuantileQ,
    StatelessStructuredActionQuantileQ,
    StructuredActionRecurrentQuantileQ,
    quantile_huber_loss,
)


def _features():
    names = ["global.fill"]
    for vessel in ("a", "b", "c"):
        names.extend(
            [
                f"{vessel}.cargo",
                f"{vessel}.mode_loading",
                f"{vessel}.sailing_to_terminal",
                f"greedy_proposal.{vessel}.native_action_0",
            ]
        )
    return names


def test_recurrent_quantile_network_shape_and_hidden_state():
    names = _features()
    model = RecurrentBootstrappedQuantileQ(
        names,
        (168, 9),
        216,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(9),
        forecast_std=np.ones(9),
        return_scale=4.0,
        heads=5,
        quantiles=17,
    )
    q, hidden = model(
        torch.randn(2, 3, len(names)),
        torch.randn(2, 3, 168, 9),
        torch.tensor([[-1, 4, 7], [-1, 2, 3]]),
        torch.zeros(2, 3),
        torch.ones(2, 3),
    )
    assert q.shape == (2, 3, 5, 216, 17)
    assert hidden.shape == (1, 2, 128)
    assert all(not parameter.requires_grad for parameter in model.prior.parameters())
    recurrent, _hidden = model.recurrent_features(
        torch.randn(2, 3, len(names)),
        torch.randn(2, 3, 168, 9),
        torch.tensor([[-1, 4, 7], [-1, 2, 3]]),
        torch.zeros(2, 3),
        torch.ones(2, 3),
    )
    assert model.quantiles_from_features(recurrent).shape == q.shape


def test_quantile_huber_loss_is_zero_for_identical_point_targets():
    predicted = torch.zeros(4, 5, 17)
    targets = torch.zeros(4, 5, 1)
    loss = quantile_huber_loss(predicted, targets)
    assert loss.shape == (4, 5)
    assert torch.all(loss == 0)


def test_recurrent_quantile_network_supports_small_mlp_forecast_encoder():
    names = _features()
    model = RecurrentBootstrappedQuantileQ(
        names,
        (168, 9),
        8,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(9),
        forecast_std=np.ones(9),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="small_mlp",
    )
    q, _hidden = model(
        torch.randn(2, 1, len(names)),
        torch.randn(2, 1, 168, 9),
        torch.full((2, 1), -1),
        torch.zeros(2, 1),
        torch.ones(2, 1),
    )
    assert q.shape == (2, 1, 2, 8, 3)


def test_quantile_huber_loss_backpropagates():
    predicted = torch.randn(3, 11, requires_grad=True)
    targets = torch.randn(3, 11)
    loss = quantile_huber_loss(predicted, targets).mean()
    loss.backward()
    assert predicted.grad is not None
    assert torch.isfinite(predicted.grad).all()


def test_structured_action_network_shares_action_components_and_has_frozen_prior():
    names = _features()
    joint_actions = np.asarray(
        [(left, right, third) for left in range(3) for right in range(3) for third in range(2)]
    )
    model = StructuredActionRecurrentQuantileQ(
        names,
        (168, 9),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(9),
        forecast_std=np.ones(9),
        return_scale=4.0,
        heads=3,
        quantiles=7,
    )
    q, hidden = model(
        torch.randn(2, 2, len(names)),
        torch.randn(2, 2, 168, 9),
        torch.tensor([[-1, 4], [-1, 2]]),
        torch.zeros(2, 2),
        torch.ones(2, 2),
    )
    assert q.shape == (2, 2, 3, len(joint_actions), 7)
    assert hidden.shape == (1, 2, 128)
    assert len(model.structured_action_embeddings) == 3
    prior_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("structured_prior")
    ]
    assert prior_parameters
    assert all(not parameter.requires_grad for parameter in prior_parameters)


def test_stateless_conversion_matches_reset_recurrent_structured_q():
    names = _features()
    joint_actions = np.asarray(
        [(left, right, third) for left in range(3) for right in range(3) for third in range(2)]
    )
    recurrent = StructuredActionRecurrentQuantileQ(
        names,
        (168, 9),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(9),
        forecast_std=np.ones(9),
        return_scale=4.0,
        heads=3,
        quantiles=7,
    ).eval()
    stateless = StatelessStructuredActionQuantileQ.from_reset_recurrent(
        recurrent, names, (168, 9), joint_actions
    ).eval()
    states = torch.randn(5, 1, len(names))
    forecasts = recurrent.forecast_mean.reshape(1, 1, 1, -1).expand(
        5, 1, 168, 9
    )
    with torch.no_grad():
        expected, _hidden = recurrent(
            states,
            forecasts,
            torch.full((5, 1), -1),
            torch.zeros(5, 1),
            torch.zeros(5, 1),
            recurrent.initial_hidden(5),
        )
        actual = stateless(states)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert not hasattr(stateless, "action_embedding")
    assert not hasattr(stateless, "forecast_encoder")
    assert not hasattr(stateless, "gru")
    assert not hasattr(stateless, "initial_hidden")
    assert not hasattr(stateless, "recurrent_features")
    assert "weight_hh" not in dict(stateless.named_parameters())


def test_action_aligned_forecast_residual_starts_from_state_only_q_values():
    names = _features()
    joint_actions = np.asarray(
        [
            (left, right, third)
            for left in range(6)
            for right in range(6)
            for third in range(6)
        ]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StructuredActionRecurrentQuantileQ(
        names,
        (168, 9),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(9),
        forecast_std=np.ones(9),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="action_aligned",
        forecast_channel_names=forecast_names,
    )
    states = torch.randn(1, 1, len(names))
    forecasts = torch.randn(1, 1, 168, 9)
    previous_actions = torch.full((1, 1), -1)
    previous_rewards = torch.zeros(1, 1)
    previous_durations = torch.ones(1, 1)
    q, _hidden = model(
        states,
        forecasts,
        previous_actions,
        previous_rewards,
        previous_durations,
    )
    base_q, _hidden = RecurrentBootstrappedQuantileQ.forward(
        model,
        states,
        torch.zeros_like(forecasts),
        previous_actions,
        previous_rewards,
        previous_durations,
    )
    assert torch.equal(q, base_q)


def _eta_features():
    emitters = ("a", "b", "c")
    vessels = ("ship_a", "ship_b", "ship_c")
    names = [
        "a.fill",
        "b.fill",
        "c.fill",
        "oygarden_terminal.fill",
        "weather.speed_now",
    ]
    for vessel in vessels:
        for destination in ("oygarden_terminal", *emitters):
            names.append(f"{vessel}.to_{destination}.travel_hours_now")
    for vessel in vessels:
        names.extend(
            f"greedy_proposal.{vessel}.native_action_{action}"
            for action in range(5)
        )
    return names


def test_eta_aligned_forecast_uses_destination_arrival_time():
    names = _eta_features()
    joint_actions = np.asarray(
        [
            (left, right, third)
            for left in range(6)
            for right in range(6)
            for third in range(6)
        ]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StructuredActionRecurrentQuantileQ(
        names,
        (168, 9),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(9),
        forecast_std=np.ones(9),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="eta_aligned",
        forecast_channel_names=forecast_names,
        episode_hours=720,
    )
    state = torch.zeros(2, len(names))
    state[:, names.index("weather.speed_now")] = 1.0
    state[:, names.index("b.fill")] = 0.25
    travel_index = names.index("ship_a.to_b.travel_hours_now")
    state[0, travel_index] = 24.0 / 720.0
    state[1, travel_index] = 48.0 / 720.0
    forecast = torch.ones(2, 168, 9)
    summaries = model.eta_aligned_residual._local_action_summaries(
        state, forecast
    )
    destination_b_action = 3
    short = summaries[0, 0, destination_b_action]
    long = summaries[1, 0, destination_b_action]
    assert long[0] > short[0]
    assert long[5] > short[5]


def test_eta_aligned_forecast_residual_starts_from_state_only_q_values():
    names = _eta_features()
    joint_actions = np.asarray(
        [
            (left, right, third)
            for left in range(6)
            for right in range(6)
            for third in range(6)
        ]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StructuredActionRecurrentQuantileQ(
        names,
        (168, 9),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(9),
        forecast_std=np.ones(9),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="eta_aligned",
        forecast_channel_names=forecast_names,
    )
    states = torch.zeros(1, 1, len(names))
    states[..., names.index("weather.speed_now")] = 1.0
    forecasts = torch.ones(1, 1, 168, 9)
    previous_actions = torch.full((1, 1), -1)
    previous_rewards = torch.zeros(1, 1)
    previous_durations = torch.ones(1, 1)
    q, _hidden = model(
        states,
        forecasts,
        previous_actions,
        previous_rewards,
        previous_durations,
    )
    base_q, _hidden = RecurrentBootstrappedQuantileQ.forward(
        model,
        states,
        torch.zeros_like(forecasts),
        previous_actions,
        previous_rewards,
        previous_durations,
    )
    assert torch.equal(q, base_q)


def test_stateless_eta_model_uses_future_without_history_inputs():
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="eta_aligned",
        forecast_channel_names=forecast_names,
    )
    states = torch.zeros(2, 1, len(names))
    states[..., names.index("weather.speed_now")] = 1.0
    forecasts = torch.ones(2, 1, 168, len(forecast_names))
    q = model(states, forecasts)
    assert q.shape == (2, 1, 2, len(joint_actions), 3)
    assert model.eta_aligned_residual is not None
    assert not hasattr(model, "action_embedding")
    assert not hasattr(model, "gru")
    assert not hasattr(model, "initial_hidden")


def test_stateless_eta_joint_predicts_full_q_without_state_only_addition():
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="eta_joint",
        forecast_channel_names=forecast_names,
    )
    with torch.no_grad():
        model.value.bias.fill_(5.0)
    states = torch.zeros(1, 1, len(names))
    states[..., names.index("weather.speed_now")] = 1.0
    forecasts = torch.ones(1, 1, 168, len(forecast_names))
    q = model(states, forecasts)
    assert q.shape == (1, 1, 2, len(joint_actions), 3)
    assert torch.count_nonzero(q) == 0
    assert model.eta_joint_q is not None
    assert model.eta_aligned_residual is None


def test_window_summary_forecast_uses_requested_horizons():
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="window_summary_24_72",
        forecast_channel_names=forecast_names,
    )
    forecast = torch.ones(1, 168, len(forecast_names))
    forecast[:, 24:, 3] = 0.0
    forecast[:, 12:24, 6] = 0.0
    forecast[:, 10, 7] = 0.25
    forecast[:, 20, 8] = 0.5
    summary = model.window_summary_residual._summaries(forecast)
    assert model.window_summary_residual.horizons == (24, 72)
    assert summary.shape == (1, 14)
    assert summary[0, 0] == 1.0
    assert torch.isclose(summary[0, 7], torch.tensor(1.0 / 3.0))
    assert summary[0, 3] == 0.5
    assert summary[0, 4] == 0.25
    assert summary[0, 6] == 0.5


def test_window_summary_joint_predicts_full_q_without_state_only_addition():
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="window_summary_joint_168",
        forecast_channel_names=forecast_names,
    )
    with torch.no_grad():
        model.value.bias.fill_(5.0)
    states = torch.zeros(1, 1, len(names))
    forecasts = torch.ones(1, 1, 168, len(forecast_names))
    q = model(states, forecasts)
    assert q.shape == (1, 1, 2, len(joint_actions), 3)
    assert torch.count_nonzero(q) == 0
    assert model.window_summary_joint_q is not None
    assert model.window_summary_residual is None


def test_stateless_arrival_time_reads_destination_forecast_at_arrival():
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="arrival_time",
        forecast_channel_names=forecast_names,
    )
    state = torch.zeros(1, len(names))
    state[:, names.index("weather.speed_now")] = 1.0
    state[:, names.index("ship_a.to_b.travel_hours_now")] = 10.0 / 720.0
    state[:, names.index("greedy_proposal.ship_a.native_action_3")] = 1.0
    forecast = torch.zeros(1, 168, len(forecast_names))
    forecast[..., forecast_names.index("weather.global_speed_factor")] = 0.5
    forecast[..., forecast_names.index("emitter_available.b")] = 1.0
    forecast[:, 18:23, forecast_names.index("capture.b")] = 1.0
    summaries = model.arrival_time_residual._local_action_summaries(
        state, forecast
    )
    destination_b_action = 3
    destination_a_action = 2
    follow_action = 5
    assert torch.isclose(
        summaries[0, 0, destination_b_action, 0],
        torch.tensor(20.0 / 168.0),
    )
    assert summaries[0, 0, destination_b_action, 2] > 0.0
    assert summaries[0, 0, destination_a_action, 2] == 0.0
    assert torch.equal(
        summaries[0, 0, follow_action],
        summaries[0, 0, destination_b_action],
    )


def test_stateless_arrival_time_starts_from_state_only_q_values():
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.ones(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="arrival_time",
        forecast_channel_names=forecast_names,
    )
    states = torch.zeros(2, 1, len(names))
    states[..., names.index("weather.speed_now")] = 1.0
    forecasts = torch.rand(2, 1, 168, len(forecast_names))
    with torch.no_grad():
        q = model(states, forecasts)
        model.arrival_time_residual = None
        state_only_q = model(states, forecasts)
    assert q.shape == (2, 1, 2, len(joint_actions), 3)
    assert torch.equal(q, state_only_q)


def test_stateless_small_mlp_selects_forecast_for_each_destination():
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="small_mlp",
        forecast_channel_names=forecast_names,
    )
    state = torch.zeros(1, len(names))
    state[:, names.index("greedy_proposal.ship_a.native_action_3")] = 1.0
    forecast = torch.zeros(1, 168, len(forecast_names))
    forecast[..., forecast_names.index("capture.b")] = 2.0
    selected, local_actions = model.small_mlp_residual._selected_future(
        state, forecast
    )
    assert torch.all(selected[0, 0, 3, :, 0] == 2.0)
    assert torch.all(selected[0, 0, 2, :, 0] == 0.0)
    assert torch.all(selected[0, 0, 5, :, 0] == 2.0)
    assert torch.equal(
        local_actions[0, 0, 5],
        state[0, model.small_mlp_residual.proposal_indices[0]],
    )


def test_stateless_small_mlp_starts_from_state_only_q_values():
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="small_mlp",
        forecast_channel_names=forecast_names,
    )
    states = torch.zeros(2, 1, len(names))
    states[..., names.index("greedy_proposal.ship_a.native_action_2")] = 1.0
    states[..., names.index("greedy_proposal.ship_b.native_action_3")] = 1.0
    states[..., names.index("greedy_proposal.ship_c.native_action_4")] = 1.0
    forecasts = torch.randn(2, 1, 168, len(forecast_names))
    with torch.no_grad():
        q = model(states, forecasts)
        model.small_mlp_residual = None
        state_only_q = model(states, forecasts)
    assert q.shape == (2, 1, 2, len(joint_actions), 3)
    assert torch.equal(q, state_only_q)


def test_stateless_action_aligned_starts_from_state_only_q_values():
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.well",
        "injectivity.well",
        "weather.global_speed_factor",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="action_aligned",
        forecast_channel_names=forecast_names,
    )
    states = torch.zeros(2, 1, len(names))
    forecasts = torch.randn(2, 1, 168, len(forecast_names))
    with torch.no_grad():
        q = model(states, forecasts)
        model.action_aligned_residual = None
        state_only_q = model(states, forecasts)
    assert q.shape == (2, 1, 2, len(joint_actions), 3)
    assert torch.equal(q, state_only_q)


def test_stateless_temporal_attention_uses_raw_future_sequence():
    torch.manual_seed(3)
    names = _eta_features()
    joint_actions = np.asarray(
        [(action, action, action) for action in range(6)]
    )
    forecast_names = [
        "capture.a",
        "capture.b",
        "capture.c",
        "emitter_available.a",
        "emitter_available.b",
        "emitter_available.c",
        "well_available.main",
        "injectivity.main",
        "weather.main",
    ]
    model = StatelessStructuredActionQuantileQ(
        names,
        (168, len(forecast_names)),
        joint_actions,
        state_mean=np.zeros(len(names)),
        state_std=np.ones(len(names)),
        forecast_mean=np.zeros(len(forecast_names)),
        forecast_std=np.ones(len(forecast_names)),
        return_scale=4.0,
        heads=2,
        quantiles=3,
        forecast_encoder="temporal_attention",
        forecast_channel_names=forecast_names,
    )
    states = torch.zeros(1, 1, len(names))
    states[..., names.index("greedy_proposal.ship_a.native_action_3")] = 1.0
    states[..., names.index("greedy_proposal.ship_b.native_action_3")] = 1.0
    states[..., names.index("greedy_proposal.ship_c.native_action_3")] = 1.0
    forecasts = torch.zeros(1, 1, 168, len(forecast_names))
    with torch.no_grad():
        state_only_q = model(states, forecasts)
        torch.nn.init.normal_(
            model.temporal_attention_residual.residual_head[-1].weight
        )
        changed = forecasts.clone()
        changed[..., 72, forecast_names.index("capture.b")] = 4.0
        future_q = model(states, changed)
    assert state_only_q.shape == (1, 1, 2, len(joint_actions), 3)
    assert not torch.equal(future_q, state_only_q)
    assert model.temporal_attention_residual.temporal_attention is True
