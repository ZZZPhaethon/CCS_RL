import numpy as np

from scripts import compare_forecast_encoders_rl as compare
from sim.control.baselines import greedy_shuttle_policy
from sim.environment.event_residual_gym import EventJointResidualGymEnv


def _native_env(hours=72):
    args = compare.parse_args(
        [
            "train",
            "--variant",
            "future_mlp_mode",
            "--demo-cache",
            "unused.npz",
            "--timesteps",
            "0",
            "--bc-only",
            "--episode-hours",
            str(hours),
            "--device",
            "cpu",
        ]
    )
    return compare.make_experiment_env(args, demonstration=False)


def test_joint_action_encoding_round_trip_and_duplicate_mask():
    wrapper = EventJointResidualGymEnv(_native_env(48))
    wrapper.reset(seed=11)
    masks = wrapper.action_masks()
    assert masks.shape == (wrapper.action_space.n,)
    assert masks[wrapper.follow_action()]
    for index in np.flatnonzero(masks):
        assert wrapper.encode_action(wrapper.decode_action(index)) == index

    base = wrapper.residual_env._base_action["vessels"]
    follow = wrapper.residual_env.follow_indices
    for vessel_index, base_action in enumerate(base):
        residual = follow.copy()
        residual[vessel_index] = int(base_action)
        assert not masks[wrapper.encode_action(residual)]


def test_all_follow_matches_native_greedy_economics():
    seed = 23
    hours = 96
    native = _native_env(hours)
    native.reset(seed=seed)
    while native.t < native.n_steps:
        native.step(greedy_shuttle_policy(native))

    event_native = _native_env(hours)
    wrapper = EventJointResidualGymEnv(event_native)
    observation, _info = wrapper.reset_native_seed(seed)

    done = False
    total_reward = 0.0
    while not done:
        assert wrapper.observation_space.contains(observation)
        observation, reward, terminated, truncated, _info = wrapper.step(
            wrapper.follow_action()
        )
        total_reward += reward
        done = terminated or truncated

    assert event_native.t == native.t == hours
    assert event_native.ledger.total_cost == native.ledger.total_cost
    assert event_native.ledger.vented_t == native.ledger.vented_t
    assert event_native.ledger.stored_t == native.ledger.stored_t
    # Greedy control variate makes all-FOLLOW identically zero reward.
    assert abs(total_reward) < 1e-9
