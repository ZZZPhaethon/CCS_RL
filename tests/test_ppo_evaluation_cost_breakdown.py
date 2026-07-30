from __future__ import annotations

from types import SimpleNamespace

import pytest

from sim.control.event_based.residual_rl_v2 import evaluation as residual
from sim.control.event_based.rl import evaluate_high_level_ppo as high_level
from sim.control.hourly_ppo import train_hourly_ppo as hourly


def _ledger() -> SimpleNamespace:
    return SimpleNamespace(
        vessel_fuel=11.0,
        conditioning=13.0,
        reconditioning=17.0,
        loading=19.0,
        unloading=23.0,
        operating_cost=83.0,
        vent_penalty=29.0,
        storage_shortfall_penalty=31.0,
        total_cost=143.0,
        vented_t=5.0,
    )


def _assert_cost_breakdown(row: dict[str, object]) -> None:
    assert row["episode_vessel_fuel_eur"] == pytest.approx(11.0)
    assert row["episode_conditioning_eur"] == pytest.approx(13.0)
    assert row["episode_reconditioning_eur"] == pytest.approx(17.0)
    assert row["episode_loading_eur"] == pytest.approx(19.0)
    assert row["episode_unloading_eur"] == pytest.approx(23.0)
    assert row["episode_operating_cost_eur"] == pytest.approx(83.0)
    assert row["episode_vent_penalty_eur"] == pytest.approx(29.0)
    assert row["episode_storage_shortfall_penalty_eur"] == pytest.approx(31.0)
    assert row["episode_total_cost_eur"] == pytest.approx(143.0)
    assert row["terminal_cleanup_operating_cost_eur"] == pytest.approx(37.0)
    assert row["operating_cost_eur"] == pytest.approx(120.0)
    assert row["total_cost_eur"] == pytest.approx(180.0)


class _Model:
    def predict(self, *_args, **_kwargs):
        return 0, None


class _WrappedEnv:
    def __init__(self, *, residual_info: bool = False) -> None:
        self.env = SimpleNamespace(
            ledger=_ledger(),
            cumulative_stored_t=7.0,
            cumulative_captured_t=10.0,
        )
        self._residual_info = residual_info

    def reset(self, *, seed: int):
        return object()

    def action_masks(self):
        return [True]

    def step(self, _action: int):
        info = {
            "elapsed_hours": 24.0,
            "action_label": "noop",
            "decision_trigger": "terminal",
            "violation_counts": {},
            "terminal_cleanup_operating_cost_eur": 37.0,
        }
        if self._residual_info:
            info.update(
                {
                    "intervention_selected": False,
                    "intervention_feasible_at_decision": False,
                    "native_action_changed": False,
                    "changed_native_steps": 0,
                    "avoided_vent_t": 0.0,
                    "incremental_stored_t": 0.0,
                    "total_cost_saving_eur": 0.0,
                }
            )
        return object(), -1.0, True, False, info


def test_hourly_evaluation_exports_episode_cost_breakdown(monkeypatch) -> None:
    env = SimpleNamespace(
        t=0,
        n_steps=0,
        reset=lambda **_kwargs: None,
        ledger=_ledger(),
        cost_model=SimpleNamespace(parameters=object()),
        simulator=SimpleNamespace(state=SimpleNamespace(time_h=720.0)),
        cumulative_stored_t=7.0,
        cumulative_captured_t=10.0,
    )
    monkeypatch.setattr(
        hourly,
        "_terminal_cleanup_cost_for_state",
        lambda *_args: 37.0,
    )

    row = hourly.evaluate_seed(None, env, seed=9)

    _assert_cost_breakdown(row)


def test_high_level_evaluation_exports_episode_cost_breakdown() -> None:
    row = high_level._evaluate_seed(_Model(), _WrappedEnv(), seed=9)

    _assert_cost_breakdown(row)


def test_residual_evaluation_exports_episode_cost_breakdown() -> None:
    row = residual.evaluate_seed(
        _Model(),
        _WrappedEnv(residual_info=True),
        seed=9,
    )

    _assert_cost_breakdown(row)
