"""RL environment package for CCS training and evaluation."""

from importlib import import_module

from .env import (
    CCSEnv,
    CCSEnvConfig,
    MAX_WELL_RATE_INDEX,
    MAX_WELL_RATE_MTPA,
    MIN_WELL_RATE_INDEX,
    MIN_WELL_RATE_MTPA,
    OFF_WELL_RATE_INDEX,
    VESSEL_ACTIONS,
    VESSEL_GO_EMITTER_BASE,
    VESSEL_GO_TERMINAL,
    VESSEL_WAIT,
    WELL_RATE_BOUNDS_MTPA,
    WELL_RATE_LEVELS_MTPA,
)
from .factories import build_phase1_env
from .forecast import (
    FORECAST_HORIZON_H,
    current_state_feature_names,
    current_state_observation,
    forecast_channel_names,
    future_forecast_observation,
    masked_future_forecast_observation,
)

_FORECAST_GYM_EXPORTS = frozenset(
    {
        "ForecastGymEnv",
        "forecast_policy_observation",
        "make_forecast_ppo_policy",
    }
)


def __getattr__(name: str):
    if name in _FORECAST_GYM_EXPORTS:
        module = import_module(".forecast_gym", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CCSEnv",
    "CCSEnvConfig",
    "FORECAST_HORIZON_H",
    "ForecastGymEnv",
    "MAX_WELL_RATE_INDEX",
    "MAX_WELL_RATE_MTPA",
    "MIN_WELL_RATE_INDEX",
    "MIN_WELL_RATE_MTPA",
    "OFF_WELL_RATE_INDEX",
    "VESSEL_ACTIONS",
    "VESSEL_GO_EMITTER_BASE",
    "VESSEL_GO_TERMINAL",
    "VESSEL_WAIT",
    "WELL_RATE_BOUNDS_MTPA",
    "WELL_RATE_LEVELS_MTPA",
    "build_phase1_env",
    "current_state_feature_names",
    "current_state_observation",
    "forecast_channel_names",
    "forecast_policy_observation",
    "future_forecast_observation",
    "masked_future_forecast_observation",
    "make_forecast_ppo_policy",
]
