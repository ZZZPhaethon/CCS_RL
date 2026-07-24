"""Convert rolling LSTM wave-height forecasts into vessel-speed scenarios.

将滚动 LSTM 波高预测转换为船舶航速场景。
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ...entities.vessel import Vessel
from ...ship_speed import NORTHERN_LIGHTS_SHIP, ShipSpeedParameters, speed_factor_series
from ..generator import Scenario, ScenarioConfig, ScenarioGenerator


@dataclass(frozen=True)
class ForecastWindow:
    """One rolling forecast window from the LSTM prediction CSV."""

    start_global_record: int
    wave_height_by_vessel: dict[str, list[float]]


class LSTMWaveHeightForecastReader:
    """Reads rolling LSTM predictions into route-level forecast windows.

    ``predict_lstm.py`` writes one row per vessel and forecast horizon hour. The
    first ``global_record`` for each vessel sample is the forecast start. This
    reader groups those rows back into windows that can drive one MPC episode.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        prediction_column: str = "predicted",
        vessel_ids: list[str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.prediction_column = prediction_column
        self._windows = self._read_windows(set(vessel_ids or ()))

    @property
    def start_records(self) -> list[int]:
        return sorted(self._windows)

    def window(self, start_global_record: int) -> ForecastWindow:
        try:
            return self._windows[int(start_global_record)]
        except KeyError as exc:
            raise ValueError(f"No LSTM forecast window starts at global_record={start_global_record}") from exc

    def _read_windows(self, required_vessels: set[str]) -> dict[int, ForecastWindow]:
        grouped: dict[tuple[str, int], list[tuple[int, float]]] = defaultdict(list)
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                vessel_id = row["vessel_id"]
                if required_vessels and vessel_id not in required_vessels:
                    continue
                horizon_index = int(row["horizon_index"])
                global_record = int(row["global_record"])
                start_record = global_record - horizon_index
                wave_height_m = max(0.0, float(row[self.prediction_column]))
                grouped[(vessel_id, start_record)].append((horizon_index, wave_height_m))

        by_start: dict[int, dict[str, list[float]]] = defaultdict(dict)
        for (vessel_id, start_record), values in grouped.items():
            values.sort(key=lambda item: item[0])
            if not values or values[0][0] != 0:
                continue
            expected = list(range(values[-1][0] + 1))
            actual = [index for index, _height in values]
            if actual != expected:
                continue
            by_start[start_record][vessel_id] = [height for _index, height in values]

        windows: dict[int, ForecastWindow] = {}
        for start_record, wave_height_by_vessel in by_start.items():
            if required_vessels and not required_vessels.issubset(wave_height_by_vessel):
                continue
            windows[start_record] = ForecastWindow(
                start_global_record=start_record,
                wave_height_by_vessel=dict(wave_height_by_vessel),
            )
        if not windows:
            raise ValueError(f"No usable LSTM forecast windows found in {self.path}")
        return windows


class LSTMWaveHeightScenarioGenerator(ScenarioGenerator):
    """Scenario generator driven by precomputed rolling LSTM wave forecasts."""

    def __init__(
        self,
        prediction_csv: str | Path,
        *,
        routes: Mapping[str, Mapping[str, object]],
        ship_parameters_by_vessel: Mapping[str, ShipSpeedParameters] | None = None,
        default_ship_parameters: ShipSpeedParameters = NORTHERN_LIGHTS_SHIP,
        prediction_column: str = "predicted",
        fixed_start_global_record: int | None = None,
        config: ScenarioConfig | None = None,
        seed: int | None = None,
        reader: LSTMWaveHeightForecastReader | None = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        self.routes = routes
        self.ship_parameters_by_vessel = dict(ship_parameters_by_vessel or {})
        self.default_ship_parameters = default_ship_parameters
        self.fixed_start_global_record = fixed_start_global_record
        self.reader = reader or LSTMWaveHeightForecastReader(
            prediction_csv,
            prediction_column=prediction_column,
            vessel_ids=list(routes),
        )
        self.last_start_global_record: int | None = None

    @classmethod
    def from_env(
        cls,
        env,
        prediction_csv: str | Path,
        *,
        ship_parameters_by_vessel: Mapping[str, ShipSpeedParameters] | None = None,
        default_ship_parameters: ShipSpeedParameters = NORTHERN_LIGHTS_SHIP,
        prediction_column: str = "predicted",
        fixed_start_global_record: int | None = None,
        config: ScenarioConfig | None = None,
        seed: int | None = None,
    ) -> "LSTMWaveHeightScenarioGenerator":
        return cls(
            prediction_csv,
            routes=env._routes,
            ship_parameters_by_vessel=ship_parameters_by_vessel,
            default_ship_parameters=default_ship_parameters,
            prediction_column=prediction_column,
            fixed_start_global_record=fixed_start_global_record,
            config=config,
            seed=seed,
        )

    def sample(self, network, seed: int | None = None) -> Scenario:
        scenario = super().sample(network, seed=seed)
        start_record = self._sample_start_record(seed)
        window = self.reader.window(start_record)
        self.last_start_global_record = start_record

        vessel_speed_factor: dict[str, list[float]] = {}
        for vessel_id in network._entities_of_type(Vessel):
            route = self.routes.get(vessel_id)
            wave_heights = window.wave_height_by_vessel.get(vessel_id)
            if route is None or wave_heights is None:
                continue
            if len(wave_heights) < scenario.n_steps:
                raise ValueError(
                    f"LSTM forecast for {vessel_id} at global_record={start_record} has "
                    f"{len(wave_heights)} hours, but scenario needs {scenario.n_steps}."
                )
            parameters = self.ship_parameters_by_vessel.get(vessel_id, self.default_ship_parameters)
            nominal_speed_knots = float(route.get("speed_knots") or parameters.design_speed_knots)
            factors = speed_factor_series(
                wave_heights[:scenario.n_steps],
                parameters,
                nominal_speed_knots=nominal_speed_knots,
            )
            vessel_speed_factor[vessel_id] = [min(1.0, max(0.0, factor)) for factor in factors]

        if vessel_speed_factor:
            scenario.vessel_speed_factor = vessel_speed_factor
        return scenario

    def _sample_start_record(self, seed: int | None) -> int:
        if self.fixed_start_global_record is not None:
            return int(self.fixed_start_global_record)
        starts = self.reader.start_records
        episode_seed = seed if seed is not None else self.seed
        rng = random.Random(f"lstm-wave-height:{episode_seed}") if episode_seed is not None else random.Random()
        return rng.choice(starts)
