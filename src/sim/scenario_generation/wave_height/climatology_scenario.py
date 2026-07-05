from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from ...entities.storage import InjectionWell
from ...entities.vessel import Vessel
from ..generator import Scenario, ScenarioConfig, ScenarioGenerator


class LegWaveClimatology:
    """Seasonal mean speed factors keyed by sailing leg and hour of year."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        speed_factor_column: str = "speed_factor_p75",
        period_hours: int = 8784,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.speed_factor_column = speed_factor_column
        self.period_hours = int(period_hours)
        self._series_by_leg = self._read_climatology()

    @property
    def leg_ids(self) -> list[str]:
        return sorted(self._series_by_leg)

    def series(self, leg_id: str, *, start_hour_of_year: int, hours: int) -> list[float]:
        try:
            climatology = self._series_by_leg[leg_id]
        except KeyError as exc:
            raise ValueError(f"No climatology found for leg {leg_id!r}") from exc
        return [
            climatology[(start_hour_of_year + offset) % self.period_hours]
            for offset in range(hours)
        ]

    def _read_climatology(self) -> dict[str, list[float]]:
        buckets: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        with self.csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                leg_id = row.get("leg_id")
                if not leg_id:
                    raise ValueError("Leg-level climatology CSV must contain a 'leg_id' column.")
                hour = int(row["source_record"]) % self.period_hours
                buckets[leg_id][hour].append(min(1.0, max(0.0, float(row[self.speed_factor_column]))))

        series_by_leg: dict[str, list[float]] = {}
        for leg_id, by_hour in buckets.items():
            known_mean = {
                hour: sum(values) / len(values)
                for hour, values in by_hour.items()
                if values
            }
            if not known_mean:
                continue
            fallback = sum(known_mean.values()) / len(known_mean)
            series_by_leg[leg_id] = [
                known_mean.get(hour, known_mean.get(hour % 8760, fallback))
                for hour in range(self.period_hours)
            ]
        if not series_by_leg:
            raise ValueError(f"No usable leg climatology rows found in {self.csv_path}")
        return series_by_leg


class LegWaveClimatologyScenarioGenerator(ScenarioGenerator):
    """Scenario generator using seasonal mean speed factors for each sailing leg."""

    def __init__(
        self,
        leg_wave_csv: str | Path,
        *,
        speed_factor_column: str = "speed_factor_p75",
        period_hours: int = 8784,
        fixed_start_hour_of_year: int | None = None,
        keep_base_vessel_weather: bool = False,
        keep_base_injectivity: bool = False,
        config: ScenarioConfig | None = None,
        seed: int | None = None,
        climatology: LegWaveClimatology | None = None,
    ) -> None:
        super().__init__(config=config, seed=seed)
        self.climatology = climatology or LegWaveClimatology(
            leg_wave_csv,
            speed_factor_column=speed_factor_column,
            period_hours=period_hours,
        )
        self.fixed_start_hour_of_year = fixed_start_hour_of_year
        self.keep_base_vessel_weather = keep_base_vessel_weather
        self.keep_base_injectivity = keep_base_injectivity
        self.last_start_hour_of_year: int | None = None

    def sample(self, network, seed: int | None = None) -> Scenario:
        scenario = super().sample(network, seed=seed)
        if not self.keep_base_vessel_weather:
            scenario.vessel_speed_factor = {
                vessel_id: [1.0] * scenario.n_steps
                for vessel_id in network._entities_of_type(Vessel)
            }
        if not self.keep_base_injectivity:
            scenario.injectivity_factor = {
                well_id: [1.0] * scenario.n_steps
                for well_id in network._entities_of_type(InjectionWell)
            }
        start_hour = self._sample_start_hour(seed)
        self.last_start_hour_of_year = start_hour
        scenario.leg_speed_factor = {
            leg_id: self.climatology.series(
                leg_id,
                start_hour_of_year=start_hour,
                hours=scenario.n_steps,
            )
            for leg_id in self.climatology.leg_ids
        }
        return scenario

    def _sample_start_hour(self, seed: int | None) -> int:
        if self.fixed_start_hour_of_year is not None:
            return self.fixed_start_hour_of_year % self.climatology.period_hours
        episode_seed = seed if seed is not None else self.seed
        rng = random.Random(f"leg-wave-climatology:{episode_seed}") if episode_seed is not None else random.Random()
        return rng.randrange(self.climatology.period_hours)
