import csv
import tempfile
import unittest
from pathlib import Path

from sim.scenario_generation.wave_height.prediction import (
    SeasonalClimatology,
    WaveRouteDataset,
    evaluate_persistence,
    evaluate_seasonal_climatology,
)


FIELDNAMES = [
    "global_record",
    "source_file",
    "source_record",
    "vessel_id",
    "origin",
    "destination",
    "route_provider",
    "distance_km",
    "speed_knots",
    "hs_mean_m",
    "hs_p75_m",
    "hs_p90_m",
    "hs_max_m",
    "speed_factor_mean",
    "speed_factor_p75",
    "speed_factor_p90",
    "speed_factor_max",
]


def _write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        global_record = 0
        for year in (2010, 2011):
            for hour in range(8):
                for vessel_id, offset in (("ship_a", 0.0), ("ship_b", 1.0)):
                    hs = offset + hour * 0.1 + (year - 2010)
                    writer.writerow(
                        {
                            "global_record": global_record,
                            "source_file": f"wam10ei_{year}.nc",
                            "source_record": hour,
                            "vessel_id": vessel_id,
                            "origin": "source",
                            "destination": "terminal",
                            "route_provider": "test",
                            "distance_km": 100.0,
                            "speed_knots": 12.0,
                            "hs_mean_m": hs,
                            "hs_p75_m": hs + 0.1,
                            "hs_p90_m": hs + 0.2,
                            "hs_max_m": hs + 0.3,
                            "speed_factor_mean": 0.9,
                            "speed_factor_p75": 0.8,
                            "speed_factor_p90": 0.7,
                            "speed_factor_max": 0.6,
                        }
                    )
                global_record += 1


class WaveHeightPredictionTests(unittest.TestCase):
    def test_dataset_loads_groups_and_splits_by_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.csv"
            _write_csv(path)

            dataset = WaveRouteDataset.from_csv(path)
            splits = dataset.split_by_years(train_years=(2010,), test_years=(2011,))

        self.assertEqual(len(dataset.by_vessel()), 2)
        self.assertEqual(len(splits["train"].records), 16)
        self.assertEqual(len(splits["test"].records), 16)

    def test_make_samples_builds_contiguous_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.csv"
            _write_csv(path)
            dataset = WaveRouteDataset.from_csv(path)

            samples = dataset.make_samples(lookback_hours=3, horizon_hours=2)

        self.assertTrue(samples)
        self.assertEqual(len(samples[0].history), 3)
        self.assertEqual(len(samples[0].target), 2)
        self.assertGreater(len(samples[0].history[0]), 5)  # wave features + time features

    def test_persistence_and_seasonal_baselines_return_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wave.csv"
            _write_csv(path)
            dataset = WaveRouteDataset.from_csv(path)
            splits = dataset.split_by_years(train_years=(2010,), test_years=(2011,))
            samples = splits["test"].make_samples(lookback_hours=3, horizon_hours=2)

            persistence = evaluate_persistence(samples)
            seasonal = evaluate_seasonal_climatology(splits["train"], samples)
            model = SeasonalClimatology().fit(splits["train"])
            prediction = model.predict_sample(samples[0])

        self.assertGreater(persistence.count, 0)
        self.assertGreater(seasonal.count, 0)
        self.assertEqual(len(prediction), 2)


if __name__ == "__main__":
    unittest.main()
