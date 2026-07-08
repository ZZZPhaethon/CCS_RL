import unittest

from sim.ship_speed import (
    BJERKETVEDT_2020_SHIPS,
    NORTHERN_LIGHTS_SHIP,
    speed_factor,
    speed_factor_series,
    speed_knots,
)


class ShipSpeedTests(unittest.TestCase):
    def test_bjerketvedt_ship_returns_design_speed_in_calm_water(self):
        vessel = BJERKETVEDT_2020_SHIPS[5000]

        self.assertAlmostEqual(speed_knots(0.0, vessel), 12.0, places=6)

    def test_bjerketvedt_ship_slows_as_significant_wave_height_increases(self):
        vessel = BJERKETVEDT_2020_SHIPS[5000]

        calm = speed_knots(0.0, vessel)
        moderate = speed_knots(4.0, vessel)
        storm = speed_knots(8.0, vessel)

        self.assertGreater(calm, moderate)
        self.assertGreater(moderate, storm)
        self.assertAlmostEqual(moderate, 9.97, delta=0.05)
        self.assertAlmostEqual(storm, 6.62, delta=0.05)

    def test_northern_lights_ship_uses_reported_design_parameters(self):
        self.assertEqual(NORTHERN_LIGHTS_SHIP.power_kw, 5500.0)
        self.assertEqual(NORTHERN_LIGHTS_SHIP.beam_m, 21.2)
        self.assertEqual(NORTHERN_LIGHTS_SHIP.waterline_length_m, 130.0)
        self.assertEqual(NORTHERN_LIGHTS_SHIP.design_speed_knots, 14.0)
        self.assertAlmostEqual(speed_knots(0.0, NORTHERN_LIGHTS_SHIP), 14.0, places=6)

    def test_speed_factor_matches_simulator_weather_multiplier_interface(self):
        vessel = BJERKETVEDT_2020_SHIPS[5000]

        self.assertAlmostEqual(speed_factor(0.0, vessel), 1.0, places=6)
        self.assertAlmostEqual(speed_factor(4.0, vessel), speed_knots(4.0, vessel) / 12.0)
        self.assertLess(speed_factor(8.0, vessel), speed_factor(4.0, vessel))

    def test_speed_factor_series_converts_hourly_wave_heights(self):
        vessel = BJERKETVEDT_2020_SHIPS[5000]

        series = speed_factor_series([0.0, 4.0, 8.0], vessel)

        self.assertEqual(len(series), 3)
        self.assertAlmostEqual(series[0], 1.0, places=6)
        self.assertGreater(series[1], series[2])


if __name__ == "__main__":
    unittest.main()
