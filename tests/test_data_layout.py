from pathlib import Path
import unittest

import sim.network_scenarios as scenarios


ROOT = Path(__file__).resolve().parents[1]


class DataLayoutTests(unittest.TestCase):
    def test_scenario_files_live_in_root_scenarios_folder(self):
        scenario_dir = ROOT / "scenarios"

        self.assertEqual(scenarios.NORTHERN_LIGHTS_PHASE1_DATA_PATH, scenario_dir / "northern_lights_phase1.json")
        self.assertEqual(scenarios.NORTHERN_LIGHTS_PHASE1_2WELL_DATA_PATH, scenario_dir / "northern_lights_phase1_2well.json")
        self.assertEqual(scenarios.NORTHERN_LIGHTS_PHASE2_DATA_PATH, scenario_dir / "northern_lights_phase2.json")
        self.assertEqual(getattr(scenarios, "TOY_DATA_PATH", None), scenario_dir / "toy.json")
        self.assertTrue(scenarios.NORTHERN_LIGHTS_PHASE1_DATA_PATH.exists())
        self.assertTrue(scenarios.NORTHERN_LIGHTS_PHASE1_2WELL_DATA_PATH.exists())
        self.assertTrue(scenarios.NORTHERN_LIGHTS_PHASE2_DATA_PATH.exists())
        self.assertTrue(scenarios.TOY_DATA_PATH.exists())
        self.assertFalse((ROOT / "data" / "northern_lights_phase1_demo.json").exists())
        self.assertFalse((ROOT / "data" / "northern_lights_phase1_plus_yara_2026.json").exists())

    def test_capture_rate_files_are_grouped_under_data(self):
        capture_rate_dir = ROOT / "data" / "capture_rates"

        self.assertEqual(
            getattr(scenarios, "NORTHERN_LIGHTS_PHASE1_CAPTURE_PROFILE_PATH", None),
            capture_rate_dir / "phase1plus_emitters_capture_rate_profile_hourly.csv",
        )
        self.assertTrue(capture_rate_dir.is_dir())
        self.assertTrue((capture_rate_dir / "phase1plus_emitters_capture_rate_profile_daily.csv").exists())
        self.assertTrue((capture_rate_dir / "phase1plus_emitters_capture_rate_profile_hourly.csv").exists())
        self.assertTrue((capture_rate_dir / "phase1plus_emitters_capture_rate_profile_metadata.json").exists())
        self.assertTrue((capture_rate_dir / "phase1plus_emitters_monthly_capture_profile.csv").exists())


if __name__ == "__main__":
    unittest.main()
