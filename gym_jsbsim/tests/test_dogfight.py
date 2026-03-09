import math
import unittest

from gym_jsbsim import properties as prp
from gym_jsbsim.dogfight import compute_relative_geometry, telemetry_fieldnames


class FakeSim(dict):
    def __getitem__(self, prop):
        return dict.__getitem__(self, prop.name)

    def __setitem__(self, prop, value):
        dict.__setitem__(self, prop.name, value)


def make_sim(lat_deg, lon_deg, altitude_ft, heading_deg, pitch_deg=0.0):
    sim = FakeSim()
    sim[prp.lat_geod_deg] = lat_deg
    sim[prp.lng_geoc_deg] = lon_deg
    sim[prp.altitude_sl_ft] = altitude_ft
    sim[prp.heading_deg] = heading_deg
    sim[prp.pitch_rad] = math.radians(pitch_deg)
    return sim


class TestDogfightGeometry(unittest.TestCase):
    def test_target_directly_ahead_has_small_bearing_error(self):
        own = make_sim(51.3781, -2.3273, 5000, 0.0)
        target = make_sim(51.3790, -2.3273, 5000, 0.0)
        rel = compute_relative_geometry(own, target)
        self.assertGreater(rel.forward_m, 0.0)
        self.assertAlmostEqual(rel.bearing_error_deg, 0.0, places=3)

    def test_target_to_right_has_positive_bearing_error(self):
        own = make_sim(51.3781, -2.3273, 5000, 0.0)
        target = make_sim(51.3790, -2.3268, 5000, 0.0)
        rel = compute_relative_geometry(own, target)
        self.assertGreater(rel.bearing_error_deg, 0.0)

    def test_target_above_has_positive_elevation_error(self):
        own = make_sim(51.3781, -2.3273, 5000, 0.0)
        target = make_sim(51.3790, -2.3273, 5600, 0.0)
        rel = compute_relative_geometry(own, target)
        self.assertGreater(rel.elevation_error_deg, 0.0)


class TestTelemetryFieldnames(unittest.TestCase):
    def test_fieldnames_preserve_base_columns(self):
        rows = [
            {"plane_id": "plane_a", "episode": 0, "step": 0, "reward": 1.0, "done": False, "range_m": 100.0},
            {"plane_id": "plane_b", "episode": 0, "step": 0, "reward": 0.5, "done": False, "bearing_error_deg": 2.0},
        ]
        names = telemetry_fieldnames(rows)
        self.assertEqual(names[:5], ["plane_id", "episode", "step", "reward", "done"])
        self.assertIn("range_m", names)
        self.assertIn("bearing_error_deg", names)
