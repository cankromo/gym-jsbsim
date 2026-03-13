import math
import unittest
import types

from gym_jsbsim import properties as prp
from gym_jsbsim.dogfight import DogfightEnv, compute_relative_geometry, telemetry_fieldnames


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


class TestDogfightRewardHelpers(unittest.TestCase):
    def test_shot_quality_detects_fire_solution_ahead(self):
        env = DogfightEnv.__new__(DogfightEnv)
        rel = compute_relative_geometry(
            make_sim(51.3781, -2.3273, 5000, 0.0),
            make_sim(51.3790, -2.3273, 5000, 0.0),
        )
        shot = env._shot_quality(rel)
        self.assertTrue(shot["in_firing_cone"])
        self.assertTrue(shot["fire_solution"])

    def test_reverse_geometry_can_represent_defensive_threat(self):
        env = DogfightEnv.__new__(DogfightEnv)
        own = make_sim(51.3781, -2.3273, 5000, 0.0)
        target = make_sim(51.3790, -2.3273, 5000, 0.0)
        threat = env._shot_quality(compute_relative_geometry(target, own))
        self.assertFalse(threat["fire_solution"])

    def test_roll_quality_is_high_when_current_roll_matches_target(self):
        env = DogfightEnv.__new__(DogfightEnv)
        env.world = types.SimpleNamespace(
            sims={"plane_a": FakeSim()},
            tasks={"plane_a": types.SimpleNamespace(target_roll_rad=prp.Property("target/roll-rad", ""))},
        )
        env._sim = lambda agent: env.world.sims[agent]
        sim = env.world.sims["plane_a"]
        sim[prp.roll_rad] = math.radians(25.0)
        sim[env.world.tasks["plane_a"].target_roll_rad] = math.radians(25.0)

        info = env._roll_quality("plane_a")

        self.assertAlmostEqual(info["roll_quality"], 1.0)
        self.assertAlmostEqual(info["roll_error_deg"], 0.0)

    def test_roll_quality_is_zero_when_roll_error_exceeds_scale(self):
        env = DogfightEnv.__new__(DogfightEnv)
        env.world = types.SimpleNamespace(
            sims={"plane_a": FakeSim()},
            tasks={"plane_a": types.SimpleNamespace(target_roll_rad=prp.Property("target/roll-rad", ""))},
        )
        env._sim = lambda agent: env.world.sims[agent]
        sim = env.world.sims["plane_a"]
        sim[prp.roll_rad] = math.radians(45.0)
        sim[env.world.tasks["plane_a"].target_roll_rad] = math.radians(0.0)

        info = env._roll_quality("plane_a")

        self.assertAlmostEqual(info["roll_quality"], 0.0)
        self.assertAlmostEqual(info["roll_error_deg"], 45.0)
