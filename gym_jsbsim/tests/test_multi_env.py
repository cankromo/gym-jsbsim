import unittest
import sys
import types

import numpy as np

if 'jsbsim' not in sys.modules:
    sys.modules['jsbsim'] = types.SimpleNamespace(FGFDMExec=object)

import gym_jsbsim.properties as prp
from gym_jsbsim.multi_env import FormationMember, SharedWorldJsbSimEnv
from gym_jsbsim.tests.stubs import BasicFlightTask, SimStub
from gym_jsbsim import utils


class SharedWorldSimStub(SimStub):
    def reinitialise(self, init_conditions=None):
        if init_conditions:
            for prop, value in init_conditions.items():
                self[prop] = value

    def close(self):
        pass


class SharedWorldEnvStub(SharedWorldJsbSimEnv):
    def _init_new_sim(self, aircraft, initial_conditions):
        sim = SharedWorldSimStub()
        for prop, value in initial_conditions.items():
            sim[prop] = value
        for task in self.tasks.values():
            for prop in task.state_variables:
                sim[prop] = 0.0
        sim[prp.heading_deg] = 270.0
        sim[prp.altitude_sl_ft] = initial_conditions[prp.initial_altitude_ft]
        sim[prp.v_north_fps] = 0.0
        sim[prp.v_east_fps] = 0.0
        sim[prp.roll_rad] = 0.0
        sim[prp.pitch_rad] = 0.0
        sim[prp.lat_geod_deg] = initial_conditions[prp.initial_latitude_geod_deg]
        sim[prp.lng_geoc_deg] = initial_conditions[prp.initial_longitude_geoc_deg]
        return sim


class TestSharedWorldJsbSimEnv(unittest.TestCase):
    def make_env(self):
        return SharedWorldEnvStub([
            FormationMember(name='plane_a', task_type=BasicFlightTask),
            FormationMember(name='plane_b', task_type=BasicFlightTask, east_offset_m=120.0),
        ])

    def test_reset_returns_observations_for_both_planes(self):
        env = self.make_env()
        obs = env.reset()
        self.assertEqual(tuple(obs.keys()), ('plane_a', 'plane_b'))
        self.assertIsInstance(obs['plane_a'], np.ndarray)
        self.assertIsInstance(obs['plane_b'], np.ndarray)

    def test_offset_initial_conditions_are_applied(self):
        env = self.make_env()
        env.reset()
        lat_a = env.sims['plane_a'][prp.lat_geod_deg]
        lon_a = env.sims['plane_a'][prp.lng_geoc_deg]
        lat_b = env.sims['plane_b'][prp.lat_geod_deg]
        lon_b = env.sims['plane_b'][prp.lng_geoc_deg]
        self.assertAlmostEqual(lat_a, lat_b, places=4)
        self.assertNotEqual(lon_a, lon_b)

    def test_snapshot_rows_include_plane_id(self):
        env = self.make_env()
        env.reset()
        rows = env.snapshot_rows(episode=2, step=7)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['plane_id'], 'plane_a')
        self.assertEqual(rows[1]['plane_id'], 'plane_b')
        self.assertEqual(rows[0]['episode'], 2)
        self.assertEqual(rows[0]['step'], 7)


class TestOffsetGeodeticPosition(unittest.TestCase):
    def test_east_offset_changes_longitude(self):
        lat_deg, lon_deg = utils.offset_geodetic_position(51.3781, -2.3273, east_m=100.0)
        self.assertAlmostEqual(lat_deg, 51.3781, places=4)
        self.assertNotEqual(lon_deg, -2.3273)
