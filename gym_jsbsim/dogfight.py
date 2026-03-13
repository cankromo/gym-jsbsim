import math
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import gym
import numpy as np

from gym_jsbsim import aircraft, properties as prp, simulation as simulation_module
from gym_jsbsim.dogfight_scenarios import get_scenario
from gym_jsbsim.multi_env import FormationMember, SharedWorldJsbSimEnv
from gym_jsbsim.tasks import Shaping, TurnHeadingControlTask

try:  # pragma: no cover - optional dependency
    from pettingzoo import ParallelEnv
except Exception:  # pragma: no cover - optional dependency
    ParallelEnv = object


def apply_jsbsim_runtime_compat() -> None:
    """
    Runtime compatibility patching for newer JSBSim releases.

    This mirrors the existing SB3 helper pattern and keeps library defaults
    untouched outside of the dogfight path.
    """
    sim_cls = simulation_module.Simulation
    if getattr(sim_cls, "_dogfight_runtime_compat_applied", False):
        return

    root_dir = os.environ.get("JSBSIM_ROOT_DIR")
    if not root_dir:
        try:
            import jsbsim

            default_root = getattr(jsbsim, "get_default_root_dir", lambda: None)()
            if default_root:
                root_dir = default_root
        except Exception:
            root_dir = None

    if root_dir:
        resolved_root = os.path.abspath(os.path.expanduser(str(root_dir)))
        if os.path.isdir(resolved_root):
            sim_cls.ROOT_DIR = resolved_root

    def _initialise_compat(self, dt, model_name, init_conditions=None) -> None:
        if init_conditions is not None:
            ic_file = "minimal_ic.xml"
        else:
            ic_file = "basic_ic.xml"

        ic_path = os.path.join(os.path.dirname(os.path.abspath(simulation_module.__file__)), ic_file)
        try:
            self.jsbsim.load_ic(ic_path, useStoredPath=False)
        except TypeError:
            self.jsbsim.load_ic(ic_path, False)

        self.load_model(model_name)
        self.jsbsim.set_dt(dt)
        self.set_custom_initial_conditions(init_conditions)

        success = self.jsbsim.run_ic()
        if not success:
            raise RuntimeError("JSBSim failed to init simulation conditions.")

    sim_cls.initialise = _initialise_compat
    sim_cls._dogfight_runtime_compat_applied = True


@dataclass(frozen=True)
class RelativeGeometry:
    range_m: float
    forward_m: float
    right_m: float
    up_m: float
    bearing_error_deg: float
    elevation_error_deg: float
    heading_difference_deg: float


class PursuitDogfightTask(TurnHeadingControlTask):
    """
    Turn-task variant used for pursuit/dogfight scenarios.

    The outer dogfight env updates the target track each step based on the
    current line-of-sight to the opponent.
    """

    DEFAULT_EPISODE_STEPS = 900
    MAX_ALTITUDE_DEVIATION_FT = 3000

    def _get_target_track(self) -> float:
        if self._current_initial_heading is not None:
            return float(self._current_initial_heading)
        return float(self.FIXED_INITIAL_HEADING_DEG)

    def _is_terminal(self, sim, reward=None) -> bool:
        terminal_step = sim[self.steps_left] <= 0
        return terminal_step or self._altitude_out_of_bounds(sim)


def _geo_delta_m(sim_from, sim_to) -> tuple[float, float, float]:
    lat_from = float(sim_from[prp.lat_geod_deg])
    lon_from = float(sim_from[prp.lng_geoc_deg])
    lat_to = float(sim_to[prp.lat_geod_deg])
    lon_to = float(sim_to[prp.lng_geoc_deg])
    mean_lat_rad = math.radians((lat_from + lat_to) * 0.5)
    north_m = (lat_to - lat_from) * 111_320.0
    east_m = (lon_to - lon_from) * 111_320.0 * max(0.05, math.cos(mean_lat_rad))
    up_m = (float(sim_to[prp.altitude_sl_ft]) - float(sim_from[prp.altitude_sl_ft])) * 0.3048
    return east_m, north_m, up_m


def compute_relative_geometry(own_sim, target_sim) -> RelativeGeometry:
    east_m, north_m, up_m = _geo_delta_m(own_sim, target_sim)
    own_heading_rad = math.radians(float(own_sim[prp.heading_deg]))
    own_pitch_deg = math.degrees(float(own_sim[prp.pitch_rad]))

    forward_m = east_m * math.sin(own_heading_rad) + north_m * math.cos(own_heading_rad)
    right_m = east_m * math.cos(own_heading_rad) - north_m * math.sin(own_heading_rad)
    horiz_range_m = math.hypot(forward_m, right_m)
    range_m = math.sqrt(horiz_range_m ** 2 + up_m ** 2)
    bearing_error_deg = math.degrees(math.atan2(right_m, max(1.0, forward_m)))
    elevation_error_deg = math.degrees(math.atan2(up_m, max(1.0, horiz_range_m))) - own_pitch_deg
    heading_difference_deg = ((float(target_sim[prp.heading_deg]) - float(own_sim[prp.heading_deg]) + 180.0) % 360.0) - 180.0

    return RelativeGeometry(
        range_m=range_m,
        forward_m=forward_m,
        right_m=right_m,
        up_m=up_m,
        bearing_error_deg=bearing_error_deg,
        elevation_error_deg=elevation_error_deg,
        heading_difference_deg=heading_difference_deg,
    )


class DogfightEnv:
    """
    Additive shared-world combat environment built on top of gym-jsbsim.

    This is the core environment. It keeps two aircraft in one shared notional
    world, computes relative geometry, and returns per-agent observations and
    rewards. Existing single-aircraft envs remain untouched.
    """

    FIRE_AZIMUTH_DEG = 60.0
    FIRE_ELEVATION_DEG = 20.0
    FIRE_SOLUTION_DEG = 6.0
    FIRE_RANGE_M = 1200.0
    OPTIMAL_RANGE_M = 800.0
    RANGE_SCALE_M = 4000.0
    DEFENSIVE_CONE_WEIGHT = 0.20
    DEFENSIVE_FIRE_SOLUTION_PENALTY = 0.60
    DEFAULT_SPAWN_SEPARATION_M = 900.0
    DEFAULT_ALTITUDE_SEPARATION_FT = 200.0

    def __init__(self,
                 agent_interaction_freq: int = 5,
                 shaping: Shaping = Shaping.EXTRA_SEQUENTIAL,
                 aircraft_type=aircraft.f16,
                 spawn_separation_m: float = DEFAULT_SPAWN_SEPARATION_M,
                 altitude_separation_ft: float = DEFAULT_ALTITUDE_SEPARATION_FT,
                 spawn_seed: Optional[int] = None,
                 scenario_name: Optional[str] = None):
        apply_jsbsim_runtime_compat()
        self.agent_order = ("plane_a", "plane_b")
        self.spawn_separation_m = float(max(200.0, spawn_separation_m))
        self.altitude_separation_ft = float(max(0.0, altitude_separation_ft))
        self._spawn_rng = np.random.default_rng(spawn_seed)
        self._scenario_name = scenario_name
        self.current_scenario_name = scenario_name or "random"
        self.world = SharedWorldJsbSimEnv(
            members=(
                FormationMember(
                    name="plane_a",
                    task_type=PursuitDogfightTask,
                    aircraft=aircraft_type,
                    shaping=shaping,
                    east_offset_m=-0.5 * self.spawn_separation_m,
                    altitude_offset_ft=-0.5 * self.altitude_separation_ft,
                ),
                FormationMember(
                    name="plane_b",
                    task_type=PursuitDogfightTask,
                    aircraft=aircraft_type,
                    shaping=shaping,
                    east_offset_m=0.5 * self.spawn_separation_m,
                    altitude_offset_ft=0.5 * self.altitude_separation_ft,
                ),
            ),
            agent_interaction_freq=agent_interaction_freq,
        )
        self.observation_spaces = {
            agent: self._build_observation_space(agent)
            for agent in self.agent_order
        }
        self.action_spaces = {
            agent: self.world.action_space[agent]
            for agent in self.agent_order
        }
        self._previous_ranges: Dict[str, float] = {}

    def set_scenario(self, scenario_name: Optional[str]) -> None:
        self._scenario_name = scenario_name
        self.current_scenario_name = scenario_name or "random"

    def _apply_scenario(self, scenario_name: str) -> None:
        scenario = get_scenario(scenario_name)
        member_a, member_b = self.world.members
        member_a.north_offset_m = scenario.plane_a.north_offset_m
        member_a.east_offset_m = scenario.plane_a.east_offset_m
        member_a.altitude_offset_ft = scenario.plane_a.altitude_offset_ft
        member_a.initial_conditions[prp.initial_heading_deg] = scenario.plane_a.heading_deg
        member_b.north_offset_m = scenario.plane_b.north_offset_m
        member_b.east_offset_m = scenario.plane_b.east_offset_m
        member_b.altitude_offset_ft = scenario.plane_b.altitude_offset_ft
        member_b.initial_conditions[prp.initial_heading_deg] = scenario.plane_b.heading_deg
        self.current_scenario_name = scenario.name

    def _randomize_spawn_offsets(self) -> None:
        half_sep = 0.5 * self.spawn_separation_m
        bearing_rad = float(self._spawn_rng.uniform(0.0, 2.0 * math.pi))
        north_offset = math.cos(bearing_rad) * half_sep
        east_offset = math.sin(bearing_rad) * half_sep
        altitude_offset = 0.5 * self.altitude_separation_ft
        member_a, member_b = self.world.members
        member_a.north_offset_m = -north_offset
        member_a.east_offset_m = -east_offset
        member_a.altitude_offset_ft = -altitude_offset
        member_b.north_offset_m = north_offset
        member_b.east_offset_m = east_offset
        member_b.altitude_offset_ft = altitude_offset
        member_a.initial_conditions.pop(prp.initial_heading_deg, None)
        member_b.initial_conditions.pop(prp.initial_heading_deg, None)
        self.current_scenario_name = "random"

    def close(self):
        self.world.close()

    def _build_observation_space(self, agent: str) -> gym.spaces.Box:
        base_space = self.world.observation_space[agent]
        extra_low = np.array([0.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        extra_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        low = np.concatenate([base_space.low.astype(np.float32), extra_low])
        high = np.concatenate([base_space.high.astype(np.float32), extra_high])
        return gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def _sim(self, agent: str):
        return self.world.sims[agent]

    def _opponent(self, agent: str) -> str:
        return self.agent_order[1] if agent == self.agent_order[0] else self.agent_order[0]

    def _line_of_sight_heading_deg(self, own_sim, target_sim) -> float:
        east_m, north_m, _ = _geo_delta_m(own_sim, target_sim)
        return (math.degrees(math.atan2(east_m, north_m)) + 360.0) % 360.0

    def _update_dynamic_targets(self) -> None:
        for agent in self.agent_order:
            opponent = self._opponent(agent)
            own_sim = self._sim(agent)
            target_sim = self._sim(opponent)
            task = self.world.tasks[agent]
            own_sim[task.target_track_deg] = self._line_of_sight_heading_deg(own_sim, target_sim)

    def _observation_for(self, agent: str) -> np.ndarray:
        own_sim = self._sim(agent)
        opponent_sim = self._sim(self._opponent(agent))
        task = self.world.tasks[agent]
        base_obs = np.array([own_sim[prop] for prop in task.state_variables], dtype=np.float32)
        rel = compute_relative_geometry(own_sim, opponent_sim)
        extra = np.array([
            min(1.0, rel.range_m / self.RANGE_SCALE_M),
            rel.bearing_error_deg / 180.0,
            rel.elevation_error_deg / 90.0,
            rel.heading_difference_deg / 180.0,
            max(-1.0, min(1.0, rel.up_m / 800.0)),
            max(-1.0, min(1.0, rel.forward_m / 3000.0)),
        ], dtype=np.float32)
        return np.concatenate([base_obs, extra], dtype=np.float32)

    def _shot_quality(self, rel: RelativeGeometry) -> Dict[str, float | bool]:
        aim_quality = max(0.0, 1.0 - abs(rel.bearing_error_deg) / self.FIRE_AZIMUTH_DEG)
        elevation_quality = max(0.0, 1.0 - abs(rel.elevation_error_deg) / self.FIRE_ELEVATION_DEG)
        range_quality = 1.0 / (1.0 + abs(rel.range_m - self.OPTIMAL_RANGE_M) / self.OPTIMAL_RANGE_M)
        in_firing_cone = (
            rel.forward_m > 0.0
            and abs(rel.bearing_error_deg) <= self.FIRE_AZIMUTH_DEG
            and abs(rel.elevation_error_deg) <= self.FIRE_ELEVATION_DEG
            and rel.range_m <= self.FIRE_RANGE_M
        )
        fire_solution = (
            in_firing_cone
            and abs(rel.bearing_error_deg) <= self.FIRE_SOLUTION_DEG
            and abs(rel.elevation_error_deg) <= self.FIRE_SOLUTION_DEG
        )
        return {
            "aim_quality": aim_quality,
            "elevation_quality": elevation_quality,
            "range_quality": range_quality,
            "in_firing_cone": in_firing_cone,
            "fire_solution": fire_solution,
        }

    def _reward_for(self, agent: str) -> tuple[float, Dict]:
        own_sim = self._sim(agent)
        opponent_sim = self._sim(self._opponent(agent))
        rel = compute_relative_geometry(own_sim, opponent_sim)
        threat_rel = compute_relative_geometry(opponent_sim, own_sim)
        shot = self._shot_quality(rel)
        threat = self._shot_quality(threat_rel)
        prev_range = self._previous_ranges.get(agent)
        closure_bonus = 0.0 if prev_range is None else max(-0.15, min(0.15, (prev_range - rel.range_m) / 250.0))
        self._previous_ranges[agent] = rel.range_m
        defensive_penalty = (
            self.DEFENSIVE_CONE_WEIGHT * float(threat["aim_quality"])
            + (self.DEFENSIVE_FIRE_SOLUTION_PENALTY if threat["fire_solution"] else 0.0)
        )
        reward = (
            0.45 * float(shot["aim_quality"])
            + 0.15 * float(shot["elevation_quality"])
            + 0.25 * float(shot["range_quality"])
            + 0.15 * (closure_bonus + 0.15)
            + (0.75 if shot["fire_solution"] else 0.0)
            - defensive_penalty
        )
        return float(reward), {
            "relative_geometry": rel,
            "threat_relative_geometry": threat_rel,
            "aim_quality": float(shot["aim_quality"]),
            "elevation_quality": float(shot["elevation_quality"]),
            "range_quality": float(shot["range_quality"]),
            "closure_bonus": closure_bonus,
            "fire_solution": bool(shot["fire_solution"]),
            "defensive_cone_penalty": self.DEFENSIVE_CONE_WEIGHT * float(threat["aim_quality"]),
            "defensive_fire_solution": bool(threat["fire_solution"]),
            "defensive_penalty": defensive_penalty,
        }

    def reset(self) -> Dict[str, np.ndarray]:
        if self._scenario_name:
            self._apply_scenario(self._scenario_name)
        else:
            self._randomize_spawn_offsets()
        self.world.reset()
        self._update_dynamic_targets()
        self._previous_ranges = {}
        return {
            agent: self._observation_for(agent)
            for agent in self.agent_order
        }

    def step(self, actions: Mapping[str, np.ndarray]):
        self._update_dynamic_targets()
        _, _, world_done, info = self.world.step(actions)
        observations = {
            agent: self._observation_for(agent)
            for agent in self.agent_order
        }
        rewards = {}
        infos = {}
        for agent in self.agent_order:
            reward, reward_info = self._reward_for(agent)
            rewards[agent] = reward
            infos[agent] = reward_info
        dones = {agent: bool(world_done) for agent in self.agent_order}
        dones["__all__"] = bool(world_done)
        return observations, rewards, dones, infos

    def sample_random_actions(self, rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
        rng = rng or np.random.default_rng()
        actions = {}
        for agent in self.agent_order:
            space = self.action_spaces[agent]
            actions[agent] = rng.uniform(low=space.low, high=space.high).astype(np.float32)
        return actions

    def telemetry_rows(self, episode: int, step: int, rewards: Mapping[str, float], dones: Mapping[str, bool]) -> list[dict]:
        rows = self.world.snapshot_rows(episode=episode, step=step, rewards=rewards, dones=dones)
        by_agent = {row["plane_id"]: row for row in rows}
        for agent in self.agent_order:
            row = by_agent[agent]
            own_sim = self._sim(agent)
            opponent_sim = self._sim(self._opponent(agent))
            rel = compute_relative_geometry(own_sim, opponent_sim)
            task = self.world.tasks[agent]
            track_err = float(own_sim[task.track_error_deg])
            target_roll_deg = float(task.get_target_roll_deg(track_err)) if hasattr(task, "get_target_roll_deg") else 0.0
            current_roll_deg = math.degrees(float(own_sim[prp.roll_rad]))
            row["target_roll_deg"] = target_roll_deg
            row["current_roll_deg"] = current_roll_deg
            row["roll_error_deg"] = current_roll_deg - target_roll_deg
            row["range_m"] = rel.range_m
            row["bearing_error_deg"] = rel.bearing_error_deg
            row["elevation_error_deg"] = rel.elevation_error_deg
            row["heading_difference_deg"] = rel.heading_difference_deg
            row["opponent_plane_id"] = self._opponent(agent)
            row["scenario_name"] = self.current_scenario_name
        return rows


class DogfightParallelEnv(ParallelEnv):  # pragma: no cover - PettingZoo optional at runtime
    metadata = {"name": "gym_jsbsim_dogfight_parallel_v0"}

    def __init__(self,
                 agent_interaction_freq: int = 5,
                 shaping: Shaping = Shaping.EXTRA_SEQUENTIAL,
                 aircraft_type=aircraft.f16):
        if ParallelEnv is object:
            raise RuntimeError("pettingzoo is not installed")
        self.core = DogfightEnv(
            agent_interaction_freq=agent_interaction_freq,
            shaping=shaping,
            aircraft_type=aircraft_type,
        )
        self.possible_agents = list(self.core.agent_order)
        self.agents = list(self.possible_agents)
        self.observation_spaces = dict(self.core.observation_spaces)
        self.action_spaces = dict(self.core.action_spaces)

    def reset(self, seed=None, options=None):
        del seed, options
        self.agents = list(self.possible_agents)
        observations = self.core.reset()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        observations, rewards, dones, infos = self.core.step(actions)
        terminations = {agent: dones[agent] for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        self.agents = [] if dones["__all__"] else list(self.possible_agents)
        return observations, rewards, terminations, truncations, infos

    def close(self):
        self.core.close()


def telemetry_fieldnames(rows: Sequence[Dict]) -> list[str]:
    fieldnames = ["plane_id", "episode", "step", "reward", "done"]
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames
