import collections
import math
from typing import Dict, Mapping, Sequence, Tuple, Type

import gym
import numpy as np

from gym_jsbsim import properties as prp
from gym_jsbsim import utils
from gym_jsbsim.aircraft import Aircraft, cessna172P
from gym_jsbsim.simulation import Simulation
from gym_jsbsim.tasks import HeadingControlTask, Shaping


class FormationMember(object):
    """
    Configuration for one aircraft inside a shared-world environment.
    """

    def __init__(self,
                 name: str,
                 task_type: Type[HeadingControlTask] = HeadingControlTask,
                 aircraft: Aircraft = cessna172P,
                 shaping: Shaping = Shaping.STANDARD,
                 north_offset_m: float = 0.0,
                 east_offset_m: float = 0.0,
                 altitude_offset_ft: float = 0.0,
                 initial_conditions: Dict[prp.Property, float] = None):
        self.name = name
        self.task_type = task_type
        self.aircraft = aircraft
        self.shaping = shaping
        self.north_offset_m = north_offset_m
        self.east_offset_m = east_offset_m
        self.altitude_offset_ft = altitude_offset_ft
        self.initial_conditions = dict(initial_conditions or {})


class SharedWorldJsbSimEnv(gym.Env):
    """
    Additive multi-aircraft wrapper for a shared notional world.

    Each aircraft still has its own JSBSim simulation instance. The shared
    world is represented by common geodetic initial conditions and consistent
    local offsets between aircraft.
    """

    JSBSIM_DT_HZ = 60
    metadata = {'render.modes': ['human']}
    PLANE_ID = 'plane_id'

    def __init__(self,
                 members: Sequence[FormationMember],
                 agent_interaction_freq: int = 5):
        self.members = tuple(members)
        if len(self.members) != 2:
            raise ValueError('SharedWorldJsbSimEnv currently supports exactly two aircraft')
        if len({member.name for member in self.members}) != len(self.members):
            raise ValueError('formation member names must be unique')
        if agent_interaction_freq > self.JSBSIM_DT_HZ:
            raise ValueError('agent interaction frequency must be <= JSBSIM_DT_HZ')

        self.sim_steps_per_agent_step = self.JSBSIM_DT_HZ // agent_interaction_freq
        self.tasks = collections.OrderedDict(
            (member.name, member.task_type(member.shaping, agent_interaction_freq, member.aircraft))
            for member in self.members
        )
        self.sims = collections.OrderedDict((member.name, None) for member in self.members)
        self.observation_space = gym.spaces.Dict(collections.OrderedDict(
            (member.name, self.tasks[member.name].get_state_space())
            for member in self.members
        ))
        self.action_space = gym.spaces.Dict(collections.OrderedDict(
            (member.name, self.tasks[member.name].get_action_space())
            for member in self.members
        ))

    def _init_new_sim(self, aircraft: Aircraft, initial_conditions: Dict[prp.Property, float]) -> Simulation:
        return Simulation(sim_frequency_hz=self.JSBSIM_DT_HZ,
                          aircraft=aircraft,
                          init_conditions=initial_conditions,
                          allow_flightgear_output=False)

    def _base_world_conditions(self) -> Tuple[float, float, float]:
        leader = self.members[0]
        leader_task = self.tasks[leader.name]
        init_conditions = dict(leader_task.get_initial_conditions() or {})
        init_conditions.update(leader.initial_conditions)
        base_ic = HeadingControlTask.base_initial_conditions
        latitude_deg = float(init_conditions.get(prp.initial_latitude_geod_deg,
                                                 base_ic[prp.initial_latitude_geod_deg]))
        longitude_deg = float(init_conditions.get(prp.initial_longitude_geoc_deg,
                                                  base_ic[prp.initial_longitude_geoc_deg]))
        altitude_ft = float(init_conditions.get(prp.initial_altitude_ft,
                                                base_ic[prp.initial_altitude_ft]))
        return latitude_deg, longitude_deg, altitude_ft

    def _member_initial_conditions(self, member: FormationMember) -> Dict[prp.Property, float]:
        task = self.tasks[member.name]
        init_conditions = dict(task.get_initial_conditions() or {})
        init_conditions.update(member.initial_conditions)

        world_lat_deg, world_lon_deg, world_alt_ft = self._base_world_conditions()
        latitude_deg, longitude_deg = utils.offset_geodetic_position(
            world_lat_deg,
            world_lon_deg,
            north_m=member.north_offset_m,
            east_m=member.east_offset_m,
        )
        init_conditions[prp.initial_latitude_geod_deg] = latitude_deg
        init_conditions[prp.initial_longitude_geoc_deg] = longitude_deg
        init_conditions[prp.initial_altitude_ft] = world_alt_ft + member.altitude_offset_ft
        return init_conditions

    def reset(self):
        observations = collections.OrderedDict()
        for member in self.members:
            init_conditions = self._member_initial_conditions(member)
            sim = self.sims[member.name]
            if sim is None:
                sim = self._init_new_sim(member.aircraft, init_conditions)
                self.sims[member.name] = sim
            else:
                sim.reinitialise(init_conditions)

            state = self.tasks[member.name].observe_first_state(sim)
            observations[member.name] = np.array(state)
        return observations

    def _normalise_actions(self, actions):
        if isinstance(actions, Mapping):
            return collections.OrderedDict(
                (member.name, np.asarray(actions[member.name]))
                for member in self.members
            )
        if len(actions) != len(self.members):
            raise ValueError('actions must contain one entry per aircraft')
        return collections.OrderedDict(
            (member.name, np.asarray(action))
            for member, action in zip(self.members, actions)
        )

    def step(self, actions):
        actions_by_member = self._normalise_actions(actions)
        observations = collections.OrderedDict()
        rewards = collections.OrderedDict()
        dones = collections.OrderedDict()
        infos = collections.OrderedDict()

        for member in self.members:
            task = self.tasks[member.name]
            sim = self.sims[member.name]
            if sim is None:
                raise RuntimeError('reset() must be called before step()')

            action = actions_by_member[member.name]
            if action.shape != task.get_action_space().shape:
                raise ValueError(f'action for {member.name} does not match action space')

            state, reward, done, info = task.task_step(sim, action, self.sim_steps_per_agent_step)
            observations[member.name] = np.array(state)
            rewards[member.name] = reward
            dones[member.name] = done
            infos[member.name] = info

        world_done = any(dones.values())
        return observations, rewards, world_done, {
            'dones': dones,
            'members': infos,
            'telemetry': self.snapshot_rows(rewards=rewards, dones=dones),
        }

    def close(self):
        for sim in self.sims.values():
            if sim:
                sim.close()

    def _telemetry_props(self, task) -> Tuple[prp.Property, ...]:
        props = list(task.state_variables)
        extras = [
            prp.heading_deg,
            prp.lat_geod_deg,
            prp.lng_geoc_deg,
            prp.altitude_rate_fps,
            prp.v_north_fps,
            prp.v_east_fps,
            prp.engine_thrust_lbs,
            prp.engine_running,
            prp.throttle_cmd,
            prp.mixture_cmd,
            prp.aileron_cmd,
            prp.elevator_cmd,
            prp.rudder_cmd,
            prp.gear,
            task.last_agent_reward,
            task.last_assessment_reward,
        ]
        for attr_name in ('target_track_deg', 'track_error_deg', 'altitude_error_ft',
                          'steps_left', 'target_roll_rad'):
            prop = getattr(task, attr_name, None)
            if prop is not None and prop not in props:
                extras.append(prop)
        for prop in extras:
            if prop not in props:
                props.append(prop)
        return tuple(props)

    def snapshot_rows(self,
                      episode: int = None,
                      step: int = None,
                      rewards: Mapping[str, float] = None,
                      dones: Mapping[str, bool] = None) -> list:
        rows = []
        rewards = rewards or {}
        dones = dones or {}
        for member in self.members:
            name = member.name
            sim = self.sims[name]
            task = self.tasks[name]
            if sim is None:
                continue

            row = {self.PLANE_ID: name}
            if episode is not None:
                row['episode'] = int(episode)
            if step is not None:
                row['step'] = int(step)
            row['reward'] = float(rewards.get(name, sim[task.last_agent_reward]))
            row['done'] = bool(dones.get(name, False))
            for prop in self._telemetry_props(task):
                try:
                    row[prop.get_legal_name()] = float(sim[prop])
                except Exception:
                    continue
            row['roll_deg'] = math.degrees(float(sim[prp.roll_rad]))
            row['pitch_deg'] = math.degrees(float(sim[prp.pitch_rad]))
            rows.append(row)
        return rows
