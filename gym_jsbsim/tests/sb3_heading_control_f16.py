import argparse
import csv
import math
import os
import random
from typing import Iterable, List

import numpy as np

from gym_jsbsim import tasks, aircraft, properties as prp
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim import simulation as simulation_module

try:
    import shimmy
except Exception:  # pragma: no cover - optional dependency
    shimmy = None


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "models", "heading_control_f16")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "ppo_heading_control_f16.zip")
DEFAULT_VECNORM_PATH = os.path.join(DEFAULT_MODEL_DIR, "vecnormalize.pkl")
DEFAULT_EVAL_CSV = os.path.join(REPO_ROOT, "models", "eval_outputs", "heading_control_f16_eval_metrics.csv")
TASK_MAP = {
    "heading": tasks.HeadingControlTask,
    "turn": tasks.TurnHeadingControlTask,
}


class EvalRandomTargetTurnHeadingControlTask(tasks.TurnHeadingControlTask):
    """Eval-only task variant: keep task dynamics, but sample random target heading."""

    def _get_target_track(self) -> float:
        return random.uniform(self.target_track_deg.min, self.target_track_deg.max)


def _apply_jsbsim_runtime_compat() -> None:
    """
    Runtime compatibility patching for newer JSBSim releases.
    Keeps library defaults untouched by patching only in this script.
    """
    sim_cls = simulation_module.Simulation
    if getattr(sim_cls, "_sb3_runtime_compat_applied", False):
        return

    # Prefer explicit override, otherwise use jsbsim package default root dir.
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
    sim_cls._sb3_runtime_compat_applied = True


def _make_env(shaping: tasks.Shaping, agent_interaction_freq: int, task_type) -> JsbSimEnv:
    _apply_jsbsim_runtime_compat()
    return JsbSimEnv(
        task_type=task_type,
        aircraft=aircraft.f16,
        agent_interaction_freq=agent_interaction_freq,
        shaping=shaping,
    )


def _wrap_gymnasium_if_needed(env):
    if shimmy is None:
        raise RuntimeError(
            "shimmy is required with stable-baselines3>=2 to wrap Gym envs. "
            "Install with: pip install shimmy"
        )
    return shimmy.GymV21CompatibilityV0(env=env)


def _make_vec_env(shaping: tasks.Shaping, agent_interaction_freq: int, task_type):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _init():
        env = _make_env(
            shaping=shaping,
            agent_interaction_freq=agent_interaction_freq,
            task_type=task_type,
        )
        env = _wrap_gymnasium_if_needed(env)
        return Monitor(env)

    vec_env = DummyVecEnv([_init])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def train_heading_control_f16(
    total_timesteps: int = 2_000_000,
    seed: int = 0,
    model_dir: str = DEFAULT_MODEL_DIR,
    agent_interaction_freq: int = 5,
    shaping: tasks.Shaping = tasks.Shaping.EXTRA_SEQUENTIAL,
    task_type=tasks.HeadingControlTask,
):
    from stable_baselines3 import PPO

    os.makedirs(model_dir, exist_ok=True)
    vec_env = _make_vec_env(
        shaping=shaping,
        agent_interaction_freq=agent_interaction_freq,
        task_type=task_type,
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=seed,
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(os.path.join(model_dir, "ppo_heading_control_f16"))
    vec_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    vec_env.close()


def _extend_props(props: List[prp.Property], extra: Iterable[prp.Property]) -> None:
    for prop in extra:
        if prop not in props:
            props.append(prop)


def _metric_props(env: JsbSimEnv) -> List[prp.Property]:
    props: List[prp.Property] = list(env.task.state_variables)
    extras = [
        prp.heading_deg,
        prp.lat_geod_deg,
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
        env.task.target_track_deg,
        env.task.track_error_deg,
        env.task.altitude_error_ft,
        env.task.last_agent_reward,
        env.task.last_assessment_reward,
        env.task.steps_left,
    ]
    _extend_props(props, extras)
    return props


def evaluate_heading_control_f16(
    model_path: str = DEFAULT_MODEL_PATH,
    vecnormalize_path: str = DEFAULT_VECNORM_PATH,
    episodes: int = 5,
    csv_path: str = DEFAULT_EVAL_CSV,
    deterministic: bool = True,
    print_every: int = 25,
    agent_interaction_freq: int = 5,
    shaping: tasks.Shaping = tasks.Shaping.EXTRA_SEQUENTIAL,
    task_type=tasks.HeadingControlTask,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _init():
        env = _make_env(
            shaping=shaping,
            agent_interaction_freq=agent_interaction_freq,
            task_type=task_type,
        )
        env = _wrap_gymnasium_if_needed(env)
        return Monitor(env)

    vec_env = DummyVecEnv([_init])
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(model_path, env=vec_env)

    base_env: JsbSimEnv = vec_env.envs[0].unwrapped
    props = _metric_props(base_env)

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = ["episode", "step", "reward", "done"] + [
        prop.get_legal_name() for prop in props
    ] + ["roll_deg", "pitch_deg"]

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for episode in range(episodes):
            obs = vec_env.reset()
            done = False
            step = 0
            ep_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = vec_env.step(action)
                reward_scalar = float(reward[0])
                done_flag = bool(done[0])
                ep_reward += reward_scalar

                sim = base_env.sim
                row = {
                    "episode": episode,
                    "step": step,
                    "reward": reward_scalar,
                    "done": done_flag,
                }
                for prop in props:
                    row[prop.get_legal_name()] = float(sim[prop])
                row["roll_deg"] = math.degrees(sim[prp.roll_rad])
                row["pitch_deg"] = math.degrees(sim[prp.pitch_rad])

                writer.writerow(row)

                if print_every > 0 and step % print_every == 0:
                    heading = sim[prp.heading_deg]
                    latitude = sim[prp.lat_geod_deg]
                    roll_deg = row["roll_deg"]
                    pitch_deg = row["pitch_deg"]
                    altitude = sim[prp.altitude_sl_ft]
                    track_err = sim[base_env.task.track_error_deg]
                    print(
                        f"ep={episode} step={step} reward={reward_scalar:.3f} "
                        f"heading_deg={heading:.2f} roll_deg={roll_deg:.2f} "
                        f"pitch_deg={pitch_deg:.2f} lat_deg={latitude:.5f} altitude_ft={altitude:.1f} "
                        f"track_error_deg={track_err:.2f}"
                    )

                done = done_flag
                step += 1

            print(f"episode {episode} total_reward={ep_reward:.3f}")

    vec_env.close()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Stable-Baselines3 training/evaluation for HeadingControlTask with F-16."
    )
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--vecnormalize-path", type=str, default=DEFAULT_VECNORM_PATH)
    parser.add_argument("--csv-path", type=str, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--agent-freq", type=int, default=5)
    parser.add_argument("--shaping", type=str, default=tasks.Shaping.EXTRA_SEQUENTIAL.name)
    parser.add_argument("--task", type=str, default="turn", choices=sorted(TASK_MAP.keys()))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--eval-random-target", action="store_true")
    return parser.parse_args()


def _parse_shaping(shaping_str: str) -> tasks.Shaping:
    return tasks.Shaping[shaping_str]


def main():
    args = _parse_args()
    shaping = _parse_shaping(args.shaping)
    task_type = TASK_MAP[args.task]

    if args.mode == "train":
        train_heading_control_f16(
            total_timesteps=args.timesteps,
            seed=args.seed,
            model_dir=args.model_dir,
            agent_interaction_freq=args.agent_freq,
            shaping=shaping,
            task_type=task_type,
        )
    else:
        eval_task_type = task_type
        if args.task == "turn" and args.eval_random_target:
            eval_task_type = EvalRandomTargetTurnHeadingControlTask
        evaluate_heading_control_f16(
            model_path=args.model_path,
            vecnormalize_path=args.vecnormalize_path,
            episodes=args.episodes,
            csv_path=args.csv_path,
            deterministic=args.deterministic,
            print_every=args.print_every,
            agent_interaction_freq=args.agent_freq,
            shaping=shaping,
            task_type=eval_task_type,
        )


if __name__ == "__main__":
    main()
