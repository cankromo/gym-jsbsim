import argparse
import csv
import os

import shimmy

from gym_jsbsim.dogfight import DogfightEnv, telemetry_fieldnames
from gym_jsbsim.dogfight_policies import StableOpponentPolicy, load_sb3_policy_opponent
from gym_jsbsim.dogfight_sb3_env import DogfightSingleAgentEnv


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_DIR = os.path.join(REPO_ROOT, "models", "dogfight")
DEFAULT_MODEL_A_DIR = os.path.join(DEFAULT_MODEL_DIR, "plane_a")
DEFAULT_MODEL_B_DIR = os.path.join(DEFAULT_MODEL_DIR, "plane_b")
DEFAULT_MODEL_A_PATH = os.path.join(DEFAULT_MODEL_A_DIR, "ppo_dogfight_plane_a.zip")
DEFAULT_VECNORM_A_PATH = os.path.join(DEFAULT_MODEL_A_DIR, "vecnormalize.pkl")
DEFAULT_MODEL_B_PATH = os.path.join(DEFAULT_MODEL_B_DIR, "ppo_dogfight_plane_b.zip")
DEFAULT_VECNORM_B_PATH = os.path.join(DEFAULT_MODEL_B_DIR, "vecnormalize.pkl")
DEFAULT_EVAL_CSV = os.path.join(REPO_ROOT, "models", "eval_outputs", "dogfight_duel_eval.csv")


def _wrap_gymnasium_if_needed(env):
    return shimmy.GymV21CompatibilityV0(env=env)


def _make_vec_env(controlled_agent: str, opponent_policy):
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    def _init():
        env = DogfightSingleAgentEnv(
            controlled_agent=controlled_agent,
            opponent_policy=opponent_policy,
        )
        env = _wrap_gymnasium_if_needed(env)
        return Monitor(env)

    vec_env = DummyVecEnv([_init])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


def _load_frozen_opponent(agent: str, model_path: str, vecnormalize_path: str):
    core = DogfightEnv()
    try:
        return load_sb3_policy_opponent(
            model_path=model_path,
            observation_space=core.observation_spaces[agent],
            action_space=core.action_spaces[agent],
            vecnormalize_path=vecnormalize_path,
            deterministic=True,
        )
    finally:
        core.close()


def train_single_agent(controlled_agent: str,
                       total_timesteps: int,
                       seed: int,
                       model_dir: str,
                       opponent_policy) -> None:
    from stable_baselines3 import PPO

    os.makedirs(model_dir, exist_ok=True)
    vec_env = _make_vec_env(controlled_agent=controlled_agent, opponent_policy=opponent_policy)
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
    model_name = f"ppo_dogfight_{controlled_agent}"
    model.save(os.path.join(model_dir, model_name))
    vec_env.save(os.path.join(model_dir, "vecnormalize.pkl"))
    vec_env.close()


def eval_duel(episodes: int,
              max_steps: int,
              model_a_path: str,
              vecnorm_a_path: str,
              model_b_path: str,
              vecnorm_b_path: str,
              csv_path: str) -> None:
    import numpy as np

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    core = DogfightEnv()
    policy_a = _load_frozen_opponent("plane_a", model_a_path, vecnorm_a_path)
    policy_b = _load_frozen_opponent("plane_b", model_b_path, vecnorm_b_path)
    rows = []
    try:
        for episode in range(episodes):
            obs = core.reset()
            policy_a.reset()
            policy_b.reset()
            for step in range(max_steps):
                action_a = policy_a.action(obs["plane_a"], core.action_spaces["plane_a"])
                action_b = policy_b.action(obs["plane_b"], core.action_spaces["plane_b"])
                obs, rewards, dones, infos = core.step({
                    "plane_a": np.asarray(action_a, dtype=np.float32),
                    "plane_b": np.asarray(action_b, dtype=np.float32),
                })
                rows.extend(core.telemetry_rows(episode=episode, step=step, rewards=rewards, dones=dones))
                if dones.get("__all__", False):
                    break
    finally:
        core.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = telemetry_fieldnames(rows)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train additive dogfight policies with SB3.")
    parser.add_argument("--mode", choices=("train_a", "train_b", "eval"), required=True)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--model-a-path", default=DEFAULT_MODEL_A_PATH)
    parser.add_argument("--vecnorm-a-path", default=DEFAULT_VECNORM_A_PATH)
    parser.add_argument("--model-b-path", default=DEFAULT_MODEL_B_PATH)
    parser.add_argument("--vecnorm-b-path", default=DEFAULT_VECNORM_B_PATH)
    parser.add_argument("--csv-path", default=DEFAULT_EVAL_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train_a":
        train_single_agent(
            controlled_agent="plane_a",
            total_timesteps=args.timesteps,
            seed=args.seed,
            model_dir=DEFAULT_MODEL_A_DIR,
            opponent_policy=StableOpponentPolicy(),
        )
        return

    if args.mode == "train_b":
        opponent_policy = _load_frozen_opponent("plane_a", args.model_a_path, args.vecnorm_a_path)
        train_single_agent(
            controlled_agent="plane_b",
            total_timesteps=args.timesteps,
            seed=args.seed,
            model_dir=DEFAULT_MODEL_B_DIR,
            opponent_policy=opponent_policy,
        )
        return

    eval_duel(
        episodes=args.episodes,
        max_steps=args.max_steps,
        model_a_path=args.model_a_path,
        vecnorm_a_path=args.vecnorm_a_path,
        model_b_path=args.model_b_path,
        vecnorm_b_path=args.vecnorm_b_path,
        csv_path=args.csv_path,
    )


if __name__ == "__main__":
    main()
