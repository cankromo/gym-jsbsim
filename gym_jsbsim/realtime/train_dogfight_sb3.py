import argparse
import csv
import os
import shutil

import numpy as np
import shimmy

from gym_jsbsim.dogfight import DogfightEnv, telemetry_fieldnames
from gym_jsbsim.dogfight_policies import StableOpponentPolicy, load_sb3_policy_opponent
from gym_jsbsim.dogfight_scenarios import list_scenarios
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


def _make_vec_env(controlled_agent: str, opponent_policy, vecnormalize_path: str = ""):
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
    if vecnormalize_path and os.path.exists(vecnormalize_path):
        vec_env = VecNormalize.load(vecnormalize_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
    else:
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
                       opponent_policy,
                       model_path: str = "",
                       vecnormalize_path: str = "",
                       snapshot_tag: str = "") -> None:
    from stable_baselines3 import PPO

    os.makedirs(model_dir, exist_ok=True)
    save_model_path = os.path.join(model_dir, f"ppo_dogfight_{controlled_agent}")
    save_vecnorm_path = os.path.join(model_dir, "vecnormalize.pkl")
    load_model_path = model_path or save_model_path
    load_vecnorm_path = vecnormalize_path or save_vecnorm_path
    resuming = os.path.exists(load_model_path + ".zip")

    vec_env = _make_vec_env(
        controlled_agent=controlled_agent,
        opponent_policy=opponent_policy,
        vecnormalize_path=load_vecnorm_path,
    )
    if resuming:
        model = PPO.load(load_model_path, env=vec_env)
    else:
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
    model.learn(total_timesteps=total_timesteps, reset_num_timesteps=not resuming)
    model_name = f"ppo_dogfight_{controlled_agent}"
    model.save(os.path.join(model_dir, model_name))
    vec_env.save(save_vecnorm_path)
    if snapshot_tag:
        snapshot_dir = os.path.join(model_dir, "history", snapshot_tag)
        os.makedirs(snapshot_dir, exist_ok=True)
        shutil.copy2(os.path.join(model_dir, model_name + ".zip"), os.path.join(snapshot_dir, model_name + ".zip"))
        shutil.copy2(save_vecnorm_path, os.path.join(snapshot_dir, "vecnormalize.pkl"))
    vec_env.close()


def train_self_play_cycle(rounds: int,
                          total_timesteps: int,
                          seed: int,
                          model_a_dir: str,
                          model_b_dir: str,
                          model_a_path: str,
                          vecnorm_a_path: str,
                          model_b_path: str,
                          vecnorm_b_path: str) -> None:
    # Bootstrap plane_a once if there is no existing A policy.
    if not os.path.exists(model_a_path):
        train_single_agent(
            controlled_agent="plane_a",
            total_timesteps=total_timesteps,
            seed=seed,
            model_dir=model_a_dir,
            opponent_policy=StableOpponentPolicy(),
            model_path=model_a_path.removesuffix(".zip"),
            vecnormalize_path=vecnorm_a_path,
            snapshot_tag="bootstrap_a",
        )

    for round_idx in range(1, rounds + 1):
        opponent_policy_b = _load_frozen_opponent("plane_a", model_a_path, vecnorm_a_path)
        train_single_agent(
            controlled_agent="plane_b",
            total_timesteps=total_timesteps,
            seed=seed + round_idx,
            model_dir=model_b_dir,
            opponent_policy=opponent_policy_b,
            model_path=model_b_path.removesuffix(".zip"),
            vecnormalize_path=vecnorm_b_path,
            snapshot_tag=f"round_{round_idx:03d}_plane_b",
        )

        opponent_policy_a = _load_frozen_opponent("plane_b", model_b_path, vecnorm_b_path)
        train_single_agent(
            controlled_agent="plane_a",
            total_timesteps=total_timesteps,
            seed=seed + 10_000 + round_idx,
            model_dir=model_a_dir,
            opponent_policy=opponent_policy_a,
            model_path=model_a_path.removesuffix(".zip"),
            vecnormalize_path=vecnorm_a_path,
            snapshot_tag=f"round_{round_idx:03d}_plane_a",
        )


def _eval_rollout(episodes: int,
                  max_steps: int,
                  policy_a,
                  policy_b,
                  csv_path: str,
                  csv_plane_id: str = "all",
                  scenario_names: list[str] | None = None) -> None:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = []
    scenario_cycle = scenario_names or [None]
    core = DogfightEnv(scenario_name=scenario_cycle[0])
    try:
        episode_index = 0
        for scenario_name in scenario_cycle:
            for _ in range(episodes):
                core.set_scenario(scenario_name)
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
                    del infos
                    step_rows = core.telemetry_rows(episode=episode_index, step=step, rewards=rewards, dones=dones)
                    if csv_plane_id != "all":
                        step_rows = [row for row in step_rows if row["plane_id"] == csv_plane_id]
                    rows.extend(step_rows)
                    if dones.get("__all__", False):
                        break
                episode_index += 1
    finally:
        core.close()

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = telemetry_fieldnames(rows)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def eval_duel(episodes: int,
              max_steps: int,
              model_a_path: str,
              vecnorm_a_path: str,
              model_b_path: str,
              vecnorm_b_path: str,
              csv_path: str,
              csv_plane_id: str = "all",
              scenario_names: list[str] | None = None) -> None:
    policy_a = _load_frozen_opponent("plane_a", model_a_path, vecnorm_a_path)
    policy_b = _load_frozen_opponent("plane_b", model_b_path, vecnorm_b_path)
    _eval_rollout(
        episodes=episodes,
        max_steps=max_steps,
        policy_a=policy_a,
        policy_b=policy_b,
        csv_path=csv_path,
        csv_plane_id=csv_plane_id,
        scenario_names=scenario_names,
    )


def eval_single_agent(controlled_agent: str,
                      episodes: int,
                      max_steps: int,
                      model_path: str,
                      vecnorm_path: str,
                      csv_path: str,
                      csv_plane_id: str = "all",
                      scenario_names: list[str] | None = None) -> None:
    if controlled_agent == "plane_a":
        policy_a = _load_frozen_opponent("plane_a", model_path, vecnorm_path)
        policy_b = StableOpponentPolicy()
    else:
        policy_a = StableOpponentPolicy()
        policy_b = _load_frozen_opponent("plane_b", model_path, vecnorm_path)

    _eval_rollout(
        episodes=episodes,
        max_steps=max_steps,
        policy_a=policy_a,
        policy_b=policy_b,
        csv_path=csv_path,
        csv_plane_id=csv_plane_id,
        scenario_names=scenario_names,
    )


def _resolve_eval_scenarios(scenario_set: str, scenario_name: str) -> list[str] | None:
    if scenario_name:
        return [scenario_name]
    if scenario_set == "all":
        return list_scenarios()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train additive dogfight policies with SB3.")
    parser.add_argument("--mode", choices=("train_a", "train_b", "train_cycle", "eval", "eval_a", "eval_b"), required=True)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--model-a-dir", default=DEFAULT_MODEL_A_DIR)
    parser.add_argument("--model-b-dir", default=DEFAULT_MODEL_B_DIR)
    parser.add_argument("--model-a-path", default=DEFAULT_MODEL_A_PATH)
    parser.add_argument("--vecnorm-a-path", default=DEFAULT_VECNORM_A_PATH)
    parser.add_argument("--model-b-path", default=DEFAULT_MODEL_B_PATH)
    parser.add_argument("--vecnorm-b-path", default=DEFAULT_VECNORM_B_PATH)
    parser.add_argument("--csv-path", default=DEFAULT_EVAL_CSV)
    parser.add_argument("--csv-plane-id", choices=("all", "plane_a", "plane_b"), default="all")
    parser.add_argument("--scenario-set", choices=("all", "random"), default="all")
    parser.add_argument("--scenario-name", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "train_a":
        train_single_agent(
            controlled_agent="plane_a",
            total_timesteps=args.timesteps,
            seed=args.seed,
            model_dir=args.model_a_dir,
            opponent_policy=StableOpponentPolicy(),
        )
        return

    if args.mode == "train_b":
        opponent_policy = _load_frozen_opponent("plane_a", args.model_a_path, args.vecnorm_a_path)
        train_single_agent(
            controlled_agent="plane_b",
            total_timesteps=args.timesteps,
            seed=args.seed,
            model_dir=args.model_b_dir,
            opponent_policy=opponent_policy,
        )
        return

    if args.mode == "train_cycle":
        train_self_play_cycle(
            rounds=args.rounds,
            total_timesteps=args.timesteps,
            seed=args.seed,
            model_a_dir=args.model_a_dir,
            model_b_dir=args.model_b_dir,
            model_a_path=args.model_a_path,
            vecnorm_a_path=args.vecnorm_a_path,
            model_b_path=args.model_b_path,
            vecnorm_b_path=args.vecnorm_b_path,
        )
        return

    if args.mode == "eval_a":
        scenario_names = _resolve_eval_scenarios(args.scenario_set, args.scenario_name)
        eval_single_agent(
            controlled_agent="plane_a",
            episodes=args.episodes,
            max_steps=args.max_steps,
            model_path=args.model_a_path,
            vecnorm_path=args.vecnorm_a_path,
            csv_path=args.csv_path,
            csv_plane_id=args.csv_plane_id,
            scenario_names=scenario_names,
        )
        return

    if args.mode == "eval_b":
        scenario_names = _resolve_eval_scenarios(args.scenario_set, args.scenario_name)
        eval_single_agent(
            controlled_agent="plane_b",
            episodes=args.episodes,
            max_steps=args.max_steps,
            model_path=args.model_b_path,
            vecnorm_path=args.vecnorm_b_path,
            csv_path=args.csv_path,
            csv_plane_id=args.csv_plane_id,
            scenario_names=scenario_names,
        )
        return

    scenario_names = _resolve_eval_scenarios(args.scenario_set, args.scenario_name)
    eval_duel(
        episodes=args.episodes,
        max_steps=args.max_steps,
        model_a_path=args.model_a_path,
        vecnorm_a_path=args.vecnorm_a_path,
        model_b_path=args.model_b_path,
        vecnorm_b_path=args.vecnorm_b_path,
        csv_path=args.csv_path,
        csv_plane_id=args.csv_plane_id,
        scenario_names=scenario_names,
    )


if __name__ == "__main__":
    main()
