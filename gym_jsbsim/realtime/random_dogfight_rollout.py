import argparse
import csv
import os

import numpy as np

from gym_jsbsim.dogfight import DogfightEnv, telemetry_fieldnames


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CSV = os.path.join(REPO_ROOT, "models", "eval_outputs", "random_dogfight_eval.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a random two-aircraft dogfight rollout and save combined telemetry CSV.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--agent-freq", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-path", type=str, default=DEFAULT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.csv_path), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    env = DogfightEnv(agent_interaction_freq=args.agent_freq)
    all_rows = []

    try:
        for episode in range(args.episodes):
            env.reset()
            for step in range(args.max_steps):
                actions = env.sample_random_actions(rng)
                _, rewards, dones, _ = env.step(actions)
                rows = env.telemetry_rows(episode=episode, step=step, rewards=rewards, dones=dones)
                all_rows.extend(rows)
                if dones["__all__"]:
                    break
    finally:
        env.close()

    fieldnames = telemetry_fieldnames(all_rows)
    with open(args.csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"Saved combined dogfight CSV: {args.csv_path}")


if __name__ == "__main__":
    main()
