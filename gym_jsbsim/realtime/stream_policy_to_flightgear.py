import argparse
import os

from gym_jsbsim import tasks
from gym_jsbsim.tests.sb3_heading_control_f16 import (
    TASK_MAP,
    _apply_jsbsim_runtime_compat,
    _make_env,
    _wrap_gymnasium_if_needed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SB3 policy and stream JSBSim output to FlightGear UDP.")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--vecnormalize-path", type=str, default="")
    parser.add_argument("--task", type=str, default="turn", choices=sorted(TASK_MAP.keys()))
    parser.add_argument("--shaping", type=str, default=tasks.Shaping.EXTRA_SEQUENTIAL.name)
    parser.add_argument("--agent-freq", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def _unwrap_env(env_obj):
    cur = env_obj
    seen = set()
    while id(cur) not in seen:
        seen.add(id(cur))
        nxt = getattr(cur, "env", None)
        if nxt is None:
            break
        cur = nxt
    return cur


def main() -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    args = parse_args()
    shaping = tasks.Shaping[args.shaping]
    task_type = TASK_MAP[args.task]

    _apply_jsbsim_runtime_compat()

    def _init():
        env = _make_env(shaping=shaping, agent_interaction_freq=args.agent_freq, task_type=task_type)
        env = _wrap_gymnasium_if_needed(env)
        return Monitor(env)

    vec_env = DummyVecEnv([_init])
    if args.vecnormalize_path and os.path.exists(args.vecnormalize_path):
        vec_env = VecNormalize.load(args.vecnormalize_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(args.model_path, env=vec_env)
    base_env = _unwrap_env(vec_env.envs[0])

    for ep in range(args.episodes):
        obs = vec_env.reset()
        # reset() creates/reinitializes sim; enable streaming afterwards.
        base_env.sim.enable_flightgear_output()
        base_env.sim.set_simulation_time_factor(1)
        done = False
        step = 0
        ep_reward = 0.0
        while not done and step < args.max_steps:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done_arr, _ = vec_env.step(action)
            done = bool(done_arr[0])
            r = float(reward[0])
            ep_reward += r
            if args.print_every > 0 and step % args.print_every == 0:
                print(f"ep={ep} step={step} reward={r:.3f}")
            step += 1
        print(f"episode {ep} total_reward={ep_reward:.3f} steps={step}")

    vec_env.close()


if __name__ == "__main__":
    main()
