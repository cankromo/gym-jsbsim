import argparse
import os
import tempfile
import shutil
import signal
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "turn_heading_control_f16_v2" / "ppo_heading_control_f16.zip"
DEFAULT_VECNORM_PATH = REPO_ROOT / "models" / "turn_heading_control_f16_v2" / "vecnormalize.pkl"
DEFAULT_OUTPUT_MP4 = REPO_ROOT / "models" / "eval_outputs" / "turn_heading_control_f16_v2_flightgear_headless.mp4"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record FlightGear 3D view to MP4 in headless mode.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--vecnormalize-path", type=str, default=str(DEFAULT_VECNORM_PATH))
    parser.add_argument("--output-mp4", type=str, default=str(DEFAULT_OUTPUT_MP4))
    parser.add_argument("--task", type=str, default="turn", choices=["heading", "turn"])
    parser.add_argument("--shaping", type=str, default="EXTRA_SEQUENTIAL")
    parser.add_argument("--agent-freq", type=int, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--display", type=str, default=":99")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--fg-aircraft", type=str, default="c172p")
    parser.add_argument("--startup-wait-sec", type=float, default=18.0)
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


def _start(cmd, env, log_path):
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_f


def _tail(path: Path, n: int = 120) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="ignore").splitlines()
    return "\n".join(lines[-n:])


def _udp_5550_listening() -> bool:
    probe = subprocess.run(
        ["bash", "-lc", "ss -lunH | rg -q ':5550\\b'"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


def main() -> None:
    args = _parse_args()
    output_mp4 = Path(args.output_mp4).expanduser().resolve()
    output_mp4.parent.mkdir(parents=True, exist_ok=True)

    fgfs = shutil.which("fgfs") or "/usr/games/fgfs"
    for bin_name in (fgfs, "Xvfb", "ffmpeg", "dbus-run-session"):
        if shutil.which(bin_name) is None and not Path(bin_name).exists():
            raise RuntimeError(f"Missing required executable: {bin_name}")

    runtime_dir = Path("/tmp/runtime-root")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(0o700)
    display_lock = Path(f"/tmp/.X{args.display.lstrip(':')}-lock")
    if display_lock.exists():
        display_lock.unlink()

    env_vars = dict(os.environ)
    env_vars["DISPLAY"] = args.display
    env_vars["QT_QPA_PLATFORM"] = "xcb"
    env_vars["XDG_RUNTIME_DIR"] = str(runtime_dir)
    env_vars["DBUS_FATAL_WARNINGS"] = "0"
    if "/usr/games" not in env_vars.get("PATH", "").split(":"):
        env_vars["PATH"] = f"/usr/games:{env_vars.get('PATH', '')}"

    # Clean potentially stale processes from interrupted runs.
    subprocess.run(["pkill", "-f", "/usr/games/fgfs"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", f"Xvfb {args.display}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    xvfb_log = Path("/tmp/gymjsbsim_xvfb.log")
    fg_log = Path("/tmp/gymjsbsim_fgfs.log")
    ffmpeg_log = Path("/tmp/gymjsbsim_ffmpeg.log")

    xvfb_cmd = [
        "Xvfb",
        args.display,
        "-screen",
        "0",
        f"{args.width}x{args.height}x24",
        "-ac",
        "+extension",
        "GLX",
        "+render",
    ]
    fg_cmd = [
        "dbus-run-session",
        "--",
        fgfs,
        f"--aircraft={args.fg_aircraft}",
        "--native-fdm=socket,in,60,localhost,5550,udp",
        "--fdm=external",
        "--disable-ai-traffic",
        "--disable-real-weather-fetch",
        "--timeofday=dusk",
    ]
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "x11grab",
        "-video_size",
        f"{args.width}x{args.height}",
        "-framerate",
        str(args.fps),
        "-i",
        f"{args.display}.0",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_mp4),
    ]

    xvfb_proc = fg_proc = ffmpeg_proc = None
    xvfb_f = fg_f = ffmpeg_f = None
    fg_home_dir = Path(tempfile.mkdtemp(prefix="fg-home-", dir="/tmp"))

    try:
        xvfb_proc, xvfb_f = _start(xvfb_cmd, env_vars, xvfb_log)
        time.sleep(2.0)
        if xvfb_proc.poll() is not None:
            raise RuntimeError(f"Xvfb failed:\n{_tail(xvfb_log)}")

        fg_env = dict(env_vars)
        fg_env["HOME"] = str(fg_home_dir)
        fg_proc, fg_f = _start(fg_cmd, fg_env, fg_log)
        time.sleep(3.0)
        if fg_proc.poll() is not None:
            raise RuntimeError(f"FlightGear failed at startup:\n{_tail(fg_log)}")

        ffmpeg_proc, ffmpeg_f = _start(ffmpeg_cmd, env_vars, ffmpeg_log)

        # Import gym-jsbsim stack only after FlightGear is already running.
        from gym_jsbsim import tasks
        from gym_jsbsim.tests.sb3_heading_control_f16 import (
            TASK_MAP,
            _apply_jsbsim_runtime_compat,
            _make_env,
            _wrap_gymnasium_if_needed,
        )
        shaping = tasks.Shaping[args.shaping]
        task_type = TASK_MAP[args.task]
        _apply_jsbsim_runtime_compat()

        # Import heavy ML stack after FlightGear process is fully spawned to avoid
        # fork/thread interaction issues in child GUI process.
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
        if not hasattr(base_env, "sim"):
            raise RuntimeError("Could not unwrap to base JsbSimEnv.")

        # Give FlightGear time to open and bind its UDP input socket.
        deadline = time.time() + args.startup_wait_sec
        while time.time() < deadline:
            if fg_proc.poll() is not None:
                raise RuntimeError(f"FlightGear exited before rollout:\n{_tail(fg_log)}")
            if _udp_5550_listening():
                break
            time.sleep(0.5)
        if fg_proc.poll() is not None:
            raise RuntimeError(f"FlightGear exited before rollout:\n{_tail(fg_log)}")
        if not _udp_5550_listening():
            raise RuntimeError(
                "FlightGear did not open UDP port 5550 before rollout.\n"
                f"Recent FlightGear logs:\n{_tail(fg_log)}"
            )

        for ep in range(args.episodes):
            obs = vec_env.reset()
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
    finally:
        if ffmpeg_proc and ffmpeg_proc.poll() is None:
            ffmpeg_proc.send_signal(signal.SIGINT)
        for proc in (fg_proc, xvfb_proc):
            if proc and proc.poll() is None:
                proc.terminate()

        for proc in (ffmpeg_proc, fg_proc, xvfb_proc):
            if proc:
                try:
                    proc.wait(timeout=12)
                except Exception:
                    proc.kill()

        for fh in (xvfb_f, fg_f, ffmpeg_f):
            if fh:
                fh.close()
        shutil.rmtree(fg_home_dir, ignore_errors=True)

    if not output_mp4.exists() or output_mp4.stat().st_size < 10000:
        raise RuntimeError(
            f"MP4 not produced correctly: {output_mp4}\n"
            f"FlightGear log tail:\n{_tail(fg_log)}\n\n"
            f"ffmpeg log tail:\n{_tail(ffmpeg_log)}"
        )

    print(f"Saved MP4: {output_mp4}")


if __name__ == "__main__":
    main()
