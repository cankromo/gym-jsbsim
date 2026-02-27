import argparse
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from pyngrok import ngrok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Dash realtime viewer in Colab.")
    parser.add_argument("--ngrok-token", type=str, required=True)
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/content/gym-jsbsim/models/heading_control_f16/eval_metrics.csv",
    )
    parser.add_argument("--dash-port", type=int, default=8050)
    return parser.parse_args()


def _start(cmd: list[str], cwd: Path, env: dict) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _print_url(name: str, url: str) -> None:
    print(f"{name}: {url}", flush=True)


def _poll_output(proc: subprocess.Popen, label: str, max_lines: int = 3) -> None:
    if not proc.stdout:
        return
    for _ in range(max_lines):
        line = proc.stdout.readline()
        if not line:
            break
        print(f"[{label}] {line.rstrip()}", flush=True)


def _drain_output(proc: subprocess.Popen, label: str) -> str:
    if not proc.stdout:
        return ""
    lines = []
    while True:
        line = proc.stdout.readline()
        if not line:
            break
        lines.append(line.rstrip())
    for line in lines[-50:]:
        print(f"[{label}] {line}", flush=True)
    return "\n".join(lines[-50:])


def _is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main() -> None:
    args = parse_args()
    realtime_dir = Path(__file__).resolve().parent
    package_dir = realtime_dir.parent
    repo_root = package_dir.parent
    if _is_port_in_use(args.dash_port):
        raise RuntimeError(
            f"Port {args.dash_port} is already in use. "
            "Choose another with --dash-port or stop the existing process."
        )

    ngrok.set_auth_token(args.ngrok_token)

    dash_public = ngrok.connect(args.dash_port, "http").public_url
    dash_cmd = [
        sys.executable,
        str(realtime_dir / "dash_live.py"),
        "--csv-path",
        args.csv_path,
        "--port",
        str(args.dash_port),
    ]

    child_env = dict(os.environ)
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    prefix = str(repo_root)
    child_env["PYTHONPATH"] = f"{prefix}:{existing_pythonpath}" if existing_pythonpath else prefix

    dash_proc = _start(dash_cmd, cwd=repo_root, env=child_env)

    _print_url("Dash URL", dash_public)
    print("Press Ctrl+C to stop service.", flush=True)

    try:
        while True:
            if dash_proc.poll() is not None:
                tail = _drain_output(dash_proc, "dash")
                raise RuntimeError(
                    f"dash_live exited with code {dash_proc.returncode}\n"
                    f"Recent dash logs:\n{tail}"
                )
            _poll_output(dash_proc, "dash")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Stopping services...", flush=True)
    finally:
        if dash_proc.poll() is None:
            dash_proc.terminate()
        try:
            dash_proc.wait(timeout=5)
        except Exception:
            dash_proc.kill()


if __name__ == "__main__":
    main()
