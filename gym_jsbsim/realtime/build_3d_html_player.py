import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "models" / "eval_outputs" / "turn_heading_control_f16_v2_eval.csv"
DEFAULT_HTML = REPO_ROOT / "models" / "eval_outputs" / "turn_heading_control_f16_v2_3d_player.html"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a 3D HTML flight playback page from eval CSV.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--output-html", type=str, default=str(DEFAULT_HTML))
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dt-sec", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def _build_track(df: pd.DataFrame, dt_sec: float):
    x_m = [0.0]
    y_m = [0.0]
    z_m = [float(df.iloc[0]["position_h_sl_ft"]) * 0.3048]
    for i in range(1, len(df)):
        vn = float(df.iloc[i]["velocities_v_north_fps"]) * 0.3048
        ve = float(df.iloc[i]["velocities_v_east_fps"]) * 0.3048
        x_m.append(x_m[-1] + ve * dt_sec)
        y_m.append(y_m[-1] + vn * dt_sec)
        z_m.append(float(df.iloc[i]["position_h_sl_ft"]) * 0.3048)
    return x_m, y_m, z_m


def _html_with_data(data_json: str) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>3D Flight Playback</title>
  <style>
    :root {
      --bg-top: #e2edf5;
      --bg-bottom: #f8fbff;
      --panel: rgba(255, 255, 255, 0.78);
      --panel-border: rgba(148, 163, 184, 0.35);
      --text-main: #0f172a;
      --text-muted: #334155;
      --accent: #0f766e;
      --accent-2: #ea580c;
      --accent-3: #2563eb;
    }
    * {
      box-sizing: border-box;
    }
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background:
        radial-gradient(1200px 620px at 85% -15%, rgba(15, 118, 110, 0.16), transparent 62%),
        radial-gradient(860px 500px at 10% 12%, rgba(234, 88, 12, 0.14), transparent 58%),
        linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 65%);
      color: var(--text-main);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    #app {
      position: fixed;
      inset: 0;
    }
    .panel {
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      backdrop-filter: blur(8px);
      background: var(--panel);
      box-shadow: 0 12px 38px rgba(15, 23, 42, 0.12);
    }
    #title {
      position: fixed;
      top: 14px;
      right: 14px;
      padding: 10px 14px;
      font-weight: 700;
      letter-spacing: 0.4px;
      color: var(--text-main);
    }
    #hud {
      position: fixed;
      top: 14px;
      left: 14px;
      min-width: 280px;
      padding: 12px 14px;
      animation: fade-up 500ms ease-out;
    }
    #hud h3 {
      margin: 0 0 8px;
      font-size: 16px;
      color: var(--text-main);
    }
    .kv {
      display: grid;
      grid-template-columns: 116px auto;
      gap: 6px;
      font-size: 13px;
      line-height: 1.25;
      margin: 3px 0;
    }
    .k {
      color: var(--text-muted);
      font-weight: 600;
    }
    .v {
      color: var(--text-main);
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }
    #controls {
      position: fixed;
      right: 14px;
      bottom: 14px;
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 12px;
      animation: fade-up 700ms ease-out;
    }
    #btnPlay {
      background: linear-gradient(135deg, var(--accent), #0d9488);
      border: 1px solid rgba(15, 118, 110, 0.55);
      color: #f8fafc;
      border-radius: 10px;
      padding: 7px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    #btnPlay:hover {
      filter: brightness(1.03);
    }
    label {
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.15px;
    }
    select {
      background: #ffffff;
      border: 1px solid rgba(100, 116, 139, 0.45);
      color: var(--text-main);
      border-radius: 10px;
      padding: 7px 10px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
    }
    #timeline {
      width: 320px;
      accent-color: var(--accent);
      cursor: pointer;
    }
    #progress {
      min-width: 66px;
      text-align: right;
      color: var(--text-muted);
      font-size: 12px;
      font-weight: 700;
    }
    @keyframes fade-up {
      from {
        opacity: 0;
        transform: translateY(10px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    @media (max-width: 900px) {
      #hud {
        min-width: 240px;
        max-width: calc(100vw - 20px);
      }
      #controls {
        left: 10px;
        right: 10px;
        bottom: 10px;
        flex-wrap: wrap;
      }
      #timeline {
        width: 100%;
      }
      #progress {
        margin-left: auto;
      }
      #title {
        top: auto;
        bottom: 92px;
        right: 10px;
      }
    }
  </style>
  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
</head>
<body>
  <div id="app"></div>
  <div id="title" class="panel">3D Flight Playback</div>

  <div id="hud" class="panel">
    <h3>Aircraft State</h3>
    <div class="kv"><div class="k">Step</div><div class="v" id="s_step">-</div></div>
    <div class="kv"><div class="k">Heading</div><div class="v" id="s_hdg">-</div></div>
    <div class="kv"><div class="k">Roll</div><div class="v" id="s_roll">-</div></div>
    <div class="kv"><div class="k">Pitch</div><div class="v" id="s_pitch">-</div></div>
    <div class="kv"><div class="k">Altitude</div><div class="v" id="s_alt">-</div></div>
    <div class="kv"><div class="k">Track Error</div><div class="v" id="s_err">-</div></div>
    <div class="kv"><div class="k">Reward</div><div class="v" id="s_reward">-</div></div>
  </div>

  <div id="controls" class="panel">
    <button id="btnPlay">Pause</button>
    <label for="speedSel">Speed</label>
    <select id="speedSel">
      <option value="0.5">0.5x</option>
      <option value="1" selected>1x</option>
      <option value="2">2x</option>
      <option value="4">4x</option>
    </select>
    <input id="timeline" type="range" min="0" max="100" value="0" />
    <div id="progress">0 / 0</div>
  </div>

  <script>
    const DATA = __DATA_JSON__;
    const N = DATA.step.length;
    const app = document.getElementById("app");
    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0xd8e6f3, 500, 4300);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    app.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 12000);
    camera.position.set(0, -180, 80);

    const hemi = new THREE.HemisphereLight(0xf1f5f9, 0xb6c6d7, 1.2);
    scene.add(hemi);
    const dir = new THREE.DirectionalLight(0xffffff, 0.85);
    dir.position.set(220, -160, 300);
    scene.add(dir);

    const ground = new THREE.Mesh(
      new THREE.CircleGeometry(4200, 100),
      new THREE.MeshStandardMaterial({ color: 0xeaf2fa, roughness: 0.96, metalness: 0.03 })
    );
    ground.rotation.x = -Math.PI / 2;
    scene.add(ground);

    const grid = new THREE.GridHelper(5000, 110, 0x8fb0ca, 0xc7d8e7);
    grid.rotation.x = Math.PI / 2;
    scene.add(grid);

    const aircraft = new THREE.Group();
    const fuselage = new THREE.Mesh(
      new THREE.BoxGeometry(18, 64, 10),
      new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.28, roughness: 0.45 })
    );
    aircraft.add(fuselage);

    const wings = new THREE.Mesh(
      new THREE.BoxGeometry(74, 8, 2),
      new THREE.MeshStandardMaterial({ color: 0x0f766e, metalness: 0.18, roughness: 0.35 })
    );
    wings.position.y = -3;
    aircraft.add(wings);

    const tail = new THREE.Mesh(
      new THREE.BoxGeometry(14, 16, 2),
      new THREE.MeshStandardMaterial({ color: 0xea580c, metalness: 0.18, roughness: 0.35 })
    );
    tail.position.set(0, 26, 8);
    aircraft.add(tail);
    scene.add(aircraft);

    const pts = DATA.x.map((x, i) => new THREE.Vector3(x, DATA.y[i], DATA.z[i]));
    const trailGeo = new THREE.BufferGeometry().setFromPoints(pts);
    trailGeo.setDrawRange(0, 2);
    const trail = new THREE.Line(
      trailGeo,
      new THREE.LineBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.88 })
    );
    scene.add(trail);

    const velArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 40, 0x0f766e, 10, 6);
    scene.add(velArrow);

    const targetArrow = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 36, 0xea580c, 8, 5);
    scene.add(targetArrow);

    const el = (id) => document.getElementById(id);
    const timeline = el("timeline");
    timeline.max = String(Math.max(0, N - 1));

    let idx = 0;
    let playing = true;
    let speed = 1.0;
    let acc = 0;
    const dt = DATA.dt_sec;

    el("btnPlay").onclick = () => {
      playing = !playing;
      el("btnPlay").textContent = playing ? "Pause" : "Play";
    };

    el("speedSel").onchange = (e) => {
      speed = parseFloat(e.target.value);
    };

    timeline.oninput = (e) => {
      idx = Math.max(0, Math.min(N - 1, parseInt(e.target.value, 10)));
      renderFrame(idx);
    };

    function headingToYawRad(deg) {
      return deg * Math.PI / 180.0;
    }

    function renderFrame(i) {
      const x = DATA.x[i], y = DATA.y[i], z = DATA.z[i];
      aircraft.position.set(x, y, z);

      const roll = DATA.roll_deg[i] * Math.PI / 180.0;
      const pitch = DATA.pitch_deg[i] * Math.PI / 180.0;
      const yaw = headingToYawRad(DATA.heading_deg[i]);
      aircraft.rotation.set(pitch, 0, -yaw, "ZYX");
      aircraft.rotateY(Math.PI);
      aircraft.rotateZ(roll);

      trail.geometry.setDrawRange(0, Math.max(2, i + 1));

      const ve = DATA.ve_mps[i], vn = DATA.vn_mps[i];
      const v = new THREE.Vector3(ve, vn, 0.0);
      const vLen = Math.max(12, Math.min(85, Math.hypot(ve, vn) * 1.25));
      if (v.lengthSq() < 1e-6) v.set(1, 0, 0);
      v.normalize();
      velArrow.position.set(x, y, z + 6);
      velArrow.setDirection(v);
      velArrow.setLength(vLen, 10, 6);

      const tgt = headingToYawRad(DATA.target_track_deg[i]);
      const tDir = new THREE.Vector3(Math.sin(tgt), Math.cos(tgt), 0);
      targetArrow.position.set(x, y, z + 3);
      targetArrow.setDirection(tDir);
      targetArrow.setLength(34, 8, 5);

      const chaseDist = 150;
      const chaseH = 52;
      const forward = new THREE.Vector3(Math.sin(yaw), Math.cos(yaw), 0).normalize();
      const camPos = new THREE.Vector3(x, y, z + chaseH).addScaledVector(forward, -chaseDist);
      camera.position.lerp(camPos, 0.15);
      camera.lookAt(x, y, z + 10);

      el("s_step").textContent = String(DATA.step[i]);
      el("s_hdg").textContent = DATA.heading_deg[i].toFixed(1) + " deg";
      el("s_roll").textContent = DATA.roll_deg[i].toFixed(1) + " deg";
      el("s_pitch").textContent = DATA.pitch_deg[i].toFixed(1) + " deg";
      el("s_alt").textContent = DATA.alt_ft[i].toFixed(0) + " ft";
      el("s_err").textContent = DATA.track_err_deg[i].toFixed(2) + " deg";
      el("s_reward").textContent = DATA.reward[i].toFixed(3);
      el("progress").textContent = `${i + 1} / ${N}`;
      timeline.value = String(i);
    }

    let lastT = performance.now();
    function tick(now) {
      const delta = (now - lastT) / 1000.0;
      lastT = now;
      if (playing) {
        acc += delta * speed;
        while (acc >= dt && idx < N - 1) {
          idx += 1;
          acc -= dt;
        }
      }
      renderFrame(idx);
      renderer.render(scene, camera);
      requestAnimationFrame(tick);
    }

    renderFrame(0);
    requestAnimationFrame(tick);

    window.addEventListener("resize", () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  </script>
</body>
</html>
"""
    return template.replace("__DATA_JSON__", data_json)


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    ep_df = df[df["episode"] == args.episode].copy()
    if ep_df.empty:
        raise RuntimeError(f"No rows for episode={args.episode} in {csv_path}")
    ep_df = ep_df.iloc[:: max(1, args.stride)].reset_index(drop=True)

    x_m, y_m, z_m = _build_track(ep_df, args.dt_sec)
    payload = {
        "dt_sec": args.dt_sec,
        "step": ep_df["step"].astype(int).tolist(),
        "reward": ep_df["reward"].astype(float).tolist(),
        "heading_deg": ep_df["attitude_psi_deg"].astype(float).tolist(),
        "roll_deg": ep_df["roll_deg"].astype(float).tolist(),
        "pitch_deg": ep_df["pitch_deg"].astype(float).tolist(),
        "target_track_deg": ep_df["target_track_deg"].astype(float).tolist(),
        "track_err_deg": ep_df["error_track_error_deg"].astype(float).tolist(),
        "alt_ft": ep_df["position_h_sl_ft"].astype(float).tolist(),
        "ve_mps": (ep_df["velocities_v_east_fps"].astype(float) * 0.3048).tolist(),
        "vn_mps": (ep_df["velocities_v_north_fps"].astype(float) * 0.3048).tolist(),
        "x": x_m,
        "y": y_m,
        "z": z_m,
    }

    html = _html_with_data(json.dumps(payload, separators=(",", ":")))
    output_html.write_text(html, encoding="utf-8")
    print(f"Saved HTML: {output_html}")


if __name__ == "__main__":
    main()
