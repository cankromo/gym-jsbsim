import argparse
import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "models" / "eval_outputs" / "dogfight_cycle_eval.csv"
DEFAULT_HTML = REPO_ROOT / "models" / "eval_outputs" / "dogfight_cockpit_player.html"
FIRE_AZIMUTH_DEG = 60.0
FIRE_ELEVATION_DEG = 20.0
FIRE_SOLUTION_DEG = 6.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a standalone first-person dogfight cockpit HTML player from eval CSV.")
    parser.add_argument("--csv-path", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--output-html", type=str, default=str(DEFAULT_HTML))
    parser.add_argument("--dt-sec", type=float, default=0.2)
    parser.add_argument("--stride", type=int, default=1)
    return parser.parse_args()


def _series(df: pd.DataFrame, key: str, default: float = 0.0) -> list[float]:
    if key in df.columns:
        return df[key].astype(float).tolist()
    return [default] * len(df)


def _build_payload(df: pd.DataFrame, stride: int) -> dict:
    work = df.copy()
    if "plane_id" not in work.columns:
        work["plane_id"] = "plane_0"
    if "episode" not in work.columns:
        work["episode"] = 0
    work = work.iloc[:: max(1, stride)].reset_index(drop=True)

    episodes_payload = {}
    for episode, ep_df in work.groupby("episode", sort=True):
        plane_payload = {}
        for plane_id, plane_df in ep_df.groupby("plane_id", sort=True):
            plane_df = plane_df.sort_values("step").reset_index(drop=True)
            plane_payload[str(plane_id)] = {
                "step": plane_df["step"].astype(int).tolist(),
                "reward": _series(plane_df, "reward"),
                "heading_deg": _series(plane_df, "attitude_psi_deg"),
                "roll_deg": _series(plane_df, "roll_deg"),
                "pitch_deg": _series(plane_df, "pitch_deg"),
                "target_track_deg": _series(plane_df, "target_track_deg"),
                "track_err_deg": _series(plane_df, "error_track_error_deg"),
                "alt_ft": _series(plane_df, "position_h_sl_ft"),
                "speed_kts": [v * 0.592484 for v in _series(plane_df, "velocities_u_fps")],
                "range_m": _series(plane_df, "range_m", 9999.0),
                "bearing_error_deg": _series(plane_df, "bearing_error_deg"),
                "elevation_error_deg": _series(plane_df, "elevation_error_deg"),
                "heading_difference_deg": _series(plane_df, "heading_difference_deg"),
                "target_roll_deg": _series(plane_df, "target_roll_deg"),
                "current_roll_deg": _series(plane_df, "current_roll_deg"),
                "roll_error_deg": _series(plane_df, "roll_error_deg"),
                "throttle_cmd": _series(plane_df, "fcs_throttle_cmd_norm", 0.8),
                "aileron_cmd": _series(plane_df, "fcs_aileron_cmd_norm"),
                "elevator_cmd": _series(plane_df, "fcs_elevator_cmd_norm"),
                "rudder_cmd": _series(plane_df, "fcs_rudder_cmd_norm"),
                "done": [bool(v) for v in plane_df.get("done", pd.Series([False] * len(plane_df)))],
                "opponent_plane_id": plane_df.get("opponent_plane_id", pd.Series([""] * len(plane_df))).astype(str).tolist(),
            }
        episodes_payload[str(int(episode))] = plane_payload
    return {"episodes": episodes_payload}


def _html_with_data(data_json: str, dt_sec: float) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Dogfight Cockpit Playback</title>
  <style>
    :root {
      --bg0: #05070d;
      --bg1: #08101f;
      --panel: rgba(10, 16, 28, 0.84);
      --panel-2: rgba(14, 23, 37, 0.88);
      --line: rgba(126, 163, 212, 0.28);
      --text: #e8f0ff;
      --muted: #97aac7;
      --hud: #7df9ff;
      --warn: #ffb703;
      --danger: #ff5e5b;
      --good: #52ffa8;
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      color: var(--text);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background:
        radial-gradient(1200px 700px at 10% 0%, rgba(125, 249, 255, 0.06), transparent 55%),
        radial-gradient(900px 560px at 90% 100%, rgba(255, 183, 3, 0.08), transparent 55%),
        linear-gradient(180deg, var(--bg0), var(--bg1));
    }
    #root {
      position: fixed;
      inset: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 10px;
      padding: 10px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      backdrop-filter: blur(8px);
      box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
    }
    #topbar {
      display: grid;
      grid-template-columns: 1fr auto auto auto auto auto;
      gap: 10px;
      align-items: center;
      padding: 12px;
    }
    #title {
      font-size: 20px;
      font-weight: 760;
      letter-spacing: 0.2px;
    }
    #subtitle {
      color: var(--muted);
      font-size: 12px;
      margin-top: 4px;
    }
    .ctrl {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 120px;
      color: var(--muted);
      font-size: 12px;
    }
    .ctrl label { font-weight: 700; }
    select, button, input[type="range"] {
      min-height: 36px;
      border-radius: 10px;
      border: 1px solid rgba(96, 124, 168, 0.5);
    }
    select, button {
      background: #0a1321;
      color: var(--text);
      padding: 8px 10px;
    }
    button {
      cursor: pointer;
      font-weight: 700;
    }
    #btnPlay {
      background: linear-gradient(135deg, #00b4d8, #0077b6);
      color: white;
    }
    #stage {
      position: relative;
      overflow: hidden;
      min-height: 540px;
      background:
        radial-gradient(1000px 600px at 50% -20%, rgba(125, 249, 255, 0.06), transparent 60%),
        linear-gradient(180deg, #08121f, #04080f);
    }
    #view {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      display: block;
    }
    #frame {
      position: absolute;
      inset: 18px;
      border: 1px solid rgba(125, 249, 255, 0.14);
      border-radius: 28px;
      pointer-events: none;
      box-shadow: inset 0 0 120px rgba(0, 0, 0, 0.42);
    }
    #hudOverlay {
      position: absolute;
      inset: 0;
      pointer-events: none;
    }
    .hudBox {
      position: absolute;
      background: rgba(2, 8, 18, 0.56);
      border: 1px solid rgba(125, 249, 255, 0.22);
      border-radius: 14px;
      padding: 10px 12px;
      color: var(--hud);
      font-family: "IBM Plex Mono", "Fira Code", monospace;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
    }
    #leftStats { left: 18px; top: 18px; min-width: 190px; }
    #rightStats { right: 18px; top: 18px; min-width: 190px; text-align: right; }
    #bottomCenter {
      left: 50%;
      bottom: 18px;
      transform: translateX(-50%);
      min-width: 420px;
      text-align: center;
    }
    .kv {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      font-size: 13px;
      line-height: 1.35;
      margin: 2px 0;
    }
    .kv .k { color: var(--muted); }
    .kv .v { color: var(--text); font-weight: 700; }
    #flightPathMarker {
      position: absolute;
      width: 34px;
      height: 34px;
      border: 2px solid var(--hud);
      border-radius: 50%;
      transform: translate(-50%, -50%);
      box-shadow: 0 0 16px rgba(125, 249, 255, 0.35);
    }
    #flightPathMarker::before, #flightPathMarker::after {
      content: "";
      position: absolute;
      top: 50%;
      width: 18px;
      border-top: 2px solid var(--hud);
    }
    #flightPathMarker::before { right: 100%; margin-right: 4px; }
    #flightPathMarker::after { left: 100%; margin-left: 4px; }
    #targetBox {
      position: absolute;
      width: 46px;
      height: 46px;
      transform: translate(-50%, -50%);
      border: 2px solid var(--warn);
      border-radius: 8px;
      box-shadow: 0 0 18px rgba(255, 183, 3, 0.32);
      display: none;
    }
    #targetLabel {
      position: absolute;
      transform: translate(-50%, calc(-100% - 10px));
      color: var(--warn);
      font-family: "IBM Plex Mono", monospace;
      font-size: 12px;
      font-weight: 700;
      text-shadow: 0 0 10px rgba(255, 183, 3, 0.4);
      display: none;
      white-space: nowrap;
    }
    #gunsight {
      position: absolute;
      left: 50%;
      top: 50%;
      width: 54px;
      height: 54px;
      transform: translate(-50%, -50%);
      border: 2px solid rgba(125, 249, 255, 0.75);
      border-radius: 50%;
      box-shadow: 0 0 22px rgba(125, 249, 255, 0.22);
    }
    #gunsight::before, #gunsight::after {
      content: "";
      position: absolute;
      left: 50%;
      top: 50%;
      background: rgba(125, 249, 255, 0.8);
      transform: translate(-50%, -50%);
    }
    #gunsight::before { width: 2px; height: 66px; }
    #gunsight::after { width: 66px; height: 2px; }
    #fireCue {
      position: absolute;
      left: 50%;
      top: 16%;
      transform: translateX(-50%);
      padding: 8px 14px;
      border-radius: 999px;
      font-family: "IBM Plex Mono", monospace;
      font-weight: 800;
      letter-spacing: 1.5px;
      background: rgba(82, 255, 168, 0.14);
      border: 1px solid rgba(82, 255, 168, 0.45);
      color: var(--good);
      display: none;
    }
    #warningCue {
      position: absolute;
      left: 50%;
      top: 23%;
      transform: translateX(-50%);
      padding: 7px 12px;
      border-radius: 999px;
      font-family: "IBM Plex Mono", monospace;
      font-weight: 800;
      background: rgba(255, 94, 91, 0.14);
      border: 1px solid rgba(255, 94, 91, 0.4);
      color: var(--danger);
      display: none;
    }
    #timelineBar {
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: center;
      padding: 10px 12px;
    }
    #timeline {
      width: 100%;
      accent-color: #7df9ff;
      min-height: 0;
    }
    #progress {
      min-width: 110px;
      text-align: right;
      color: var(--muted);
      font-family: "IBM Plex Mono", monospace;
    }
    @media (max-width: 980px) {
      #topbar {
        grid-template-columns: 1fr 1fr;
      }
      #bottomCenter {
        min-width: 0;
        width: calc(100% - 28px);
      }
      #leftStats, #rightStats {
        min-width: 0;
        width: 180px;
      }
    }
  </style>
  <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
</head>
<body>
  <div id="root">
    <div id="topbar" class="panel">
      <div>
        <div id="title">Dogfight First-Person Cockpit</div>
        <div id="subtitle">Standalone playback built from eval CSV. Choose episode and cockpit aircraft.</div>
      </div>
      <div class="ctrl">
        <label for="episodeSel">Episode</label>
        <select id="episodeSel"></select>
      </div>
      <div class="ctrl">
        <label for="planeSel">Cockpit</label>
        <select id="planeSel"></select>
      </div>
      <div class="ctrl">
        <label for="speedSel">Speed</label>
        <select id="speedSel">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1x</option>
          <option value="2">2x</option>
          <option value="4">4x</option>
        </select>
      </div>
      <div class="ctrl">
        <label>&nbsp;</label>
        <button id="btnPlay">Pause</button>
      </div>
      <div class="ctrl">
        <label>Status</label>
        <div id="status" style="min-height:36px;display:flex;align-items:center;color:var(--muted)">Ready</div>
      </div>
    </div>
    <div id="stage" class="panel">
      <canvas id="view"></canvas>
      <div id="frame"></div>
      <div id="hudOverlay">
        <div id="leftStats" class="hudBox"></div>
        <div id="rightStats" class="hudBox"></div>
        <div id="bottomCenter" class="hudBox"></div>
        <div id="flightPathMarker"></div>
        <div id="gunsight"></div>
        <div id="targetBox"></div>
        <div id="targetLabel">TARGET</div>
        <div id="fireCue">FIRE CUE</div>
        <div id="warningCue">THREAT</div>
      </div>
    </div>
    <div id="timelineBar" class="panel">
      <div>Timeline</div>
      <input id="timeline" type="range" min="0" max="0" value="0" />
      <div id="progress">0 / 0</div>
    </div>
  </div>

  <script>
    const DATA = __DATA_JSON__;
    const DT_SEC = __DT_SEC__;
    const FIRE_RANGE_M = 1200.0;
    const FIRE_AZIMUTH_DEG = __FIRE_AZIMUTH_DEG__;
    const FIRE_SOLUTION_DEG = __FIRE_SOLUTION_DEG__;
    const FIRE_ELEVATION_DEG = __FIRE_ELEVATION_DEG__;
    const HUD_H_FOV_DEG = FIRE_AZIMUTH_DEG * 2.0;
    const HUD_V_FOV_DEG = FIRE_ELEVATION_DEG * 2.0;
    const canvas = document.getElementById("view");
    const ctx = canvas.getContext("2d");
    const stage = document.getElementById("stage");
    const state = {
      episode: null,
      plane: null,
      idx: 0,
      playing: true,
      speed: 1.0,
      acc: 0.0,
      lastTs: performance.now(),
    };

    const $ = (id) => document.getElementById(id);
    const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
    const lerp = (a, b, t) => a + (b - a) * t;

    let rendererMode = "fallback";
    let renderer = null;
    let scene = null;
    let camera = null;
    let targetJet = null;
    let contrail = null;
    let sky = null;
    let horizonRing = null;
    let ground = null;
    let targetWorld = null;

    function initThree() {
      if (typeof THREE === "undefined") {
        return false;
      }
      rendererMode = "three";
      renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      scene = new THREE.Scene();
      scene.fog = new THREE.Fog(0x7aa9d6, 900, 9000);
      scene.background = new THREE.Color(0x7aa9d6);
      camera = new THREE.PerspectiveCamera(HUD_H_FOV_DEG, 1, 0.1, 20000);
      targetWorld = new THREE.Vector3(0, 0, -1200);

      const hemi = new THREE.HemisphereLight(0xd9ecff, 0x202533, 1.35);
      scene.add(hemi);
      const sun = new THREE.DirectionalLight(0xffffff, 1.1);
      sun.position.set(1400, 2200, 900);
      scene.add(sun);

      sky = new THREE.Mesh(
        new THREE.SphereGeometry(12000, 48, 24),
        new THREE.MeshBasicMaterial({ color: 0x7aa9d6, side: THREE.BackSide })
      );
      scene.add(sky);

      ground = new THREE.Mesh(
        new THREE.PlaneGeometry(18000, 18000, 100, 100),
        new THREE.MeshStandardMaterial({ color: 0x334438, roughness: 1.0, metalness: 0.0 })
      );
      ground.rotation.x = -Math.PI / 2;
      scene.add(ground);

      const grid = new THREE.GridHelper(18000, 180, 0x6c8aa6, 0x415265);
      grid.position.y = 0.25;
      scene.add(grid);

      horizonRing = new THREE.Mesh(
        new THREE.TorusGeometry(3500, 14, 12, 120),
        new THREE.MeshBasicMaterial({ color: 0xeaf4ff, transparent: true, opacity: 0.18 })
      );
      horizonRing.rotation.x = Math.PI / 2;
      scene.add(horizonRing);

      targetJet = new THREE.Group();
      const body = new THREE.Mesh(
        new THREE.CylinderGeometry(2.4, 5.4, 34, 12),
        new THREE.MeshStandardMaterial({ color: 0xdee7f2, metalness: 0.32, roughness: 0.48 })
      );
      body.rotation.z = Math.PI / 2;
      targetJet.add(body);
      const nose = new THREE.Mesh(
        new THREE.ConeGeometry(2.6, 8, 10),
        new THREE.MeshStandardMaterial({ color: 0xf8fafc, metalness: 0.15, roughness: 0.4 })
      );
      nose.rotation.z = -Math.PI / 2;
      nose.position.x = 21;
      targetJet.add(nose);
      const wings = new THREE.Mesh(
        new THREE.BoxGeometry(12, 1.4, 54),
        new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.28, roughness: 0.55 })
      );
      targetJet.add(wings);
      const tail = new THREE.Mesh(
        new THREE.BoxGeometry(8, 10, 1.2),
        new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.22, roughness: 0.55 })
      );
      tail.position.set(-12, 6, 0);
      targetJet.add(tail);
      scene.add(targetJet);

      contrail = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
        new THREE.LineBasicMaterial({ color: 0x9dd9ff, transparent: true, opacity: 0.45 })
      );
      scene.add(contrail);

      const cockpit = new THREE.Group();
      camera.add(cockpit);
      scene.add(camera);

      const panelMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.82, metalness: 0.08 });
      const panel = new THREE.Mesh(new THREE.BoxGeometry(1.9, 0.28, 0.72), panelMat);
      panel.position.set(0, -0.95, -1.55);
      cockpit.add(panel);
      const canopy = new THREE.LineSegments(
        new THREE.EdgesGeometry(new THREE.BoxGeometry(2.6, 1.6, 2.2)),
        new THREE.LineBasicMaterial({ color: 0x7df9ff, transparent: true, opacity: 0.18 })
      );
      canopy.position.set(0, 0.1, -0.7);
      cockpit.add(canopy);
      const hudGlass = new THREE.Mesh(
        new THREE.PlaneGeometry(0.95, 0.7),
        new THREE.MeshBasicMaterial({ color: 0x7df9ff, transparent: true, opacity: 0.06, side: THREE.DoubleSide })
      );
      hudGlass.position.set(0, 0.08, -1.06);
      cockpit.add(hudGlass);
      return true;
    }

    function episodes() {
      return Object.keys(DATA.episodes).sort((a, b) => Number(a) - Number(b));
    }

    function planesForEpisode(ep) {
      return Object.keys(DATA.episodes[ep] || {}).sort();
    }

    function series() {
      return (((DATA.episodes || {})[state.episode] || {})[state.plane]) || null;
    }

    function opponentPlane() {
      const s = series();
      if (!s || !s.opponent_plane_id || !s.opponent_plane_id.length) return "";
      return s.opponent_plane_id[Math.min(state.idx, s.opponent_plane_id.length - 1)] || "";
    }

    function fillSelectors() {
      const epSel = $("episodeSel");
      epSel.innerHTML = "";
      episodes().forEach((ep) => {
        const opt = document.createElement("option");
        opt.value = ep;
        opt.textContent = ep;
        epSel.appendChild(opt);
      });
      state.episode = episodes()[0];
      epSel.value = state.episode;
      refillPlanes();
    }

    function refillPlanes() {
      const planeSel = $("planeSel");
      planeSel.innerHTML = "";
      planesForEpisode(state.episode).forEach((plane) => {
        const opt = document.createElement("option");
        opt.value = plane;
        opt.textContent = plane;
        planeSel.appendChild(opt);
      });
      state.plane = planesForEpisode(state.episode)[0];
      planeSel.value = state.plane;
      state.idx = 0;
      syncTimeline();
      updateStatus();
    }

    function syncTimeline() {
      const s = series();
      const n = s ? s.step.length : 0;
      $("timeline").max = String(Math.max(0, n - 1));
      $("timeline").value = String(clamp(state.idx, 0, Math.max(0, n - 1)));
      $("progress").textContent = `${Math.min(state.idx + 1, Math.max(1, n))} / ${n}`;
    }

    function updateStatus() {
      const opp = opponentPlane() || "none";
      $("status").textContent = `Episode ${state.episode} | cockpit ${state.plane} | target ${opp}`;
    }

    function resize() {
      const width = stage.clientWidth;
      const height = stage.clientHeight;
      canvas.width = Math.floor(width * Math.min(window.devicePixelRatio, 2));
      canvas.height = Math.floor(height * Math.min(window.devicePixelRatio, 2));
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      if (rendererMode === "three") {
        renderer.setSize(width, height, false);
        camera.aspect = width / Math.max(1, height);
        camera.fov = HUD_H_FOV_DEG;
        camera.updateProjectionMatrix();
      }
    }

    function drawVelocityVector(cx, cy, rollDeg, pitchDeg) {
      const fpm = $("flightPathMarker");
      const x = cx + clamp(rollDeg * 3.0, -90, 90);
      const y = cy + clamp(-pitchDeg * 5.2, -110, 110);
      fpm.style.left = `${x}px`;
      fpm.style.top = `${y}px`;
    }

    function updateHud(s, i) {
      const heading = s.heading_deg[i];
      const roll = s.roll_deg[i];
      const pitch = s.pitch_deg[i];
      const targetHeading = s.target_track_deg[i];
      const bearing = s.bearing_error_deg[i];
      const elevation = s.elevation_error_deg[i];
      const range = s.range_m[i];
      const fireCue = range <= FIRE_RANGE_M && Math.abs(bearing) <= FIRE_SOLUTION_DEG && Math.abs(elevation) <= FIRE_SOLUTION_DEG;
      const threat = Math.abs(bearing) > 140 && range < 1600;

      $("leftStats").innerHTML = `
        <div class="kv"><div class="k">STEP</div><div class="v">${s.step[i]}</div></div>
        <div class="kv"><div class="k">ALT</div><div class="v">${s.alt_ft[i].toFixed(0)} ft</div></div>
        <div class="kv"><div class="k">SPD</div><div class="v">${s.speed_kts[i].toFixed(0)} kt</div></div>
        <div class="kv"><div class="k">HDG</div><div class="v">${heading.toFixed(1)} deg</div></div>
        <div class="kv"><div class="k">ROLL</div><div class="v">${roll.toFixed(1)} deg</div></div>
        <div class="kv"><div class="k">PITCH</div><div class="v">${pitch.toFixed(1)} deg</div></div>
      `;
      $("rightStats").innerHTML = `
        <div class="kv"><div class="k">TARGET</div><div class="v">${targetHeading.toFixed(1)} deg</div></div>
        <div class="kv"><div class="k">TRACK ERR</div><div class="v">${s.track_err_deg[i].toFixed(2)} deg</div></div>
        <div class="kv"><div class="k">RANGE</div><div class="v">${range.toFixed(0)} m</div></div>
        <div class="kv"><div class="k">BEARING</div><div class="v">${bearing.toFixed(1)} deg</div></div>
        <div class="kv"><div class="k">ELEV</div><div class="v">${elevation.toFixed(1)} deg</div></div>
        <div class="kv"><div class="k">REWARD</div><div class="v">${s.reward[i].toFixed(3)}</div></div>
      `;
      $("bottomCenter").innerHTML = `
        <div style="font-size:13px;color:var(--muted);margin-bottom:4px">Control Inputs and Geometry</div>
        <div class="kv"><div class="k">AIL / EL / RUD</div><div class="v">${s.aileron_cmd[i].toFixed(2)} / ${s.elevator_cmd[i].toFixed(2)} / ${s.rudder_cmd[i].toFixed(2)}</div></div>
        <div class="kv"><div class="k">TARGET ROLL / ERR</div><div class="v">${s.target_roll_deg[i].toFixed(1)} / ${s.roll_error_deg[i].toFixed(1)} deg</div></div>
        <div class="kv"><div class="k">ASPECT DIFF</div><div class="v">${s.heading_difference_deg[i].toFixed(1)} deg</div></div>
      `;
      $("fireCue").style.display = fireCue ? "block" : "none";
      $("warningCue").style.display = threat ? "block" : "none";
      updateStatus();
    }

    function updateTargetCue(s, i) {
      const box = $("targetBox");
      const label = $("targetLabel");
      const bearing = s.bearing_error_deg[i];
      const elevation = s.elevation_error_deg[i];
      const range = s.range_m[i];
      let cx = 0;
      let cy = 0;
      let visible = isFinite(bearing) && isFinite(elevation) && range < 8000;
      if (rendererMode === "three") {
        const projected = targetJet.position.clone().project(camera);
        visible = visible && projected.z > -1 && projected.z < 1;
        if (visible) {
          const rect = stage.getBoundingClientRect();
          cx = (projected.x * 0.5 + 0.5) * rect.width;
          cy = (-projected.y * 0.5 + 0.5) * rect.height;
        }
      } else {
        const nx = clamp(bearing / FIRE_AZIMUTH_DEG, -1, 1);
        const ny = clamp(-elevation / FIRE_ELEVATION_DEG, -1, 1);
        const rect = stage.getBoundingClientRect();
        cx = rect.width * 0.5 + nx * rect.width * 0.32;
        cy = rect.height * 0.5 + ny * rect.height * 0.32;
      }
      if (!visible) {
        box.style.display = "none";
        label.style.display = "none";
        return;
      }
      box.style.display = "block";
      label.style.display = "block";
      box.style.left = `${cx}px`;
      box.style.top = `${cy}px`;
      label.style.left = `${cx}px`;
      label.style.top = `${cy}px`;
      label.textContent = `${opponentPlane() || "TARGET"} | ${range.toFixed(0)} m`;
      const fireLike = range <= FIRE_RANGE_M && Math.abs(bearing) <= FIRE_SOLUTION_DEG && Math.abs(elevation) <= FIRE_ELEVATION_DEG;
      box.style.borderColor = fireLike ? "var(--good)" : "var(--warn)";
      box.style.boxShadow = fireLike ? "0 0 18px rgba(82,255,168,0.32)" : "0 0 18px rgba(255,183,3,0.32)";
      label.style.color = fireLike ? "var(--good)" : "var(--warn)";
      label.textContent += ` | ${bearing.toFixed(1)} / ${elevation.toFixed(1)} deg`;
    }

    function updateScene(s, i) {
      if (rendererMode !== "three") {
        return;
      }
      const altitudeM = s.alt_ft[i] * 0.3048;
      const pitchRad = s.pitch_deg[i] * Math.PI / 180.0;
      const rollRad = s.roll_deg[i] * Math.PI / 180.0;
      const yawRad = s.heading_deg[i] * Math.PI / 180.0;
      camera.position.set(0, altitudeM, 0);
      camera.rotation.order = "ZYX";
      camera.rotation.set(pitchRad, 0, -yawRad);
      camera.rotateZ(rollRad);

      const bearingRad = s.bearing_error_deg[i] * Math.PI / 180.0;
      const elevationRad = s.elevation_error_deg[i] * Math.PI / 180.0;
      const range = Math.max(60, s.range_m[i]);
      const forward = Math.cos(elevationRad) * Math.cos(bearingRad) * range;
      const right = Math.cos(elevationRad) * Math.sin(bearingRad) * range;
      const up = Math.sin(elevationRad) * range;
      targetWorld.set(right, altitudeM + up, -forward);
      targetJet.position.copy(targetWorld);
      targetJet.rotation.set(0.08 * Math.sin(i * 0.07), -yawRad + s.heading_difference_deg[i] * Math.PI / 180.0, 0.1 * Math.sin(i * 0.11));

      const trailPoints = [targetWorld.clone(), targetWorld.clone().add(new THREE.Vector3(0, 0, range * 0.18))];
      contrail.geometry.dispose();
      contrail.geometry = new THREE.BufferGeometry().setFromPoints(trailPoints);

      sky.position.set(0, altitudeM * 0.15, 0);
      horizonRing.position.set(0, altitudeM * 0.35, 0);
      ground.position.y = 0;
    }

    function renderFallbackScene(s, idx) {
      const w = canvas.width;
      const h = canvas.height;
      const cx = w * 0.5;
      const cy = h * 0.5;
      ctx.clearRect(0, 0, w, h);

      const rollRad = s.roll_deg[idx] * Math.PI / 180.0;
      const pitchPx = clamp(s.pitch_deg[idx] * 4.8, -h * 0.22, h * 0.22);
      ctx.save();
      ctx.translate(cx, cy + pitchPx);
      ctx.rotate(-rollRad);
      ctx.fillStyle = "#6da6da";
      ctx.fillRect(-w, -h * 2, w * 2, h * 2);
      ctx.fillStyle = "#52694a";
      ctx.fillRect(-w, 0, w * 2, h * 2);
      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(-w, 0);
      ctx.lineTo(w, 0);
      ctx.stroke();
      ctx.strokeStyle = "rgba(230,245,255,0.25)";
      ctx.lineWidth = 1;
      for (let deg = -30; deg <= 30; deg += 5) {
        if (deg === 0) continue;
        const y = -deg * 6.6;
        const width = deg % 10 === 0 ? 90 : 48;
        ctx.beginPath();
        ctx.moveTo(-width, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
      ctx.restore();

      const grad = ctx.createLinearGradient(0, h * 0.72, 0, h);
      grad.addColorStop(0, "rgba(6, 12, 22, 0.16)");
      grad.addColorStop(1, "rgba(6, 12, 22, 0.95)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(0, h);
      ctx.lineTo(w * 0.18, h * 0.74);
      ctx.lineTo(w * 0.82, h * 0.74);
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fill();
      ctx.strokeStyle = "rgba(125,249,255,0.14)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(w * 0.18, h * 0.74);
      ctx.lineTo(w * 0.82, h * 0.74);
      ctx.stroke();
    }

    function renderFrame(i) {
      const s = series();
      if (!s || !s.step.length) return;
      const idx = clamp(i, 0, s.step.length - 1);
      state.idx = idx;
      syncTimeline();
      if (rendererMode === "three") {
        updateScene(s, idx);
        renderer.render(scene, camera);
      } else {
        renderFallbackScene(s, idx);
      }
      const rect = stage.getBoundingClientRect();
      drawVelocityVector(rect.width * 0.5, rect.height * 0.5, s.roll_deg[idx], s.pitch_deg[idx]);
      updateTargetCue(s, idx);
      updateHud(s, idx);
      $("progress").textContent = `${idx + 1} / ${s.step.length}`;
      $("timeline").value = String(idx);
    }

    function tick(ts) {
      const s = series();
      const n = s ? s.step.length : 0;
      const delta = (ts - state.lastTs) / 1000.0;
      state.lastTs = ts;
      if (state.playing && n > 0) {
        state.acc += delta * state.speed;
        while (state.acc >= DT_SEC && state.idx < n - 1) {
          state.idx += 1;
          state.acc -= DT_SEC;
        }
      }
      renderFrame(state.idx);
      requestAnimationFrame(tick);
    }

    $("btnPlay").onclick = () => {
      state.playing = !state.playing;
      $("btnPlay").textContent = state.playing ? "Pause" : "Play";
    };
    $("speedSel").onchange = (e) => { state.speed = parseFloat(e.target.value || "1"); };
    $("timeline").oninput = (e) => {
      state.idx = parseInt(e.target.value || "0", 10) || 0;
      renderFrame(state.idx);
    };
    $("episodeSel").onchange = (e) => {
      state.episode = e.target.value;
      refillPlanes();
      renderFrame(0);
    };
    $("planeSel").onchange = (e) => {
      state.plane = e.target.value;
      state.idx = 0;
      syncTimeline();
      renderFrame(0);
    };

    const hasThree = initThree();
    resize();
    fillSelectors();
    $("planeSel").value = state.plane;
    if (!hasThree) {
      $("status").textContent = "Offline fallback renderer active";
    }
    window.addEventListener("resize", () => {
      resize();
      renderFrame(state.idx);
    });
    requestAnimationFrame(tick);
  </script>
</body>
</html>
"""
    return (
        template.replace("__DATA_JSON__", data_json)
        .replace("__DT_SEC__", json.dumps(dt_sec))
        .replace("__FIRE_AZIMUTH_DEG__", json.dumps(FIRE_AZIMUTH_DEG))
        .replace("__FIRE_SOLUTION_DEG__", json.dumps(FIRE_SOLUTION_DEG))
        .replace("__FIRE_ELEVATION_DEG__", json.dumps(FIRE_ELEVATION_DEG))
    )


def main() -> None:
    args = _parse_args()
    csv_path = Path(args.csv_path).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    payload = _build_payload(df, stride=args.stride)
    html = _html_with_data(json.dumps(payload, separators=(",", ":")), args.dt_sec)
    output_html.write_text(html, encoding="utf-8")
    print(f"Saved HTML: {output_html}")


if __name__ == "__main__":
    main()
