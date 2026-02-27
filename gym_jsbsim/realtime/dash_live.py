import argparse
import json
import math
import os

from dash import Dash, Input, Output, State, dcc, html
from dash_extensions import WebSocket
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title)
    return fig


def _integrate_xy(ep_rows: list[dict], dt_sec: float = 0.2) -> tuple[list[float], list[float], list[float]]:
    x_m, y_m, z_ft = [0.0], [0.0], [_f(ep_rows[0], "position_h_sl_ft")] if ep_rows else ([0.0], [0.0], [0.0])
    for row in ep_rows[1:]:
        vn = _f(row, "velocities_v_north_fps") * 0.3048
        ve = _f(row, "velocities_v_east_fps") * 0.3048
        x_m.append(x_m[-1] + ve * dt_sec)
        y_m.append(y_m[-1] + vn * dt_sec)
        z_ft.append(_f(row, "position_h_sl_ft"))
    return x_m, y_m, z_ft


def _sim3d_figure(ep_rows: list[dict], latest_ep: int) -> go.Figure:
    if not ep_rows:
        return _empty("3D Flight Track")

    x_m, y_m, z_ft = _integrate_xy(ep_rows)
    last = ep_rows[-1]
    heading_deg = _f(last, "attitude_psi_deg")
    roll_deg = _f(last, "roll_deg")
    pitch_deg = _f(last, "pitch_deg")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=x_m,
            y=y_m,
            z=z_ft,
            mode="lines",
            name="trajectory",
            line=dict(color="#1f77b4", width=6),
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x_m[-1]],
            y=[y_m[-1]],
            z=[z_ft[-1]],
            mode="markers+text",
            name="aircraft",
            marker=dict(size=7, color="#d62728"),
            text=[f"EP{latest_ep}  HDG={heading_deg:.1f}  R={roll_deg:.1f}  P={pitch_deg:.1f}"],
            textposition="top center",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title=f"3D Flight Track (Episode {latest_ep})",
        scene=dict(
            xaxis_title="East (m)",
            yaxis_title="North (m)",
            zaxis_title="Altitude (ft)",
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        height=450,
    )
    return fig


def _attitude_figure(last: dict) -> go.Figure:
    roll_deg = _f(last, "roll_deg")
    pitch_deg = _f(last, "pitch_deg")
    roll_rad = math.radians(roll_deg)
    pitch_offset = max(-0.6, min(0.6, pitch_deg / 30.0))

    d_x = math.cos(roll_rad)
    d_y = math.sin(roll_rad)
    x0, y0 = -1.1 * d_x, pitch_offset - 1.1 * d_y
    x1, y1 = 1.1 * d_x, pitch_offset + 1.1 * d_y

    fig = go.Figure()
    fig.add_shape(type="rect", x0=-1.2, y0=pitch_offset, x1=1.2, y1=1.2, fillcolor="#a8d8ff", line_width=0)
    fig.add_shape(type="rect", x0=-1.2, y0=-1.2, x1=1.2, y1=pitch_offset, fillcolor="#c9a27e", line_width=0)
    fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color="white", width=5))
    fig.add_shape(type="circle", x0=-1.0, y0=-1.0, x1=1.0, y1=1.0, line=dict(color="#2f2f2f", width=3))
    fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[0, 0], mode="lines", line=dict(color="#111", width=6), showlegend=False))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=9, color="#111"), showlegend=False))
    fig.update_xaxes(range=[-1.1, 1.1], visible=False)
    fig.update_yaxes(range=[-1.1, 1.1], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        template="plotly_white",
        title=f"Attitude Indicator (Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°)",
        margin=dict(l=10, r=10, t=45, b=10),
        height=360,
    )
    return fig


def _hud_figure(last: dict) -> go.Figure:
    heading = _f(last, "attitude_psi_deg")
    target = _f(last, "target_track_deg")
    track_err = _f(last, "error_track_error_deg")
    altitude = _f(last, "position_h_sl_ft")
    roll = _f(last, "roll_deg")
    pitch = _f(last, "pitch_deg")

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{"type": "indicator"}] * 3, [{"type": "indicator"}] * 3],
        vertical_spacing=0.2,
        horizontal_spacing=0.1,
    )
    fig.add_trace(go.Indicator(mode="number", value=heading, title={"text": "Heading (deg)"}), row=1, col=1)
    fig.add_trace(go.Indicator(mode="number", value=target, title={"text": "Target Track (deg)"}), row=1, col=2)
    fig.add_trace(go.Indicator(mode="number", value=track_err, title={"text": "Track Error (deg)"}), row=1, col=3)
    fig.add_trace(go.Indicator(mode="number", value=altitude, title={"text": "Altitude (ft)"}), row=2, col=1)
    fig.add_trace(go.Indicator(mode="number", value=roll, title={"text": "Roll (deg)"}), row=2, col=2)
    fig.add_trace(go.Indicator(mode="number", value=pitch, title={"text": "Pitch (deg)"}), row=2, col=3)
    fig.update_layout(template="plotly_white", title="HUD", height=360, margin=dict(l=10, r=10, t=45, b=10))
    return fig


def build_dash_app(ws_url: str, csv_path: str, max_points: int) -> Dash:
    app = Dash(__name__)
    header = [html.H3("Live SB3 Flight Dashboard"), dcc.Interval(id="tick", interval=500, n_intervals=0)]
    if ws_url:
        header.extend([html.Div(f"WebSocket: {ws_url}"), WebSocket(id="ws", url=ws_url)])
    else:
        header.append(html.Div(f"CSV polling mode: {csv_path}"))

    app.layout = html.Div(
        header
        + [
            dcc.Store(id="buffer", data=[]),
            dcc.Graph(id="sim3d_fig"),
            dcc.Graph(id="hud_fig"),
            dcc.Graph(id="attitude_fig"),
            dcc.Graph(id="reward_fig"),
            dcc.Graph(id="heading_fig"),
            dcc.Graph(id="altitude_fig"),
        ]
    )

    if ws_url:
        @app.callback(Output("buffer", "data"), Input("ws", "message"), State("buffer", "data"))
        def on_ws_message(msg, buffer_rows):
            if not msg:
                return buffer_rows
            rows = json.loads(msg["data"])
            return (buffer_rows + rows)[-5000:]
    else:
        @app.callback(Output("buffer", "data"), Input("tick", "n_intervals"))
        def poll_csv(_):
            if not os.path.exists(csv_path):
                return []
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                return []
            return df.tail(5000).to_dict(orient="records")

    @app.callback(
        Output("sim3d_fig", "figure"),
        Output("hud_fig", "figure"),
        Output("attitude_fig", "figure"),
        Output("reward_fig", "figure"),
        Output("heading_fig", "figure"),
        Output("altitude_fig", "figure"),
        Input("tick", "n_intervals"),
        State("buffer", "data"),
    )
    def draw_figures(_, buffer_rows):
        if not buffer_rows:
            empty = _empty("Waiting for data...")
            return empty, empty, empty, empty, empty, empty

        rows = buffer_rows[-max_points:]
        latest_ep = max(int(r.get("episode", 0)) for r in rows)
        ep_rows = [r for r in rows if int(r.get("episode", 0)) == latest_ep]
        last = ep_rows[-1]

        steps = [r.get("step", idx) for idx, r in enumerate(ep_rows)]
        rewards = [r.get("reward", 0.0) for r in ep_rows]
        headings = [r.get("attitude_psi_deg", 0.0) for r in ep_rows]
        targets = [r.get("target_track_deg", 0.0) for r in ep_rows]
        altitudes = [r.get("position_h_sl_ft", 0.0) for r in ep_rows]
        track_err = [r.get("error_track_error_deg", 0.0) for r in ep_rows]
        roll = [_f(r, "roll_deg") for r in ep_rows]
        pitch = [_f(r, "pitch_deg") for r in ep_rows]

        sim3d_fig = _sim3d_figure(ep_rows, latest_ep)
        hud_fig = _hud_figure(last)
        attitude_fig = _attitude_figure(last)

        reward_fig = go.Figure()
        reward_fig.add_scatter(x=steps, y=rewards, mode="lines", name="reward")
        reward_fig.update_layout(template="plotly_white", title=f"Reward (episode {latest_ep})")

        heading_fig = go.Figure()
        heading_fig.add_scatter(x=steps, y=headings, mode="lines", name="heading_deg")
        heading_fig.add_scatter(x=steps, y=targets, mode="lines", name="target_track_deg")
        heading_fig.add_scatter(x=steps, y=roll, mode="lines", name="roll_deg", line=dict(dash="dot"))
        heading_fig.add_scatter(x=steps, y=pitch, mode="lines", name="pitch_deg", line=dict(dash="dash"))
        heading_fig.update_layout(template="plotly_white", title="Heading vs Target")

        altitude_fig = go.Figure()
        altitude_fig.add_scatter(x=steps, y=altitudes, mode="lines", name="altitude_ft")
        altitude_fig.add_scatter(x=steps, y=track_err, mode="lines", name="track_error_deg")
        altitude_fig.update_layout(template="plotly_white", title="Altitude + Track Error")

        return sim3d_fig, hud_fig, attitude_fig, reward_fig, heading_fig, altitude_fig

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dash realtime viewer for gym-jsbsim eval data.")
    parser.add_argument("--ws-url", type=str, default="")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/content/gym-jsbsim/models/heading_control_f16/eval_metrics.csv",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--max-points", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_dash_app(args.ws_url, args.csv_path, args.max_points)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
