import argparse
import json
import math
import os

from dash import Dash, Input, Output, State, dcc, html
from dash_extensions import WebSocket
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


THEME = {
    "bg_page": "#f4f8fc",
    "bg_panel": "#ffffff",
    "bg_panel_soft": "#eef4fb",
    "text_main": "#0f172a",
    "text_dim": "#475569",
    "line": "#d6e0ec",
    "brand": "#0f766e",
    "brand_alt": "#ea580c",
    "danger": "#dc2626",
}


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title)
    return _style_figure(fig)


def _style_figure(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    if title:
        fig.update_layout(title=title)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=THEME["bg_panel"],
        plot_bgcolor=THEME["bg_panel"],
        font={"family": "IBM Plex Sans, Segoe UI, sans-serif", "color": THEME["text_main"]},
        title={"font": {"size": 18, "color": THEME["text_main"]}, "x": 0.02, "xanchor": "left"},
        margin={"l": 16, "r": 16, "t": 54, "b": 16},
        height=height,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME["line"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=THEME["line"], zeroline=False)
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
            name="Trajectory",
            line={"color": THEME["brand"], "width": 7},
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[x_m[-1]],
            y=[y_m[-1]],
            z=[z_ft[-1]],
            mode="markers+text",
            name="Aircraft",
            marker={"size": 8, "color": THEME["brand_alt"]},
            text=[f"EP{latest_ep}  HDG={heading_deg:.1f}  R={roll_deg:.1f}  P={pitch_deg:.1f}"],
            textposition="top center",
            textfont={"size": 12, "color": THEME["text_main"]},
        )
    )
    fig.update_layout(
        title=f"3D Flight Track (Episode {latest_ep})",
        scene={
            "xaxis_title": "East (m)",
            "yaxis_title": "North (m)",
            "zaxis_title": "Altitude (ft)",
            "xaxis": {"gridcolor": THEME["line"], "zeroline": False, "backgroundcolor": THEME["bg_panel_soft"]},
            "yaxis": {"gridcolor": THEME["line"], "zeroline": False, "backgroundcolor": THEME["bg_panel_soft"]},
            "zaxis": {"gridcolor": THEME["line"], "zeroline": False, "backgroundcolor": THEME["bg_panel_soft"]},
            "camera": {"eye": {"x": 1.6, "y": -1.7, "z": 0.7}},
        },
        paper_bgcolor=THEME["bg_panel"],
        height=460,
        margin={"l": 16, "r": 16, "t": 54, "b": 16},
        font={"family": "IBM Plex Sans, Segoe UI, sans-serif", "color": THEME["text_main"]},
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
    fig.add_shape(type="rect", x0=-1.2, y0=pitch_offset, x1=1.2, y1=1.2, fillcolor="#cdeafe", line_width=0)
    fig.add_shape(type="rect", x0=-1.2, y0=-1.2, x1=1.2, y1=pitch_offset, fillcolor="#d5b28e", line_width=0)
    fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line={"color": "#ffffff", "width": 5})
    fig.add_shape(type="circle", x0=-1.0, y0=-1.0, x1=1.0, y1=1.0, line={"color": "#1f2937", "width": 3})
    fig.add_trace(go.Scatter(x=[-0.2, 0.2], y=[0, 0], mode="lines", line={"color": "#111827", "width": 6}, showlegend=False))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker={"size": 9, "color": "#111827"}, showlegend=False))
    fig.update_xaxes(range=[-1.1, 1.1], visible=False)
    fig.update_yaxes(range=[-1.1, 1.1], visible=False, scaleanchor="x", scaleratio=1)
    return _style_figure(fig, title=f"Attitude Indicator (Roll={roll_deg:.1f}°, Pitch={pitch_deg:.1f}°)", height=380)


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
        vertical_spacing=0.22,
        horizontal_spacing=0.08,
    )
    number_cfg = {"font": {"size": 28, "color": THEME["text_main"]}}
    title_cfg = lambda t: {"text": t, "font": {"size": 13, "color": THEME["text_dim"]}}
    fig.add_trace(go.Indicator(mode="number", value=heading, number=number_cfg, title=title_cfg("Heading (deg)")), row=1, col=1)
    fig.add_trace(go.Indicator(mode="number", value=target, number=number_cfg, title=title_cfg("Target Track (deg)")), row=1, col=2)
    fig.add_trace(go.Indicator(mode="number", value=track_err, number=number_cfg, title=title_cfg("Track Error (deg)")), row=1, col=3)
    fig.add_trace(go.Indicator(mode="number", value=altitude, number=number_cfg, title=title_cfg("Altitude (ft)")), row=2, col=1)
    fig.add_trace(go.Indicator(mode="number", value=roll, number=number_cfg, title=title_cfg("Roll (deg)")), row=2, col=2)
    fig.add_trace(go.Indicator(mode="number", value=pitch, number=number_cfg, title=title_cfg("Pitch (deg)")), row=2, col=3)
    return _style_figure(fig, title="HUD", height=380)


def _dashboard_shell(ws_url: str, csv_path: str):
    source_label = f"WebSocket: {ws_url}" if ws_url else f"CSV polling mode: {csv_path}"
    return html.Div(
        [
            html.Div(
                className="hero",
                children=[
                    html.Div(
                        className="hero-title",
                        children=[
                            html.H2("Live Flight Dashboard"),
                            html.P("Realtime telemetry, reward trace, and 3D trajectory"),
                        ],
                    ),
                    html.Div(className="hero-chip", children=source_label),
                ],
            ),
            dcc.Store(id="buffer", data=[]),
            dcc.Interval(id="tick", interval=500, n_intervals=0),
            WebSocket(id="ws", url=ws_url) if ws_url else html.Div(id="ws-placeholder", style={"display": "none"}),
            html.Div(
                className="grid",
                children=[
                    html.Div(className="card span-2", children=[dcc.Graph(id="sim3d_fig", config={"displaylogo": False})]),
                    html.Div(className="card", children=[dcc.Graph(id="hud_fig", config={"displaylogo": False})]),
                    html.Div(className="card", children=[dcc.Graph(id="attitude_fig", config={"displaylogo": False})]),
                    html.Div(className="card", children=[dcc.Graph(id="reward_fig", config={"displaylogo": False})]),
                    html.Div(className="card", children=[dcc.Graph(id="heading_fig", config={"displaylogo": False})]),
                    html.Div(className="card", children=[dcc.Graph(id="altitude_fig", config={"displaylogo": False})]),
                ],
            ),
        ],
        className="page",
    )


def build_dash_app(ws_url: str, csv_path: str, max_points: int) -> Dash:
    app = Dash(__name__)
    app.title = "Flight Dashboard"
    app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root {
        --page-a: #d9eef4;
        --page-b: #f5f8fc;
        --panel: #ffffff;
        --text: #0f172a;
        --dim: #475569;
        --line: #d6e0ec;
        --brand: #0f766e;
        --brand-alt: #ea580c;
      }
      * { box-sizing: border-box; }
      html, body { margin: 0; padding: 0; }
      body {
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
        background:
          radial-gradient(1100px 520px at 90% -10%, rgba(15,118,110,0.18), transparent 60%),
          radial-gradient(900px 520px at -10% 10%, rgba(234,88,12,0.12), transparent 58%),
          linear-gradient(180deg, var(--page-a) 0%, var(--page-b) 42%, #ffffff 100%);
        color: var(--text);
      }
      .page {
        max-width: 1400px;
        margin: 0 auto;
        padding: 18px 16px 28px;
      }
      .hero {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 12px;
        margin-bottom: 14px;
      }
      .hero-title h2 {
        margin: 0;
        font-size: 28px;
        letter-spacing: 0.2px;
      }
      .hero-title p {
        margin: 6px 0 0;
        color: var(--dim);
        font-size: 14px;
      }
      .hero-chip {
        border: 1px solid var(--line);
        background: rgba(255,255,255,0.85);
        color: var(--dim);
        padding: 8px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        backdrop-filter: blur(6px);
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(280px, 1fr));
        gap: 14px;
      }
      .card {
        border: 1px solid var(--line);
        border-radius: 16px;
        background: var(--panel);
        box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
        overflow: hidden;
      }
      .span-2 { grid-column: span 2; }
      .js-plotly-plot .plotly .modebar {
        background: rgba(255,255,255,0.8) !important;
        border-radius: 10px !important;
      }
      @media (max-width: 960px) {
        .hero { flex-direction: column; align-items: flex-start; }
        .grid { grid-template-columns: 1fr; }
        .span-2 { grid-column: auto; }
      }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""
    app.layout = _dashboard_shell(ws_url, csv_path)

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
        reward_fig.add_scatter(x=steps, y=rewards, mode="lines", name="Reward", line={"color": THEME["brand"], "width": 3})
        reward_fig.add_scatter(
            x=[steps[-1]],
            y=[rewards[-1]],
            mode="markers",
            marker={"size": 9, "color": THEME["brand_alt"], "line": {"color": "white", "width": 1}},
            name="Latest",
        )
        reward_fig = _style_figure(reward_fig, title=f"Reward (Episode {latest_ep})")

        heading_fig = go.Figure()
        heading_fig.add_scatter(x=steps, y=headings, mode="lines", name="Heading", line={"color": THEME["brand"], "width": 3})
        heading_fig.add_scatter(x=steps, y=targets, mode="lines", name="Target", line={"color": THEME["brand_alt"], "width": 2})
        heading_fig.add_scatter(x=steps, y=roll, mode="lines", name="Roll", line={"dash": "dot", "color": "#2563eb", "width": 2})
        heading_fig.add_scatter(x=steps, y=pitch, mode="lines", name="Pitch", line={"dash": "dash", "color": "#9a3412", "width": 2})
        heading_fig = _style_figure(heading_fig, title="Heading and Attitude")

        altitude_fig = go.Figure()
        altitude_fig.add_scatter(x=steps, y=altitudes, mode="lines", name="Altitude (ft)", line={"color": "#0f766e", "width": 3})
        altitude_fig.add_scatter(x=steps, y=track_err, mode="lines", name="Track Error (deg)", line={"color": THEME["danger"], "width": 2})
        altitude_fig = _style_figure(altitude_fig, title="Altitude and Track Error")

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
