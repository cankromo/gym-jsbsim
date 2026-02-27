import argparse
import asyncio
import json
import os
from typing import Set

import pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect


def build_app(csv_path: str, poll_sec: float) -> FastAPI:
    app = FastAPI(title="gym-jsbsim realtime publisher")
    clients: Set[WebSocket] = set()
    state = {"last_rows": 0}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "clients": len(clients), "csv_path": csv_path}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        clients.add(ws)
        try:
            while True:
                # Keep connection alive; client may send ping text.
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(ws)

    async def broadcaster() -> None:
        while True:
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    if len(df) > state["last_rows"]:
                        new_rows = df.iloc[state["last_rows"] :].to_dict(orient="records")
                        payload = json.dumps(new_rows)
                        disconnected = []
                        for ws in list(clients):
                            try:
                                await ws.send_text(payload)
                            except Exception:
                                disconnected.append(ws)
                        for ws in disconnected:
                            clients.discard(ws)
                        state["last_rows"] = len(df)
                except Exception:
                    # Ignore transient partial-write CSV parse errors.
                    pass
            await asyncio.sleep(poll_sec)

    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(broadcaster())

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish eval CSV rows over WebSocket.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="/content/gym-jsbsim/models/heading_control_f16/eval_metrics.csv",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--poll-sec", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = build_app(csv_path=args.csv_path, poll_sec=args.poll_sec)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

