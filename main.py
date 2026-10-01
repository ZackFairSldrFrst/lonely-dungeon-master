"""Lonely Dungeon Master: tabletop AR server.

    python main.py --mode calibrate   # align webcam and projector (run once per setup)
    python main.py --mode markers     # write a printable sheet of miniature tags
    python main.py --mode run         # play
    python main.py --mode demo        # play without a camera (fake miniatures)
"""

import argparse
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Path) -> None:
    """Minimal .env reader so people can drop their key in a file. Real env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip().removeprefix("export ").strip(), value.strip().strip("'\""))


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class Hub:
    """Fan-out of JSON messages to every connected projector page."""

    def __init__(self):
        self.clients: set = set()

    async def broadcast(self, message: dict) -> None:
        text = json.dumps(message)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)


def create_app(config: dict, demo: bool = False, voice_enabled: bool | None = None):
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel

    from core.ai.dm_agent import DMAgent
    from core.vision.tracker import DemoTracker, Tracker, cell_pixels

    hub = Hub()
    state: dict = {"entities": [], "last_dm": None}
    dm_lock = asyncio.Lock()
    services: dict = {}

    def client_config() -> dict:
        p = config["projector"]
        return {
            "type": "config",
            "width": p["width"],
            "height": p["height"],
            "cell_px": cell_pixels(config),
            "fog_radius_cells": config["vision"].get("fog_radius_cells", 6),
        }

    async def handle_command(text: str) -> dict:
        """Player words -> DM -> projector effects + spoken narration."""
        text = text.strip()
        if not text:
            raise ValueError("empty command")
        async with dm_lock:  # one DM turn at a time
            await hub.broadcast({"type": "thinking", "text": text})
            tracker = services["tracker"]
            snapshot = await asyncio.to_thread(tracker.snapshot_jpeg)
            result = await asyncio.to_thread(services["dm"].respond, text, state["entities"], snapshot)
            event = {"type": "dm", "player_text": text, **result, "game_state": services["dm"].game_state}
            state["last_dm"] = event
            await hub.broadcast(event)  # effects start animating...
            if services.get("voice"):
                services["voice"].speak(result["narration"])  # ...while the narration plays
            print(f"[dm] {result['narration']}")
            return event

    async def tracker_loop() -> None:
        period = 1.0 / float(config["vision"]["fps"])
        loop = asyncio.get_running_loop()
        while True:
            started = loop.time()
            try:
                state["entities"] = await asyncio.to_thread(services["tracker"].read)
                await hub.broadcast({"type": "entities", "entities": state["entities"]})
            except Exception as e:
                print(f"[tracker] {type(e).__name__}: {e}")
            await asyncio.sleep(max(0.0, period - (loop.time() - started)))

    @asynccontextmanager
    async def lifespan(app):
        loop = asyncio.get_running_loop()
        services["tracker"] = DemoTracker(config) if demo else Tracker(config)
        services["dm"] = DMAgent(config)
        use_voice = config["voice"].get("enabled", True) if voice_enabled is None else voice_enabled
        if use_voice:
            from core.audio.voice_service import VoiceService

            def on_command(text: str) -> None:  # called from voice threads
                fut = asyncio.run_coroutine_threadsafe(handle_command(text), loop)
                fut.add_done_callback(lambda f: f.exception() and print(f"[dm] failed: {f.exception()}"))

            services["voice"] = VoiceService(config, on_command)
            services["voice"].start()
        task = asyncio.create_task(tracker_loop())
        try:
            yield
        finally:
            task.cancel()
            if services.get("voice"):
                services["voice"].stop()
            services["tracker"].close()

    app = FastAPI(title="Lonely Dungeon Master", lifespan=lifespan)

    class Command(BaseModel):
        text: str

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        hub.clients.add(ws)
        try:
            await ws.send_text(json.dumps(client_config()))
            if state["last_dm"]:
                await ws.send_text(json.dumps({**state["last_dm"], "replay": True}))
            while True:
                msg = json.loads(await ws.receive_text())
                if msg.get("type") == "listen" and services.get("voice"):
                    services["voice"].push_to_talk()
                elif msg.get("type") == "command" and msg.get("text"):
                    asyncio.create_task(handle_command(msg["text"]))
        except (WebSocketDisconnect, RuntimeError, json.JSONDecodeError):
            pass
        finally:
            hub.clients.discard(ws)

    @app.post("/api/command")
    async def api_command(cmd: Command):
        try:
            return await handle_command(cmd.text)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/listen")
    async def api_listen():
        if not services.get("voice"):
            raise HTTPException(409, "voice is disabled")
        services["voice"].push_to_talk()
        return {"listening": True}

    @app.post("/api/reset")
    async def api_reset():
        services["dm"].reset()
        state["last_dm"] = None
        await hub.broadcast({"type": "reset"})
        return {"reset": True}

    @app.get("/api/state")
    async def api_state():
        return {"entities": state["entities"], "game_state": services["dm"].game_state, "last_dm": state["last_dm"]}

    @app.get("/api/config")
    async def api_config():
        return client_config()

    frontend = ROOT / "frontend" / "dist"
    if not frontend.exists():
        frontend = ROOT / "frontend"  # unbuilt: index.html pulls three.js from a CDN via import map

    @app.get("/")
    async def index():
        return FileResponse(frontend / "index.html")

    app.mount("/", StaticFiles(directory=frontend), name="frontend")
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Lonely Dungeon Master")
    parser.add_argument("--mode", choices=["calibrate", "run", "demo", "markers"], default="run")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--no-voice", action="store_true", help="disable microphone and speech")
    args = parser.parse_args()

    os.chdir(ROOT)  # calibration_matrix.json, .env and marker_sheet.png live next to main.py
    load_dotenv(ROOT / ".env")
    config = load_config(args.config)

    if args.mode == "calibrate":
        from core.vision.calibration import run_calibration
        run_calibration(config)
        return
    if args.mode == "markers":
        from core.vision.tracker import generate_marker_sheet
        path = generate_marker_sheet(config)
        print(f"Wrote {path}. Print it so each tag is about 0.75-1 inch wide and glue one under (or on top of) each miniature.")
        return

    import uvicorn
    app = create_app(config, demo=args.mode == "demo", voice_enabled=False if args.no_voice else None)
    host, port = config["server"]["host"], config["server"]["port"]
    print(f"Open http://{host}:{port} on the projector screen and press F for fullscreen.")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
