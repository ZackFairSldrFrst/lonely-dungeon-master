"""Core logic checks. Run: python -m pytest tests  (or: python tests/test_core.py)"""

import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.ai.dm_agent import DMAgent, clean_response, extract_json  # noqa: E402
from core.audio.voice_service import hotword_pattern, strip_hotword  # noqa: E402
from core.vision.calibration import (  # noqa: E402
    compute_homography, find_green_targets, render_targets, target_points, transform_point,
)
from core.vision.tracker import Tracker, get_dictionary  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())


def test_calibration_finds_targets_and_maps_points():
    # Pretend the camera sees the projector image shrunk + shifted (a simple known warp).
    proj_w, proj_h = 1920, 1080
    projected = render_targets(proj_w, proj_h)
    cam_from_proj = np.array([[0.5, 0.02, 100], [0.01, 0.5, 60], [0, 0, 1]], dtype=np.float64)
    cam = cv2.warpPerspective(projected, cam_from_proj, (1280, 720))
    found = find_green_targets(cam, CONFIG["vision"]["calibration_hsv_low"], CONFIG["vision"]["calibration_hsv_high"])
    assert found is not None
    matrix = compute_homography(found, target_points(proj_w, proj_h))
    # Any camera point must land where the true inverse warp puts it.
    for px, py in [(300, 200), (960, 540), (1700, 900)]:
        cx, cy, cw = cam_from_proj @ [px, py, 1]
        x, y = transform_point(cx / cw, cy / cw, matrix)
        assert abs(x - px) < 3 and abs(y - py) < 3, (x, y, px, py)


def test_tracker_detects_markers_on_grid():
    tracker = Tracker(CONFIG, open_camera=False)
    tracker.matrix = np.eye(3)  # camera pixels == projector pixels for this test
    frame = np.full((720, 1280), 255, np.uint8)
    dictionary = get_dictionary(CONFIG["vision"]["aruco_dictionary"])
    frame[200:300, 400:500] = cv2.aruco.generateImageMarker(dictionary, 12, 100)  # Rogue, center (450, 250)
    frame[500:600, 900:1000] = cv2.aruco.generateImageMarker(dictionary, 30, 100)  # Goblin, center (950, 550)
    found = {e["entity_id"]: e for e in tracker.update(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), now=0.0)}
    cell = CONFIG["vision"]["grid_size_inches"] * CONFIG["projector"]["pixels_per_inch"]
    rogue, goblin = found["marker_12"], found["marker_30"]
    assert (rogue["label"], rogue["role"]) == ("Rogue", "player")
    assert (goblin["label"], goblin["role"]) == ("Goblin", "monster")
    assert abs(rogue["pixel_x"] - 450) <= 2 and abs(rogue["pixel_y"] - 250) <= 2
    assert rogue["grid_x"] == int(450 // cell) and rogue["grid_y"] == int(250 // cell)
    # A hand covers the table: minis stay for lost_timeout_s, then disappear.
    blank = np.full((720, 1280, 3), 255, np.uint8)
    assert len(tracker.update(blank, now=0.5)) == 2
    assert tracker.update(blank, now=5.0) == []


def test_dm_reply_parsing():
    messy = 'Sure!\n```json\n{"narration": "The goblin flinches.", "grid_effects": [' \
            '{"type": "spell_aura", "x": 5, "y": "8", "radius": 2, "color": "gold"}, {"type": "bogus"}, "junk"],' \
            ' "state_updates": {"monster_1_hp": 12}}\n```'
    result = clean_response(extract_json(messy))
    assert result["narration"] == "The goblin flinches."
    assert result["grid_effects"] == [
        {"type": "spell_aura", "x": 5.0, "y": 8.0, "radius": 2.0, "color": "gold", "duration": 6.0}
    ]
    assert result["state_updates"] == {"monster_1_hp": 12}


def test_dm_agent_turn_with_fake_model():
    cfg = {**CONFIG, "ai": {**CONFIG["ai"], "provider": "ollama"}}  # ollama needs no key to construct
    agent = DMAgent(cfg)
    seen = {}

    def fake_call(turn_text, image):
        seen["turn"] = turn_text
        return '{"narration": "Steel rings.", "grid_effects": [], "state_updates": {"goblin_hp": 3}}'

    agent._call_model = fake_call
    entities = [{"entity_id": "marker_12", "label": "Rogue", "role": "player", "grid_x": 4, "grid_y": 7}]
    out = agent.respond("I stab the goblin", entities)
    assert out["narration"] == "Steel rings."
    assert agent.game_state == {"goblin_hp": 3}
    assert "Rogue (player, id marker_12) at grid (4, 7)" in seen["turn"]
    assert len(agent.history) == 2

    agent._call_model = lambda *a: (_ for _ in ()).throw(RuntimeError("network down"))
    assert "pauses" in agent.respond("hello?", entities)["narration"]  # never crashes the game


def test_hotword():
    p = hotword_pattern("hey dm")
    assert strip_hotword("Hey DM, I attack the goblin.", p) == "I attack the goblin."
    assert strip_hotword("hey d.m. cast fireball", p) == "cast fireball"
    assert strip_hotword("Hey DM", p) == ""
    assert strip_hotword("hey there, pass the chips", p) is None
    assert strip_hotword("Hey DMV office", p) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
