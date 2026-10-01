"""Miniature tracking with ArUco markers.

Each miniature wears a small printed 4x4 ArUco tag. The tracker finds the tags in
the overhead webcam, maps their centers into projector pixels with the
calibration homography, and snaps them to the battle grid.
"""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from core.vision.calibration import load_matrix, transform_point


def cell_pixels(config: dict) -> float:
    return config["vision"]["grid_size_inches"] * config["projector"]["pixels_per_inch"]


def entity_info(config: dict, marker_id: int) -> tuple[str, str]:
    info = (config.get("entities") or {}).get(marker_id) or (config.get("entities") or {}).get(str(marker_id))
    if info:
        return info.get("label", f"marker_{marker_id}"), info.get("role", "monster")
    return f"marker_{marker_id}", "monster"


def make_entity(config: dict, marker_id: int, pixel_x: float, pixel_y: float) -> dict:
    cell = cell_pixels(config)
    label, role = entity_info(config, marker_id)
    return {
        "entity_id": f"marker_{marker_id}",
        "label": label,
        "role": role,
        "grid_x": int(pixel_x // cell),
        "grid_y": int(pixel_y // cell),
        "pixel_x": int(round(pixel_x)),
        "pixel_y": int(round(pixel_y)),
    }


def get_dictionary(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


class Tracker:
    """Reads the webcam and returns the miniatures currently on the table."""

    def __init__(self, config: dict, open_camera: bool = True):
        self.config = config
        self.detector = cv2.aruco.ArucoDetector(
            get_dictionary(config["vision"]["aruco_dictionary"]), cv2.aruco.DetectorParameters()
        )
        self.lost_timeout = config["vision"]["lost_timeout_s"]
        self._seen: dict[str, tuple[float, dict]] = {}  # entity_id -> (last seen time, entity)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self.matrix = self._load_or_fallback_matrix()
        self.cap = None
        if open_camera:
            cam = config["camera"]
            self.cap = cv2.VideoCapture(cam["id"])
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera {cam['id']}")

    def _load_or_fallback_matrix(self) -> np.ndarray:
        path = self.config["projector"]["calibration_file"]
        try:
            return load_matrix(path)
        except FileNotFoundError:
            # Plain stretch from camera frame to projector frame. Positions will be
            # off on a real table until `--mode calibrate` has been run.
            print(f"[tracker] WARNING: {path} missing, using uncalibrated scaling. Run --mode calibrate.")
            cam, proj = self.config["camera"], self.config["projector"]
            return np.diag([proj["width"] / cam["width"], proj["height"] / cam["height"], 1.0])

    def detect(self, frame: np.ndarray) -> list[dict]:
        """Entities visible in this one frame (no memory)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return []
        out = []
        for quad, marker_id in zip(corners, ids.flatten()):
            cx, cy = quad.reshape(4, 2).mean(axis=0)
            x, y = transform_point(cx, cy, self.matrix)
            out.append(make_entity(self.config, int(marker_id), x, y))
        return out

    def update(self, frame: np.ndarray, now: float | None = None) -> list[dict]:
        """Detect in ``frame`` and merge with recently seen minis (hand-occlusion tolerance)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            self._frame = frame
            for e in self.detect(frame):
                self._seen[e["entity_id"]] = (now, e)
            self._seen = {k: v for k, v in self._seen.items() if now - v[0] <= self.lost_timeout}
            return [e for _, e in self._seen.values()]

    def read(self) -> list[dict]:
        """Grab one webcam frame and return the current entity list."""
        ok, frame = self.cap.read()
        if not ok:
            with self._lock:
                return [e for _, e in self._seen.values()]
        return self.update(frame)

    def snapshot_jpeg(self, max_width: int = 1024) -> bytes | None:
        with self._lock:
            frame = None if self._frame is None else self._frame.copy()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class DemoTracker:
    """Fake minis walking in circles. Same interface as Tracker, no camera needed."""

    def __init__(self, config: dict):
        self.config = config
        self.t0 = time.monotonic()
        ids = list((config.get("entities") or {}).keys())[:6] or [1, 12, 30]
        self.ids = [int(i) for i in ids]

    def read(self) -> list[dict]:
        t = time.monotonic() - self.t0
        w, h = self.config["projector"]["width"], self.config["projector"]["height"]
        out = []
        for i, marker_id in enumerate(self.ids):
            angle = t * (0.15 + 0.05 * i) + i * 2 * math.pi / len(self.ids)
            r = min(w, h) * (0.18 + 0.05 * (i % 3))
            out.append(make_entity(self.config, marker_id, w / 2 + r * math.cos(angle), h / 2 + r * math.sin(angle)))
        return out

    def snapshot_jpeg(self, max_width: int = 1024) -> bytes | None:
        return None

    def close(self) -> None:
        pass


def generate_marker_sheet(config: dict, path: str | Path = "marker_sheet.png", marker_px: int = 236) -> Path:
    """Printable sheet of every configured marker (236 px = 0.8 in at 300 DPI)."""
    dictionary = get_dictionary(config["vision"]["aruco_dictionary"])
    ids = sorted(int(i) for i in (config.get("entities") or {}).keys()) or list(range(8))
    cols, pad, label_h = 4, 60, 40
    rows = math.ceil(len(ids) / cols)
    cell_w, cell_h = marker_px + 2 * pad, marker_px + 2 * pad + label_h
    sheet = np.full((rows * cell_h, cols * cell_w), 255, np.uint8)
    for n, marker_id in enumerate(ids):
        r, c = divmod(n, cols)
        x, y = c * cell_w + pad, r * cell_h + pad
        sheet[y:y + marker_px, x:x + marker_px] = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)
        label, role = entity_info(config, marker_id)
        cv2.putText(sheet, f"#{marker_id} {label} ({role})", (x - pad // 2, y + marker_px + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet)
    return Path(path)
