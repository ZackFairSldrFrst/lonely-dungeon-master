"""Camera -> projector calibration.

Projects four green dots into the corners of the projector image, finds them in
the overhead webcam, and solves the 3x3 homography that maps webcam pixels to
projector pixels. The result is saved to ``calibration_matrix.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

WINDOW_PROJECTOR = "LDM calibration (projector)"
WINDOW_PREVIEW = "LDM calibration (camera preview)"
STABLE_FRAMES_NEEDED = 15
DEFAULT_FILE = "calibration_matrix.json"

_cached: dict[str, np.ndarray] = {}


# ---------------------------------------------------------------- geometry

def target_points(width: int, height: int, margin_frac: float = 0.08) -> np.ndarray:
    """Projector-space centers of the 4 targets, ordered TL, TR, BR, BL."""
    mx, my = int(width * margin_frac), int(height * margin_frac)
    return np.array(
        [[mx, my], [width - mx, my], [width - mx, height - my], [mx, height - my]],
        dtype=np.float32,
    )


def render_targets(width: int, height: int) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    radius = max(12, int(min(width, height) * 0.035))
    for x, y in target_points(width, height):
        cv2.circle(img, (int(x), int(y)), radius, (0, 255, 0), -1, cv2.LINE_AA)
    return img


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort 4 points into TL, TR, BR, BL (works for any mild rotation/keystone)."""
    pts = np.asarray(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array(
        [pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
        dtype=np.float32,
    )


def find_green_targets(frame: np.ndarray, hsv_low, hsv_high, min_area: float = 30.0) -> np.ndarray | None:
    """Return the 4 green target centers in camera pixels (TL, TR, BR, BL) or None."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low, np.uint8), np.array(hsv_high, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = sorted((c for c in contours if cv2.contourArea(c) >= min_area), key=cv2.contourArea, reverse=True)[:4]
    if len(blobs) < 4:
        return None
    centers = []
    for c in blobs:
        m = cv2.moments(c)
        centers.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
    return order_corners(np.array(centers))


def compute_homography(cam_pts: np.ndarray, proj_pts: np.ndarray) -> np.ndarray:
    matrix, _ = cv2.findHomography(np.asarray(cam_pts, np.float32), np.asarray(proj_pts, np.float32))
    if matrix is None:
        raise ValueError("Could not solve homography (are the 4 targets collinear?)")
    return matrix


# ---------------------------------------------------------------- persistence

def save_matrix(matrix: np.ndarray, path: str | Path, camera_size, projector_size) -> None:
    Path(path).write_text(json.dumps({
        "matrix": np.asarray(matrix).tolist(),
        "camera_size": list(camera_size),
        "projector_size": list(projector_size),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    _cached.pop(str(path), None)


def load_matrix(path: str | Path = DEFAULT_FILE) -> np.ndarray:
    key = str(path)
    if key not in _cached:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run `python main.py --mode calibrate` first.")
        _cached[key] = np.array(json.loads(p.read_text())["matrix"], dtype=np.float64)
    return _cached[key]


def transform_point(cam_x: float, cam_y: float, matrix: np.ndarray | None = None) -> tuple[float, float]:
    """Map a webcam pixel to the matching projector canvas pixel (X, Y)."""
    h = load_matrix() if matrix is None else matrix
    x, y, w = h @ np.array([cam_x, cam_y, 1.0])
    return float(x / w), float(y / w)


# ---------------------------------------------------------------- projector window

def projector_geometry(display_index: int, fallback_size) -> tuple[int, int, int, int]:
    """(x, y, width, height) of the projector screen on the desktop."""
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        if display_index < len(monitors):
            m = monitors[display_index]
            return m.x, m.y, m.width, m.height
        print(f"[calibrate] display {display_index} not found ({len(monitors)} screens); using primary")
    except Exception as e:  # headless / no X server info
        print(f"[calibrate] screeninfo unavailable ({e}); using config size")
    return 0, 0, int(fallback_size[0]), int(fallback_size[1])


def open_fullscreen(name: str, x: int, y: int) -> None:
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(name, x, y)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# ---------------------------------------------------------------- interactive run

def run_calibration(config: dict) -> np.ndarray | None:
    cam_cfg, proj_cfg, vis_cfg = config["camera"], config["projector"], config["vision"]
    px, py, pw, ph = projector_geometry(proj_cfg["display_index"], (proj_cfg["width"], proj_cfg["height"]))
    if (pw, ph) != (proj_cfg["width"], proj_cfg["height"]):
        print(f"[calibrate] note: projector screen is {pw}x{ph}, config says "
              f"{proj_cfg['width']}x{proj_cfg['height']}. Using the real screen size.")

    open_fullscreen(WINDOW_PROJECTOR, px, py)
    cv2.imshow(WINDOW_PROJECTOR, render_targets(pw, ph))
    cv2.namedWindow(WINDOW_PREVIEW, cv2.WINDOW_NORMAL)

    cap = cv2.VideoCapture(cam_cfg["id"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])
    if not cap.isOpened():
        cv2.destroyAllWindows()
        raise RuntimeError(f"Cannot open camera {cam_cfg['id']}")

    print("[calibrate] Clear the table. Waiting for 4 green targets... (ESC to abort)")
    stable, history, matrix = 0, [], None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[calibrate] camera read failed, retrying")
                time.sleep(0.1)
                continue
            pts = find_green_targets(frame, vis_cfg["calibration_hsv_low"], vis_cfg["calibration_hsv_high"])
            preview = frame.copy()
            if pts is not None:
                stable += 1
                history.append(pts)
                for i, (x, y) in enumerate(pts):
                    cv2.circle(preview, (int(x), int(y)), 10, (0, 0, 255), 2)
                    cv2.putText(preview, "TL TR BR BL".split()[i], (int(x) + 12, int(y)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                stable, history = 0, []
            cv2.putText(preview, f"stable {stable}/{STABLE_FRAMES_NEEDED}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(WINDOW_PREVIEW, preview)
            if (cv2.waitKey(30) & 0xFF) == 27:
                print("[calibrate] aborted")
                return None
            if stable >= STABLE_FRAMES_NEEDED:
                cam_pts = np.mean(history[-STABLE_FRAMES_NEEDED:], axis=0)  # average out jitter
                matrix = compute_homography(cam_pts, target_points(pw, ph))
                h, w = frame.shape[:2]
                save_matrix(matrix, proj_cfg["calibration_file"], (w, h), (pw, ph))
                print(f"[calibrate] saved {proj_cfg['calibration_file']}")
                return matrix
    finally:
        cap.release()
        cv2.destroyAllWindows()
