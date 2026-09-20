from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class GridGeometry:
    origin_x: float
    origin_y: float
    pitch: float
    full_columns: int
    full_rows: int

    def bounds(self, row: int, column: int) -> tuple[int, int, int, int]:
        x0 = round(self.origin_x + column * self.pitch)
        y0 = round(self.origin_y + row * self.pitch)
        x1 = round(self.origin_x + (column + 1) * self.pitch)
        y1 = round(self.origin_y + (row + 1) * self.pitch)
        return x0, y0, x1, y1

    def center(self, row: int, column: int) -> tuple[int, int]:
        return (
            round(self.origin_x + (column + 0.5) * self.pitch),
            round(self.origin_y + (row + 0.5) * self.pitch),
        )


def _axis_score(
    projection: np.ndarray,
    origin: float,
    pitch: float,
    minimum_lines: int,
) -> float:
    positions = np.arange(origin, len(projection), pitch)
    if len(positions) < minimum_lines:
        return float("-inf")
    boundary_values: list[float] = []
    middle_values: list[float] = []
    for position in positions:
        index = int(round(position))
        lo = max(0, index - 2)
        hi = min(len(projection), index + 3)
        boundary_values.append(float(np.max(projection[lo:hi])))
        middle = int(round(position + pitch * 0.5))
        if middle < len(projection):
            middle_values.append(float(projection[middle]))
    return float(np.mean(boundary_values) - 0.25 * np.mean(middle_values or [0.0]))


def _best_origin(
    projection: np.ndarray,
    pitch: float,
    start: int,
    stop: int,
    minimum_lines: int,
) -> tuple[float, float]:
    best_origin = float(start)
    best_score = float("-inf")
    for origin in np.arange(start, stop + 0.01, 0.5):
        score = _axis_score(projection, float(origin), pitch, minimum_lines)
        if score > best_score:
            best_origin = float(origin)
            best_score = score
    return best_origin, best_score


def _rewind_origin(
    projection: np.ndarray,
    origin: float,
    pitch: float,
    minimum: float,
) -> float:
    """Move an equivalent lattice origin back to the first continuous grid line."""
    forward_strengths: list[float] = []
    for position in np.arange(origin, len(projection), pitch)[:6]:
        index = int(round(position))
        forward_strengths.append(float(np.max(projection[max(0, index - 2) : index + 3])))
    reference = float(np.median(forward_strengths or [0.0]))
    threshold = max(float(np.mean(projection)) * 1.8, reference * 0.45)
    current = origin
    while current - pitch >= minimum:
        previous = current - pitch
        index = int(round(previous))
        strength = float(np.max(projection[max(0, index - 2) : min(len(projection), index + 3)]))
        if strength < threshold:
            break
        current = previous
    return current


def detect_grid(image_rgb: np.ndarray, pitch_hint: float | None = None) -> GridGeometry:
    """Detect the visible grid without retaining any world-map coordinates."""
    height, width = image_rgb.shape[:2]
    if width < 300 or height < 220:
        raise ValueError("游戏窗口过小，无法可靠识别棋盘")

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    lower_start = max(60, int(height * 0.18))
    vertical_projection = np.mean(grad_x[lower_start:, :], axis=0)
    horizontal_projection = np.mean(grad_y, axis=1)

    best: tuple[float, float, float, float] | None = None
    y_start = max(50, int(height * 0.12))
    y_stop = min(height - 100, int(height * 0.42))
    x_stop = min(35, max(12, int(width * 0.06)))
    pitches = [pitch_hint] if pitch_hint is not None else np.arange(29.0, 38.01, 0.1)
    for pitch in pitches:
        assert pitch is not None
        origin_y, score_y = _best_origin(horizontal_projection, pitch, y_start, y_stop, 7)
        origin_x, score_x = _best_origin(vertical_projection, pitch, 0, x_stop, 10)
        score = score_y + score_x
        if best is None or score > best[0]:
            best = (score, float(pitch), origin_x, origin_y)

    if best is None:
        raise ValueError("没有找到规则网格")
    _, pitch, origin_x, origin_y = best

    # Snap to the strongest edge near the fitted origin.
    x_window = range(max(0, round(origin_x) - 3), min(width, round(origin_x) + 4))
    y_window = range(max(0, round(origin_y) - 3), min(height, round(origin_y) + 4))
    origin_x = float(max(x_window, key=lambda index: vertical_projection[index]))
    origin_y = float(max(y_window, key=lambda index: horizontal_projection[index]))
    origin_x = _rewind_origin(vertical_projection, origin_x, pitch, max(4.0, width * 0.004))
    origin_y = _rewind_origin(horizontal_projection, origin_y, pitch, height * 0.18)

    edge_tolerance = max(3.0, pitch * 0.08)
    full_columns = int((width - origin_x + edge_tolerance) // pitch)
    full_rows = int((height - origin_y + edge_tolerance) // pitch)
    if full_columns < 8 or full_rows < 5:
        raise ValueError("可见完整格子数量不足")
    return GridGeometry(origin_x, origin_y, pitch, full_columns, full_rows)
