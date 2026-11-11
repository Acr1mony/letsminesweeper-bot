from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


# The pitch search covers any sane desktop zoom level; the previous 29-38 px
# window only fitted the default game size and refused every other scale.
PITCH_MIN = 20.0
PITCH_MAX = 72.0
PITCH_COARSE_STEP = 0.5
PITCH_FINE_STEP = 0.05
PITCH_FINE_RADIUS = 0.3
PITCH_FUNDAMENTAL_TOLERANCE = 0.08
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 160
MIN_COLUMNS = 6
MIN_ROWS = 4


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


def _line_budget(extent: int, pitch: float, preferred: int) -> int:
    """Grid lines that must fit in `extent` before a lattice is considered."""
    return max(4, min(preferred, int(extent / pitch) - 2))


def _fit_lattice(
    vertical_projection: np.ndarray,
    horizontal_projection: np.ndarray,
    pitches: np.ndarray,
    y_range: tuple[float, float],
    x_range: tuple[float, float],
) -> list[tuple[float, float, float, float]]:
    """Score every candidate pitch and return the lattices, strongest first."""
    width = len(vertical_projection)
    height = len(horizontal_projection)
    results: list[tuple[float, float, float, float]] = []
    for pitch in pitches:
        min_x_lines = _line_budget(width, pitch, 10)
        min_y_lines = _line_budget(height, pitch, 7)
        origin_y, score_y = _best_origin(
            horizontal_projection, float(pitch), y_range[0], y_range[1], min_y_lines
        )
        origin_x, score_x = _best_origin(
            vertical_projection, float(pitch), x_range[0], x_range[1], min_x_lines
        )
        results.append((score_y + score_x, float(pitch), origin_x, origin_y))
    results.sort(key=lambda result: result[0], reverse=True)
    return results


def _select_fundamental(
    results: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float]:
    """Prefer the smallest lattice that explains the image almost as well.

    Every multiple of the true pitch also lands on real grid lines, so a
    harmonic can outscore the fundamental. Choosing the smallest pitch whose
    score is within tolerance of the best one recovers the real cell size.
    """
    best_score = results[0][0]
    threshold = best_score - PITCH_FUNDAMENTAL_TOLERANCE * max(1.0, abs(best_score))
    return min((result for result in results if result[0] >= threshold), key=lambda result: result[1])


def detect_grid(image_rgb: np.ndarray, pitch_hint: float | None = None) -> GridGeometry:
    """Detect the visible grid without retaining any world-map coordinates."""
    height, width = image_rgb.shape[:2]
    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        raise ValueError(
            f"游戏窗口过小，无法可靠识别棋盘（至少需要 {MIN_IMAGE_WIDTH}×{MIN_IMAGE_HEIGHT} 像素）"
        )

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    lower_start = max(60, int(height * 0.18))
    vertical_projection = np.mean(grad_x[lower_start:, :], axis=0)
    horizontal_projection = np.mean(grad_y, axis=1)

    y_start = max(50, int(height * 0.12))
    y_stop = min(max(y_start + 10, height - 100), int(height * 0.42))
    x_stop = min(35, max(12, int(width * 0.06)))

    # Coarse sweep over the whole supported pitch range, then a fine sweep
    # around the winner so unusual window scales still calibrate.
    if pitch_hint is not None:
        # A cached pitch is reused as-is: the game zoom does not change between
        # frames, so a full sweep every cycle would only waste time.
        results = _fit_lattice(
            vertical_projection,
            horizontal_projection,
            np.array([float(pitch_hint)]),
            (y_start, y_stop),
            (0, x_stop),
        )
        _, pitch, origin_x, origin_y = results[0]
    else:
        results = _fit_lattice(
            vertical_projection,
            horizontal_projection,
            np.arange(PITCH_MIN, PITCH_MAX + 1e-6, PITCH_COARSE_STEP),
            (y_start, y_stop),
            (0, x_stop),
        )
        _, pitch, origin_x, origin_y = _select_fundamental(results)

        fine_pitches = np.arange(
            pitch - PITCH_FINE_RADIUS, pitch + PITCH_FINE_RADIUS + 1e-6, PITCH_FINE_STEP
        )
        fine_pitches = fine_pitches[(fine_pitches >= PITCH_MIN) & (fine_pitches <= PITCH_MAX)]
        if len(fine_pitches) > 0:
            refined = _fit_lattice(
                vertical_projection,
                horizontal_projection,
                fine_pitches,
                (y_start, y_stop),
                (0, x_stop),
            )
            _, pitch, origin_x, origin_y = refined[0]

    if pitch <= 0:
        raise ValueError("没有找到规则网格")

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
    if full_columns < MIN_COLUMNS or full_rows < MIN_ROWS:
        raise ValueError(
            f"可见完整格子数量不足（至少需要 {MIN_COLUMNS}×{MIN_ROWS} 格）"
        )
    return GridGeometry(origin_x, origin_y, pitch, full_columns, full_rows)
