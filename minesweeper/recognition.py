from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .geometry import GridGeometry


class CellKind(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    NUMBER = "number"
    MINE_MARK = "mine_mark"
    OPEN_MINE = "open_mine"
    TREASURE = "treasure"
    UNRECOGNIZED = "unrecognized"


@dataclass(frozen=True)
class Cell:
    row: int
    column: int
    kind: CellKind
    number: int | None = None
    confidence: float = 1.0
    # Only used by treasure cells: "open" or "closed", distinguishing the
    # two chest appearances so automation can pick the ones worth clicking.
    variant: str | None = None


DIGIT_COLORS = {
    1: np.array((86, 105, 221)),
    2: np.array((14, 166, 64)),
    3: np.array((239, 52, 56)),
    4: np.array((57, 51, 167)),
    5: np.array((159, 49, 47)),
    6: np.array((16, 133, 130)),
}


def _ratio_near(pixels: np.ndarray, color: np.ndarray, tolerance: int) -> float:
    delta = np.max(np.abs(pixels.astype(np.int16) - color), axis=2)
    return float(np.mean(delta <= tolerance))


def classify_cell(cell_rgb: np.ndarray, row: int, column: int) -> Cell:
    height, width = cell_rgb.shape[:2]
    if height < 18 or width < 18:
        return Cell(row, column, CellKind.UNRECOGNIZED, confidence=0.0)

    # Ignore bevels and dotted grid borders. The central area distinguishes an
    # untouched gray tile from a gray tile carrying any user-selected graphic.
    margin_x = max(4, width // 6)
    margin_y = max(4, height // 6)
    center = cell_rgb[margin_y : height - margin_y, margin_x : width - margin_x]
    gray_ratio = _ratio_near(center, np.array((169, 169, 169)), 5)
    open_ratio = _ratio_near(center, np.array((240, 240, 240)), 6)

    if gray_ratio >= 0.82:
        return Cell(row, column, CellKind.CLOSED, confidence=min(1.0, gray_ratio))
    if gray_ratio >= 0.08 and open_ratio < 0.35:
        # The actual artwork is deliberately irrelevant. In this game any
        # graphic drawn on a gray tile is a guaranteed mine marker.
        confidence = min(1.0, 0.65 + (0.82 - gray_ratio))
        return Cell(row, column, CellKind.MINE_MARK, confidence=confidence)

    black_ratio = float(np.mean(np.max(center, axis=2) < 45))
    if gray_ratio < 0.08 and open_ratio >= 0.15 and black_ratio >= 0.20:
        confidence = min(1.0, 0.55 + black_ratio)
        return Cell(row, column, CellKind.OPEN_MINE, confidence=confidence)

    channels = center.astype(np.int16)
    red, green, blue = channels[:, :, 0], channels[:, :, 1], channels[:, :, 2]
    gold = (
        (red > 140)
        & (green > 65)
        & (green < 210)
        & (blue < 110)
        & (red > green * 1.10)
    )
    brown = (
        (red >= 45)
        & (red <= 150)
        & (green >= 25)
        & (green <= 110)
        & (blue < 90)
        & (red > green * 1.10)
        & (green > blue * 1.05)
    )
    orange = (
        (red > 170)
        & (green > 55)
        & (green < 175)
        & (blue < 85)
        & (red > green * 1.25)
    )
    gold_ratio = float(np.mean(gold))
    brown_ratio = float(np.mean(brown))
    orange_ratio = float(np.mean(orange))
    is_open_chest = gold_ratio >= 0.08 and brown_ratio >= 0.04
    is_closed_chest = gold_ratio >= 0.12 and orange_ratio >= 0.08
    if open_ratio >= 0.15 and (is_open_chest or is_closed_chest):
        confidence = min(1.0, 0.55 + gold_ratio + max(brown_ratio, orange_ratio) * 0.5)
        # The two chest arts separate cleanly on brown (open ≈ 0.19,
        # closed ≈ 0.04), which decides whether the chest still wants a click.
        variant = "open" if brown_ratio >= 0.10 else "closed"
        return Cell(
            row,
            column,
            CellKind.TREASURE,
            number=0,
            confidence=confidence,
            variant=variant,
        )

    best_number = 0
    best_ratio = 0.0
    for number, color in DIGIT_COLORS.items():
        ratio = _ratio_near(center, color, 8)
        if ratio > best_ratio:
            best_number = number
            best_ratio = ratio
    if best_ratio >= 0.055:
        return Cell(row, column, CellKind.NUMBER, number=best_number, confidence=min(1.0, best_ratio * 6))

    non_background = 1.0 - open_ratio
    saturation = np.max(center, axis=2).astype(np.int16) - np.min(center, axis=2).astype(np.int16)
    saturated_ratio = float(np.mean(saturation > 20))
    if open_ratio >= 0.88 or (open_ratio >= 0.70 and saturated_ratio < 0.03) or (non_background < 0.08 and saturated_ratio < 0.03):
        return Cell(row, column, CellKind.OPEN, number=0, confidence=open_ratio)

    # Most likely a not-yet-sampled digit 6–8. It is never promoted to a mine
    # because confirmed mine graphics must retain the gray-tile background.
    return Cell(row, column, CellKind.UNRECOGNIZED, confidence=0.0)


def recognize_grid(image_rgb: np.ndarray, geometry: GridGeometry) -> list[list[Cell]]:
    grid: list[list[Cell]] = []
    for row in range(geometry.full_rows):
        cells: list[Cell] = []
        for column in range(geometry.full_columns):
            x0, y0, x1, y1 = geometry.bounds(row, column)
            crop = image_rgb[max(0, y0) : min(image_rgb.shape[0], y1), max(0, x0) : min(image_rgb.shape[1], x1)]
            cells.append(classify_cell(crop, row, column))
        grid.append(cells)
    return grid
