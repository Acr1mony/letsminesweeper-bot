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


# Colors straight out of the game's own 84x84 number sprites (flat color with
# the glyph in alpha), so the pixels on screen match these values exactly and no
# runtime tinting has to be guessed at.
DIGIT_COLORS = {
    1: np.array((86, 105, 221)),
    2: np.array((14, 166, 64)),
    3: np.array((239, 52, 56)),
    4: np.array((57, 51, 167)),
    5: np.array((159, 49, 47)),
    6: np.array((16, 133, 130)),
    7: np.array((51, 50, 50)),
    8: np.array((120, 120, 131)),
}
# Digits are anti-aliased, and the smaller the pitch the more of a glyph is made
# of blended edge pixels: at a 28px pitch a "4" keeps 0.05 of its core color with
# a tolerance of 8 but 0.10 with 12. No two digit colors are closer than 54 on
# any channel, so 12 cannot make one digit look like another.
DIGIT_COLOR_TOLERANCE = 12
DIGIT_MIN_RATIO = 0.055

TILE_GRAY = np.array((169, 169, 169))
OPEN_GRAY = np.array((240, 240, 240))
# An untouched tile measures at least 0.95 gray on real frames, while every
# marker graphic covers 0.26 or more of the tile center.
CLOSED_GRAY_RATIO = 0.82
# Graphics smaller than this count as noise on an untouched tile (the measured
# floor on real boards is 0.048 art).
ART_ON_CLOSED_TILE_RATIO = 0.10
# A tile that has not been opened keeps its gray bevel all around: that band
# measures 0.71-1.00 on real frames and at most 0.05 on every opened tile, so it
# still proves "closed" when a marker skin covers nearly the whole tile.
UNOPENED_BAND_RATIO = 0.25


def _ratio_near(pixels: np.ndarray, color: np.ndarray, tolerance: int) -> float:
    delta = np.max(np.abs(pixels.astype(np.int16) - color), axis=2)
    return float(np.mean(delta <= tolerance))


def _band_gray_ratio(cell_rgb: np.ndarray, color: np.ndarray, tolerance: int = 12) -> float:
    """Share of the tile's border band that still shows the tile background.

    The band skips the dotted grid border and the middle of the tile, leaving
    the bevel an unopened tile keeps even underneath a large marker graphic.
    """

    height, width = cell_rgb.shape[:2]
    inset_x = max(2, width // 16)
    inset_y = max(2, height // 16)
    band = cell_rgb[inset_y : height - inset_y, inset_x : width - inset_x]
    band_height, band_width = band.shape[:2]
    if band_height < 4 or band_width < 4:
        return 0.0
    margin_y = int(band_height * 0.30)
    margin_x = int(band_width * 0.30)
    mask = np.ones((band_height, band_width), dtype=bool)
    mask[margin_y : band_height - margin_y, margin_x : band_width - margin_x] = False
    pixels = band[mask]
    delta = np.max(np.abs(pixels.astype(np.int16) - color), axis=1)
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
    gray_ratio = _ratio_near(center, TILE_GRAY, 5)
    open_ratio = _ratio_near(center, OPEN_GRAY, 6)
    art_ratio = max(0.0, 1.0 - gray_ratio - open_ratio)

    if gray_ratio >= CLOSED_GRAY_RATIO and art_ratio < ART_ON_CLOSED_TILE_RATIO:
        return Cell(row, column, CellKind.CLOSED, confidence=min(1.0, gray_ratio))
    # The actual artwork is deliberately irrelevant. Any graphic on a still
    # closed tile is a guaranteed mine marker, whether it is a small flag that
    # leaves most of the tile gray or a skin that covers almost all of it.
    unopened = _band_gray_ratio(cell_rgb, TILE_GRAY) >= UNOPENED_BAND_RATIO
    if open_ratio < 0.35 and (unopened or gray_ratio >= 0.08):
        confidence = min(1.0, 0.65 + max(0.0, 0.82 - gray_ratio))
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
        ratio = _ratio_near(center, color, DIGIT_COLOR_TOLERANCE)
        if ratio > best_ratio:
            best_number = number
            best_ratio = ratio
    if best_ratio >= DIGIT_MIN_RATIO:
        return Cell(row, column, CellKind.NUMBER, number=best_number, confidence=min(1.0, best_ratio * 6))

    non_background = 1.0 - open_ratio
    saturation = np.max(center, axis=2).astype(np.int16) - np.min(center, axis=2).astype(np.int16)
    saturated_ratio = float(np.mean(saturation > 20))
    if open_ratio >= 0.88 or (open_ratio >= 0.70 and saturated_ratio < 0.03) or (non_background < 0.08 and saturated_ratio < 0.03):
        return Cell(row, column, CellKind.OPEN, number=0, confidence=open_ratio)

    # Most likely a digit or marker skin whose color was never sampled. It is
    # never promoted to a mine because a marker must keep the gray tile.
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
