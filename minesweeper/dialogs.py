from __future__ import annotations

import numpy as np

# Reference colours sampled from the game's own dialog screenshots.
# The dialog tint carries a cyan cast that plain white board tiles lack, so
# the two stay separable on every sampled frame.
DIALOG_RGB = (228, 247, 247)
REVIVE_GREEN = (51, 239, 144)
WELFARE_AMBER = (255, 201, 62)

_MIN_DIALOG_PIXELS = 3000
_MIN_BUTTON_PIXELS = 400
_MIN_HEART_PIXELS = 200
_MIN_WELFARE_PIXELS = 800


def _channels(image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    red = image_rgb[:, :, 0].astype(np.int16)
    green = image_rgb[:, :, 1].astype(np.int16)
    blue = image_rgb[:, :, 2].astype(np.int16)
    return red, green, blue


def _central_region(image_rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    height, width = image_rgb.shape[:2]
    y0, y1 = int(height * 0.30), int(height * 0.80)
    x0, x1 = int(width * 0.20), int(width * 0.80)
    return image_rgb[y0:y1, x0:x1], x0, y0


def detect_cyan_dialog(image_rgb: np.ndarray) -> bool:
    """True when a light-cyan game dialog covers the board centre.

    All three known modals (death, welfare chest, reward) share this body
    colour, and the normal board has no large cyan region."""
    region, _, _ = _central_region(image_rgb)
    if region.size == 0:
        return False
    red, green, blue = _channels(region)
    mask = (blue - red >= 10) & (blue >= 235) & (green >= 235) & (red >= 200)
    return int(np.sum(mask)) >= _MIN_DIALOG_PIXELS


def detect_heart(image_rgb: np.ndarray) -> bool:
    """True when the death dialog's red heart sits above the button."""
    region, _, _ = _central_region(image_rgb)
    if region.size == 0:
        return False
    red, green, blue = _channels(region)
    mask = (red >= 240) & (green <= 110) & (blue >= 50) & (blue <= 140) & (red > green + 120)
    return int(np.sum(mask)) >= _MIN_HEART_PIXELS


def detect_death_dialog(image_rgb: np.ndarray) -> bool:
    """Death dialog: cyan body plus the red heart. The reward dialog also
    shows a green button but carries black text instead of a heart."""
    return detect_cyan_dialog(image_rgb) and detect_heart(image_rgb)


def detect_revive_button(image_rgb: np.ndarray) -> tuple[int, int] | None:
    """Window coordinate of a bright-green confirm button centre, or None.

    The revive button and the reward dialog's 确认 share the same green, so
    callers must disambiguate with detect_death_dialog."""
    region, offset_x, offset_y = _central_region(image_rgb)
    if region.size == 0:
        return None
    red, green, blue = _channels(region)
    mask = (
        (green >= 180)
        & (green <= 255)
        & (red <= 130)
        & (blue >= 90)
        & (blue <= 210)
        & (green > red + 60)
        & (green > blue + 30)
    )
    count = int(np.sum(mask))
    if count < _MIN_BUTTON_PIXELS:
        return None
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_width = x1 - x0 + 1
    box_height = y1 - y0 + 1
    if not (40 <= box_width <= 220 and 12 <= box_height <= 80):
        return None
    if count < 0.5 * box_width * box_height:
        return None
    return (
        offset_x + round(float(np.mean(xs))),
        offset_y + round(float(np.mean(ys))),
    )


def detect_welfare_confirm(image_rgb: np.ndarray) -> tuple[int, int] | None:
    """Window coordinate of the amber 确认 button on the welfare dialog.

    Reward badges on the dialog also contain amber pixels, so the search
    walks up from the bottom-most solid amber row; the button is the lowest
    wide bar and sits clearly below the badges."""
    height, width = image_rgb.shape[:2]
    y0, y1 = int(height * 0.55), int(height * 0.92)
    x0, x1 = int(width * 0.25), int(width * 0.75)
    region = image_rgb[y0:y1, x0:x1]
    if region.size == 0:
        return None
    red, green, blue = _channels(region)
    mask = (red > 200) & (green > 130) & (green < 220) & (blue < 120) & (red > blue + 90)
    rows = mask.sum(axis=1)
    solid = rows >= 30
    if not solid.any():
        return None
    bottom = int(np.nonzero(solid)[0].max())
    top = bottom
    while top > 0 and solid[top - 1]:
        top -= 1
    cluster = mask[top : bottom + 1]
    count = int(cluster.sum())
    if count < _MIN_WELFARE_PIXELS:
        return None
    ys, xs = np.nonzero(cluster)
    box_width = int(xs.max()) - int(xs.min()) + 1
    box_height = bottom - top + 1
    if not (60 <= box_width <= 220 and 14 <= box_height <= 60):
        return None
    return (
        x0 + round(float(np.mean(xs))),
        y0 + top + round(float(np.mean(ys))),
    )
