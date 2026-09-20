from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


BOARD_X = 11.0
BOARD_Y = 98.0
CELL_PITCH = 100.0 / 3.0

STATE_COLORS = {
    "1": np.array((86, 105, 221)),
    "2": np.array((14, 166, 64)),
    "3": np.array((239, 52, 56)),
    "4": np.array((57, 51, 167)),
    "5": np.array((159, 49, 47)),
    "mine_mark": np.array((243, 153, 153)),
}


def color_count(pixels: np.ndarray, color: np.ndarray, tolerance: int = 5) -> int:
    return int(np.sum(np.max(np.abs(pixels.astype(np.int16) - color), axis=2) <= tolerance))


def classify(cell: np.ndarray) -> str:
    counts = {name: color_count(cell, color) for name, color in STATE_COLORS.items()}
    if counts["mine_mark"] >= 20:
        return "M"

    digit, digit_count = max(
        ((name, count) for name, count in counts.items() if name != "mine_mark"),
        key=lambda item: item[1],
    )
    if digit_count >= 18:
        return digit

    closed_count = color_count(cell, np.array((169, 169, 169)), tolerance=3)
    if closed_count >= max(30, cell.shape[0] * cell.shape[1] // 4):
        return "?"
    return "0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    args = parser.parse_args()

    image = Image.open(args.input).convert("RGB")
    rgb = np.array(image)
    height, width = rgb.shape[:2]
    visible_cols = math.ceil((width - BOARD_X) / CELL_PITCH)
    visible_rows = math.ceil((height - BOARD_Y) / CELL_PITCH)
    full_cols = math.floor((width - BOARD_X) / CELL_PITCH)
    full_rows = math.floor((height - BOARD_Y) / CELL_PITCH)

    grid: list[list[str]] = []
    overlay = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for row in range(visible_rows):
        states: list[str] = []
        for col in range(visible_cols):
            geometric_right = BOARD_X + (col + 1) * CELL_PITCH
            geometric_bottom = BOARD_Y + (row + 1) * CELL_PITCH
            x0 = max(0, round(BOARD_X + col * CELL_PITCH) + 4)
            y0 = max(0, round(BOARD_Y + row * CELL_PITCH) + 4)
            x1 = min(width, round(BOARD_X + (col + 1) * CELL_PITCH) - 4)
            y1 = min(height, round(BOARD_Y + (row + 1) * CELL_PITCH) - 4)
            is_partial = geometric_right > width or geometric_bottom > height
            state = (
                classify(rgb[y0:y1, x0:x1])
                if not is_partial and x1 > x0 and y1 > y0
                else "partial"
            )
            states.append(state)

            left = round(BOARD_X + col * CELL_PITCH)
            top = round(BOARD_Y + row * CELL_PITCH)
            right = min(width - 1, round(BOARD_X + (col + 1) * CELL_PITCH))
            bottom = min(height - 1, round(BOARD_Y + (row + 1) * CELL_PITCH))
            cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 255), 1)
            cv2.putText(
                overlay,
                f"{row},{col}",
                (left + 2, min(bottom - 2, top + 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.22,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        grid.append(states)

    counts = Counter(state for row in grid for state in row)
    report = {
        "source": str(args.input.resolve()),
        "window_size_px": [width, height],
        "board_origin_px": [BOARD_X, BOARD_Y],
        "cell_pitch_px": CELL_PITCH,
        "full_visible_columns": full_cols,
        "full_visible_rows": full_rows,
        "partially_visible_columns": visible_cols - full_cols,
        "partially_visible_rows": visible_rows - full_rows,
        "state_counts": dict(sorted(counts.items())),
        "legend": {"?": "closed", "0": "open_blank", "M": "mine_mark"},
        "grid": grid,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.overlay.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    cv2.imwrite(str(args.overlay), overlay)
    print(json.dumps({key: value for key, value in report.items() if key != "grid"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
