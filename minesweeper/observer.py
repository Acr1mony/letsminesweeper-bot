from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .capture import CaptureFrame, capture_window
from .geometry import GridGeometry, detect_grid
from .recognition import Cell, CellKind, recognize_grid
from .solver import SolveResult, solve_local


@dataclass(frozen=True)
class Observation:
    frame: CaptureFrame
    geometry: GridGeometry
    grid: list[list[Cell]]
    result: SolveResult
    annotated: Image.Image


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/seguisym.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def render_overlay(
    image: Image.Image,
    geometry: GridGeometry,
    grid: list[list[Cell]],
    result: SolveResult,
) -> Image.Image:
    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    for row in grid:
        for cell in row:
            if cell.kind not in {
                CellKind.MINE_MARK,
                CellKind.OPEN_MINE,
                CellKind.TREASURE,
                CellKind.UNRECOGNIZED,
            }:
                continue
            x0, y0, x1, y1 = geometry.bounds(cell.row, cell.column)
            if cell.kind == CellKind.MINE_MARK:
                color = (40, 90, 74, 210)
            elif cell.kind == CellKind.OPEN_MINE:
                color = (170, 35, 35, 235)
            elif cell.kind == CellKind.TREASURE:
                color = (224, 156, 30, 235)
            else:
                color = (142, 72, 152, 225)
            draw.rectangle((x0 + 2, y0 + 2, x1 - 2, y1 - 2), outline=color, width=2)

    for row, column in result.safe:
        x0, y0, x1, y1 = geometry.bounds(row, column)
        draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), fill=(46, 134, 171, 75), outline=(46, 134, 171, 255), width=3)
    for row, column in result.mines:
        x0, y0, x1, y1 = geometry.bounds(row, column)
        draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), fill=(217, 87, 78, 70), outline=(217, 87, 78, 255), width=3)

    if result.contradictions:
        font = _font(18)
        draw.rounded_rectangle((12, 40, min(base.width - 12, 520), 72), radius=7, fill=(217, 87, 78, 235))
        draw.text((23, 46), "识别存在矛盾，已禁止给出操作建议", font=font, fill=(255, 255, 255, 255))
    return Image.alpha_composite(base, layer).convert("RGB")


def observe_image(image: Image.Image, frame: CaptureFrame | None = None) -> Observation:
    rgb = np.array(image.convert("RGB"))
    geometry = detect_grid(rgb)
    grid = recognize_grid(rgb, geometry)
    result = solve_local(grid)
    if result.contradictions:
        result = SolveResult(frozenset(), frozenset(), result.constraints, result.contradictions)
    actual_frame = frame or CaptureFrame(image.convert("RGB"), (0, 0, image.width, image.height), False)
    annotated = render_overlay(image, geometry, grid, result)
    return Observation(actual_frame, geometry, grid, result, annotated)


def observe_window(title: str = "Let's Minesweeper") -> Observation:
    frame = capture_window(title)
    return observe_image(frame.image, frame)


def board_motion_ratio(previous_rgb: np.ndarray, current_rgb: np.ndarray, board_top: int) -> float:
    if previous_rgb.shape != current_rgb.shape:
        return 1.0
    top = max(0, min(board_top, previous_rgb.shape[0] - 1))
    previous_board = previous_rgb[top:].astype(np.int16)
    current_board = current_rgb[top:].astype(np.int16)
    difference = np.max(np.abs(previous_board - current_board), axis=2)
    return float(np.mean(difference > 10))


class WindowObserver:
    """Fast repeated observer that caches geometry, never board contents."""

    def __init__(self, title: str = "Let's Minesweeper", initial: Observation | None = None) -> None:
        self.title = title
        self.geometry = initial.geometry if initial is not None else None
        self.image_size = initial.frame.image.size if initial is not None else None

    def __call__(self) -> Observation:
        frame = capture_window(self.title)
        return self._from_frame(frame, force_redetect=False)

    def raw_frame(self) -> CaptureFrame:
        """Fresh window capture without any grid recognition."""
        return capture_window(self.title)

    def _from_frame(self, frame: CaptureFrame, force_redetect: bool) -> Observation:
        rgb = np.array(frame.image.convert("RGB"))
        size_changed = self.image_size != frame.image.size
        if self.geometry is None or size_changed or force_redetect:
            pitch_hint = None if self.geometry is None or size_changed else self.geometry.pitch
            self.geometry = detect_grid(rgb, pitch_hint=pitch_hint)
            self.image_size = frame.image.size
        grid = recognize_grid(rgb, self.geometry)
        result = solve_local(grid)
        if result.contradictions:
            result = SolveResult(frozenset(), frozenset(), result.constraints, result.contradictions)
        annotated = render_overlay(frame.image, self.geometry, grid, result)
        return Observation(frame, self.geometry, grid, result, annotated)

    def wait_for_stable(
        self,
        stop_event: threading.Event,
        *,
        force_redetect: bool,
        timeout: float = 3.0,
        initial_delay: float = 0.0,
        interval: float = 0.09,
        stable_pairs: int = 3,
        motion_threshold: float = 0.002,
    ) -> Observation:
        """Wait for animation/dragging to stop before any grid recognition."""
        if initial_delay > 0 and stop_event.wait(initial_delay):
            raise RuntimeError("自动运行已停止")
        deadline = time.monotonic() + timeout
        previous_frame = capture_window(self.title)
        previous_rgb = np.array(previous_frame.image.convert("RGB"))
        stable_count = 0
        while time.monotonic() < deadline:
            if stop_event.wait(interval):
                raise RuntimeError("自动运行已停止")
            current_frame = capture_window(self.title)
            current_rgb = np.array(current_frame.image.convert("RGB"))
            board_top = (
                round(self.geometry.origin_y)
                if self.geometry is not None
                else round(current_rgb.shape[0] * 0.18)
            )
            motion = board_motion_ratio(previous_rgb, current_rgb, board_top)
            stable_count = stable_count + 1 if motion <= motion_threshold else 0
            if stable_count >= stable_pairs:
                try:
                    return self._from_frame(current_frame, force_redetect=force_redetect)
                except ValueError:
                    stable_count = 0
            previous_rgb = current_rgb
        raise RuntimeError("拖动或动画在 3 秒内没有稳定，已暂停自动运行")
