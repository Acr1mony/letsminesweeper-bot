from __future__ import annotations

import ctypes
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass

from .capture import find_window


user32 = ctypes.WinDLL("user32", use_last_error=True)

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
VK_F8 = 0x77


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.WPARAM),
    ]


class INPUTUNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("union",)
    _fields_ = [("type", wintypes.DWORD), ("union", INPUTUNION)]


@dataclass(frozen=True)
class BatchPoints:
    marks: tuple[tuple[int, int], ...]
    safe: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class BatchExecution:
    flagged: int = 0
    opened: int = 0

    @property
    def total(self) -> int:
        return self.flagged + self.opened


def _window_rect(hwnd: int) -> tuple[int, int, int, int]:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    return rect.left, rect.top, rect.right, rect.bottom


def _send_button(flags: int) -> None:
    event = INPUT(type=0, mi=MOUSEINPUT(0, 0, 0, flags, 0, 0))
    sent = user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
    if sent != 1:
        raise ctypes.WinError(ctypes.get_last_error())


def emergency_pressed() -> bool:
    return bool(user32.GetAsyncKeyState(VK_F8) & 0x8000)


class WindowInputController:
    def __init__(
        self,
        title: str = "Let's Minesweeper",
        click_interval: float = 0.075,
        press_duration: float = 0.020,
        move_settle: float = 0.012,
    ) -> None:
        self.title = title
        self.click_interval = click_interval
        self.press_duration = press_duration
        self.move_settle = move_settle

    def activate_target(self) -> tuple[int, tuple[int, int, int, int]]:
        hwnd = find_window(self.title)
        current_rect = _window_rect(hwnd)
        if not user32.SetForegroundWindow(hwnd):
            raise RuntimeError("无法激活游戏窗口")
        time.sleep(0.04)
        if user32.GetForegroundWindow() != hwnd:
            raise RuntimeError("游戏窗口没有获得焦点")
        return hwnd, current_rect

    def _prepare(self, expected_rect: tuple[int, int, int, int]) -> tuple[int, tuple[int, int, int, int]]:
        hwnd, current_rect = self.activate_target()
        if any(abs(left - right) > 2 for left, right in zip(current_rect, expected_rect)):
            raise RuntimeError("游戏窗口在识别后发生了移动或缩放，已取消本批操作")
        return hwnd, current_rect

    @staticmethod
    def _screen_point(rect: tuple[int, int, int, int], point: tuple[int, int]) -> tuple[int, int]:
        left, top, right, bottom = rect
        x = left + point[0]
        y = top + point[1]
        if not (left + 2 <= x < right - 2 and top + 2 <= y < bottom - 2):
            raise RuntimeError("目标坐标超出游戏窗口")
        return x, y

    def click_batch(
        self,
        expected_rect: tuple[int, int, int, int],
        points: BatchPoints,
        stop_event: threading.Event,
    ) -> BatchExecution:
        hwnd, rect = self._prepare(expected_rect)
        flagged = 0
        opened = 0
        for button, positions in (("right", points.marks), ("left", points.safe)):
            down = MOUSEEVENTF_RIGHTDOWN if button == "right" else MOUSEEVENTF_LEFTDOWN
            up = MOUSEEVENTF_RIGHTUP if button == "right" else MOUSEEVENTF_LEFTUP
            for point in positions:
                if stop_event.is_set() or emergency_pressed():
                    stop_event.set()
                    return BatchExecution(flagged=flagged, opened=opened)
                if user32.GetForegroundWindow() != hwnd:
                    raise RuntimeError("游戏窗口在批量操作中失去焦点")
                x, y = self._screen_point(rect, point)
                if not user32.SetCursorPos(x, y):
                    raise ctypes.WinError(ctypes.get_last_error())
                time.sleep(self.move_settle)
                _send_button(down)
                time.sleep(self.press_duration)
                _send_button(up)
                if button == "right":
                    flagged += 1
                else:
                    opened += 1
                if stop_event.wait(self.click_interval):
                    return BatchExecution(flagged=flagged, opened=opened)
        return BatchExecution(flagged=flagged, opened=opened)

    def drag(
        self,
        expected_rect: tuple[int, int, int, int],
        start: tuple[int, int],
        end: tuple[int, int],
        stop_event: threading.Event,
        duration: float = 0.20,
    ) -> None:
        _, rect = self._prepare(expected_rect)
        start_x, start_y = self._screen_point(rect, start)
        end_x, end_y = self._screen_point(rect, end)
        if not user32.SetCursorPos(start_x, start_y):
            raise ctypes.WinError(ctypes.get_last_error())
        time.sleep(self.move_settle)
        _send_button(MOUSEEVENTF_LEFTDOWN)
        try:
            steps = 12
            for index in range(1, steps + 1):
                if stop_event.is_set() or emergency_pressed():
                    stop_event.set()
                    return
                ratio = index / steps
                x = round(start_x + (end_x - start_x) * ratio)
                y = round(start_y + (end_y - start_y) * ratio)
                user32.SetCursorPos(x, y)
                time.sleep(duration / steps)
        finally:
            _send_button(MOUSEEVENTF_LEFTUP)
