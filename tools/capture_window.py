from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path

from PIL import Image, ImageGrab


user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]


def window_for_pid(pid: int) -> int:
    matches: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def callback(hwnd: int, _: int) -> bool:
        window_pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(window_pid))
        if window_pid.value == pid and user32.IsWindowVisible(hwnd):
            length = user32.GetWindowTextLengthW(hwnd)
            if length:
                matches.append(hwnd)
        return True

    user32.EnumWindows(callback, 0)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one visible top-level window for PID {pid}, found {len(matches)}")
    return matches[0]


def capture(hwnd: int) -> tuple[Image.Image, tuple[int, int, int, int]]:
    rect = wintypes.RECT()
    if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        raise ctypes.WinError(ctypes.get_last_error())
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if rect.left <= -30000 or rect.top <= -30000:
        raise RuntimeError("The target window is minimized; restore it before capturing")
    window_dc = user32.GetWindowDC(hwnd)
    memory_dc = gdi32.CreateCompatibleDC(window_dc)
    bitmap = gdi32.CreateCompatibleBitmap(window_dc, width, height)
    previous = gdi32.SelectObject(memory_dc, bitmap)
    try:
        if not user32.PrintWindow(hwnd, memory_dc, 2):
            return ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom)), (
                rect.left,
                rect.top,
                rect.right,
                rect.bottom,
            )
        info = BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = 0
        buffer = ctypes.create_string_buffer(width * height * 4)
        rows = gdi32.GetDIBits(memory_dc, bitmap, 0, height, buffer, ctypes.byref(info), 0)
        if rows != height:
            raise RuntimeError(f"GetDIBits returned {rows} of {height} rows")
        image = Image.frombuffer("RGB", (width, height), buffer, "raw", "BGRX", 0, 1).copy()
        return image, (rect.left, rect.top, rect.right, rect.bottom)
    finally:
        gdi32.SelectObject(memory_dc, previous)
        gdi32.DeleteObject(bitmap)
        gdi32.DeleteDC(memory_dc)
        user32.ReleaseDC(hwnd, window_dc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    ctypes.windll.shcore.SetProcessDpiAwareness(2)
    hwnd = window_for_pid(args.pid)
    image, rect = capture(hwnd)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"hwnd={hwnd} rect={rect} size={image.size} output={args.output.resolve()}")


if __name__ == "__main__":
    main()
