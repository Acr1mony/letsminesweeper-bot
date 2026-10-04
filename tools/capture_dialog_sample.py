"""Capture real in-game dialog frames for detector diagnosis.

Polls the game window; whenever a loose cyan-dialog signal appears, saves
frames and prints what each detector sees. Usage:
    python tools/capture_dialog_sample.py --seconds 180
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minesweeper.capture import capture_window  # noqa: E402
from minesweeper.dialogs import (  # noqa: E402
    detect_cyan_dialog,
    detect_death_dialog,
    detect_revive_button,
    detect_welfare_confirm,
)


def cyan_count(image: np.ndarray) -> int:
    height, width = image.shape[:2]
    region = image[int(height * 0.30) : int(height * 0.80), int(width * 0.20) : int(width * 0.80)]
    red, green, blue = (
        region[:, :, 0].astype(int),
        region[:, :, 1].astype(int),
        region[:, :, 2].astype(int),
    )
    mask = (blue - red >= 6) & (blue >= 225) & (green >= 225) & (red >= 180)
    return int(mask.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=180)
    parser.add_argument("--out-dir", default="samples")
    args = parser.parse_args()

    deadline = time.monotonic() + args.seconds
    captured = 0
    last_capture = 0.0
    print(f"监控 {args.seconds} 秒，等待对话框出现…", flush=True)
    while time.monotonic() < deadline and captured < 10:
        try:
            frame = capture_window("Let's Minesweeper")
        except Exception:
            time.sleep(0.25)
            continue
        image = np.array(frame.image.convert("RGB"))
        count = cyan_count(image)
        now = time.monotonic()
        if count > 800 and now - last_capture >= 0.8:
            captured += 1
            last_capture = now
            path = f"{args.out_dir}/live-dialog-{captured:02d}.png"
            frame.image.save(path)
            print(
                f"[{captured}] cyan={count} 保存 {path} | "
                f"cyan_dialog={detect_cyan_dialog(image)} "
                f"death={detect_death_dialog(image)} "
                f"green={detect_revive_button(image)} "
                f"amber={detect_welfare_confirm(image)}",
                flush=True,
            )
        time.sleep(0.22)
    print(f"结束，共抓取 {captured} 帧。", flush=True)


if __name__ == "__main__":
    main()
