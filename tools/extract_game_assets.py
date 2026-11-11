"""Extract the game's own UI artwork so the bot's recognition stays honest.

The game is an Electron/Cocos Creator build: every sprite lives in
resources/app/web-mobile/assets/.../native/ under a UUID file name. This script
finds the board-art family (the 84x84 tiles the board is drawn from), reports the
flat colors baked into each sprite and writes a catalog to docs/game-assets.md.

Run it whenever the game updates or a new marker skin shows up:

    python tools/extract_game_assets.py
    python tools/extract_game_assets.py --game-root "D:/Games/一起来扫雷"

Nothing is written outside docs/game-assets.md unless --dump is given, which also
saves the board sprites for eyeballing.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minesweeper.recognition import DIGIT_COLORS, TILE_GRAY  # noqa: E402

DEFAULT_GAME_ROOT = Path(r"G:/SteamLibrary/steamapps/common/一起来扫雷")
BOARD_MIN_SIZE = 80
BOARD_MAX_SIZE = 92
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "game-assets.md"
DIGIT_HEX = {1: "#5669DD", 2: "#0EA640", 3: "#EF3438", 4: "#3933A7", 5: "#9F312F", 6: "#108582", 7: "#333232", 8: "#787883"}


def find_sprites(game_root: Path) -> list[Path]:
    assets = game_root / "resources" / "app" / "web-mobile" / "assets"
    if not assets.is_dir():
        raise SystemExit(f"no Cocos assets under {assets}")
    return sorted(assets.rglob("*.png"))


def flatten_on_tile(path: Path) -> tuple[np.ndarray, int] | None:
    """Board sprite composited on the gray tile, the way the game draws it."""

    with Image.open(path) as image:
        image = image.convert("RGBA")
        width, height = image.size
        if width != height or not BOARD_MIN_SIZE <= width <= BOARD_MAX_SIZE:
            return None
        backdrop = Image.new("RGBA", image.size, (*TILE_GRAY.tolist(), 255))
        backdrop.alpha_composite(image)
        return np.array(backdrop.convert("RGB")), width


def flat_colors(rgb: np.ndarray, minimum: float = 0.02) -> list[tuple[tuple[int, int, int], float]]:
    """Colors carrying at least 'minimum' of the sprite, the largest share first.

    Number and marker sprites are flat-colored with the shape in alpha, so a
    sprite's palette is exactly what ends up on screen.
    """

    pixels = rgb.reshape(-1, 3).astype(np.uint32)
    keys = (pixels[:, 0] << 16) | (pixels[:, 1] << 8) | pixels[:, 2]
    counts = Counter(keys.tolist())
    total = len(keys)
    palette = []
    for key, count in counts.most_common(8):
        share = count / total
        if share < minimum:
            break
        palette.append((((key >> 16) & 0xFF, (key >> 8) & 0xFF, key & 0xFF), share))
    return palette


def center_gray_ratio(rgb: np.ndarray) -> float:
    height, width = rgb.shape[:2]
    margin_x = max(4, width // 6)
    margin_y = max(4, height // 6)
    center = rgb[margin_y : height - margin_y, margin_x : width - margin_x]
    return float(np.mean(np.max(np.abs(center.astype(np.int16) - TILE_GRAY), axis=2) <= 5))


def label_sprite(rgb: np.ndarray) -> str:
    if center_gray_ratio(rgb) >= 0.95:
        return "closed tile"
    for number, color in DIGIT_COLORS.items():
        hit = np.max(np.abs(rgb.astype(np.int16) - color), axis=2) <= 12
        if float(np.mean(hit)) >= 0.05:
            return f"number {number}"
    channels = rgb.astype(np.int16)
    red = (channels[:, :, 0] > 140) & (channels[:, :, 1] < 90) & (channels[:, :, 2] < 90)
    if float(np.mean(red)) >= 0.03:
        return "red art"
    if float(np.mean(np.max(rgb, axis=2) < 45)) >= 0.15:
        return "dark art"
    return "other art"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-root", type=Path, default=DEFAULT_GAME_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dump", type=Path, default=None, help="also save the board sprites here")
    args = parser.parse_args()

    sprites = find_sprites(args.game_root)
    rows = []
    for path in sprites:
        flattened = flatten_on_tile(path)
        if flattened is None:
            continue
        rgb, size = flattened
        rows.append(
            {
                "path": path,
                "size": size,
                "uuid": path.stem,
                "sha": hashlib.sha256(path.read_bytes()).hexdigest()[:12],
                "label": label_sprite(rgb),
                "gray": center_gray_ratio(rgb),
                "palette": flat_colors(rgb),
            }
        )
    if not rows:
        raise SystemExit(f"no {BOARD_MIN_SIZE}-{BOARD_MAX_SIZE}px square board sprites under {args.game_root}")
    print(f"{len(sprites)} sprites scanned, {len(rows)} are board art")
    print("labels:", dict(sorted(Counter(row["label"] for row in rows).items())))

    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
        for row in rows:
            name = row["label"].replace(" ", "-") + "-" + row["sha"] + ".png"
            Image.open(row["path"]).save(args.dump / name)
        print(f"board art saved to {args.dump}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(rows, args.game_root), encoding="utf-8")
    print(f"catalog written to {args.out}")
    return 0


def render(rows: list[dict], game_root: Path) -> str:
    btn = chr(96)  # backtick, kept out of the source to stay readable
    lines = [
        "# 游戏资源目录（程序需要的格子）",
        "",
        f"由 {btn}python tools/extract_game_assets.py{btn} 从游戏安装目录自动生成；游戏换皮或改版后重跑即可。",
        f"本机游戏目录：{btn}{game_root}{btn}",
        "",
        "## 为什么要这份目录",
        "",
        f"- 游戏是 Electron + Cocos Creator 3.x 打包，素材全部是 UUID 命名的 PNG（{btn}resources/app/web-mobile/assets/**/native/*.png{btn}），没有可读名字，只能靠颜色和形状反查含义。",
        f"- 棋盘由一族 **{BOARD_MIN_SIZE}–{BOARD_MAX_SIZE} 像素**（多为 84×84）的贴图拼成：未开瓦片是 (169,169,169) 灰底加斜面，数字是“纯色 + alpha 字形”，标雷图标是灰底上加任意图形。",
        "- 识别只依赖两件事：**数字的确切颜色**、**瓦片是否还是未开的灰底**。所以这里记录数字色表，并列出旗子/地雷/未开瓦片的贴图以便核对。",
        "",
        "## 数字色表（对应 minesweeper/recognition.py 的 DIGIT_COLORS）",
        "",
        "| 数字 | RGB | 十六进制 |",
        "| --- | --- | --- |",
    ]
    for number in sorted(DIGIT_COLORS):
        color = ", ".join(str(int(v)) for v in DIGIT_COLORS[number])
        lines.append(f"| {number} | ({color}) | {DIGIT_HEX.get(number, '')} |")
    lines += [
        "",
        "数字贴图还有一套“描边版”（填充色相同，多了白/黑描边），颜色一致所以不用单独处理。",
        "",
        "## 标雷图标不止一种",
        "",
        "游戏允许换标记皮肤：目录里能找到红三角旗、黄色笑脸、蓝星、红心，实际对局里还有白猫脸。",
        "所以识别**不能**绑定某一张贴图，只能判定“这格还是未开的灰瓦片，但上面画了东西”——由",
        f"{btn}classify_cell{btn} 的中心区 art 占比（图形覆盖）与外沿灰底比（瓦片没开过）两条证据实现。",
        "",
        "## 实测特征（仓库 samples/*-raw.png 真实截图，9 帧共 2560 格）",
        "",
        "| 格子类型 | 中心区灰底比 | 中心区非背景 art | 外沿灰底比 |",
        "| --- | --- | --- | --- |",
        "| 未开瓦片 | 0.952–1.000 | 0.000–0.048 | 0.709–1.000 |",
        "| 已标雷 | 0.260–0.740 | 0.260–0.740 | 0.319–0.643 |",
        "| 已开格（数字/空/雷/宝箱） | 0.000–0.007 | 0.223–0.725 | 0.000–0.050 |",
        "",
        "结论：外沿灰底比 0.25 是“这一格没开过”的可靠判据（图标铺满整格也成立）；中心 art 占比 0.10",
        "是“灰瓦片上有图形”的判据（未开瓦片最多 0.048）。",
        "",
        "## 棋盘美术族清单",
        "",
        f"共 {len(rows)} 张棋盘贴图。标签按颜色自动判定；主色列出占比 ≥2% 的平涂色。",
        "",
        "| # | 尺寸 | 标签 | sha256[:12] | 灰底比 | 主色 | UUID |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for index, row in enumerate(rows, start=1):
        palette = " ".join(f"({c[0]},{c[1]},{c[2]}) {share:.0%}" for c, share in row["palette"])
        lines.append(
            f"| {index} | {row['size']} | {row['label']} | {btn}{row['sha']}{btn} | {row['gray']:.2f} | {palette} | {btn}{row['uuid']}{btn} |"
        )
    lines += [
        "",
        "## 重新生成",
        "",
        btn * 3 + "bash",
        "python tools/extract_game_assets.py                              # 默认游戏目录，只写本文件",
        "python tools/extract_game_assets.py --dump tools/_game_assets    # 顺便把贴图存出来看",
        btn * 3,
        "",
        "贴图只是参考，不需要随仓库分发：识别逻辑不依赖具体美术，只依赖颜色与灰底证据。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
