"""The OpenFlow mark: "Pulse" -- a waveform, iconized.

Six rounded bars on a blue-to-violet sweep, on a dark rounded tile. Drawn in
code so the tray icon can be recolored per state and the Windows .ico is
generated from the same source as everything else.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

ICO_PATH = Path(__file__).resolve().parent.parent.parent / "assets" / "openflow.ico"

# Geometry on a 130-unit tile: (x, y, width, height) per bar.
_BARS = (
    (26, 53, 9, 24),
    (40, 42, 9, 46),
    (54, 30, 9, 70),
    (68, 38, 9, 54),
    (82, 50, 9, 30),
    (96, 58, 9, 14),
)
_TILE = 130
_GRAD_A = (0x6C, 0x8C, 0xFF)   # accent blue
_GRAD_B = (0xA7, 0x8B, 0xFA)   # violet


def _lerp(a: tuple, b: tuple, t: float) -> tuple:
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def mic_image(size: int = 64, color: str | None = None, tile: bool = True):
    """Render the mark at ``size`` px. ``color`` forces all bars to one color
    (used by the tray to signal state); None gives the gradient sweep."""
    from PIL import Image, ImageDraw

    scale = size / _TILE
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    if tile:
        # The tile is part of the mark and stays dark regardless of app theme.
        draw.rounded_rectangle(
            (0, 0, size - 1, size - 1),
            radius=int(30 * scale),
            fill="#14171C",
        )

    last = len(_BARS) - 1
    for i, (x, y, w, h) in enumerate(_BARS):
        fill = color or ("#%02x%02x%02x" % _lerp(_GRAD_A, _GRAD_B, i / last))
        draw.rounded_rectangle(
            (x * scale, y * scale, (x + w) * scale, (y + h) * scale),
            radius=max(1, int(4.5 * scale)),
            fill=fill,
        )
    return image


def write_ico(path: Path | None = None) -> Path | None:
    """Generate the multi-resolution .ico used by the Windows shortcuts."""
    path = path or ICO_PATH
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        log.info("Pillow not installed; skipping icon generation")
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    base = mic_image(256)
    base.save(path, format="ICO",
              sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


def ensure_ico() -> Path | None:
    if ICO_PATH.exists():
        return ICO_PATH
    return write_ico()


if __name__ == "__main__":
    print(write_ico())
