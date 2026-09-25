"""Hand-drawn explainer frames: text that writes on, wobbly boxes, arrows, highlights, token chips."""

import random
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 704, 1280
PAPER = "#FFFDF8"
INK = "#222222"
FONT = Path(__file__).resolve().parent / "assets" / "fonts" / "Caveat.ttf"
CHIP_COLORS = ["#BDE0FE", "#B9F2C8", "#FFE8A3", "#D9C8FF", "#FFC8DD"]


@lru_cache(maxsize=None)
def hand(size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(str(FONT), size)
    try:
        font.set_variation_by_name("Bold")
    except (OSError, ValueError):
        pass
    return font


@lru_cache(maxsize=None)
def encoder(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


def progress(t: float, at: float, dur: float) -> float:
    if t < at:
        return 0.0
    return 1.0 if dur <= 0 else min(1.0, (t - at) / dur)


def ease(p: float) -> float:
    return 1 - (1 - p) ** 3


def wobble_line(draw, p0, p1, seed: int, width: int = 3, color: str = INK, frac: float = 1.0) -> None:
    rng = random.Random(seed)
    (x0, y0), (x1, y1) = p0, p1
    x1, y1 = x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac
    points = []
    for i in range(9):
        s = i / 8
        jitter = rng.uniform(-1.6, 1.6)
        dx, dy = y1 - y0, x0 - x1
        norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
        points.append((x0 + (x1 - x0) * s + dx / norm * jitter, y0 + (y1 - y0) * s + dy / norm * jitter))
    draw.line(points, fill=color, width=width, joint="curve")


def wobble_rect(draw, box, seed: int, fill: str | None = None, frac: float = 1.0) -> None:
    x0, y0, x1, y1 = box
    if fill and frac > 0:
        draw.rounded_rectangle((x0 + 4, y0 + 4, x1 - 2, y1 - 2), radius=10, fill=fill)
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    edges = list(zip(corners, corners[1:]))
    for pass_seed in (seed, seed + 97):
        for i, (a, b) in enumerate(edges):
            edge_frac = min(1.0, max(0.0, frac * 4 - i))
            if edge_frac > 0:
                wobble_line(draw, a, b, pass_seed + i, width=3, frac=edge_frac)


def write_text(img: Image.Image, xy, text: str, size: int, color: str, anchor: str, p: float) -> None:
    if p <= 0:
        return
    font = hand(size)
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.multiline_text(xy, text, font=font, fill=color, anchor=anchor, align="center", spacing=6)
    if p < 1:
        bbox = draw.multiline_textbbox(xy, text, font=font, anchor=anchor, align="center", spacing=6)
        cut = bbox[0] + (bbox[2] - bbox[0]) * p
        layer.paste((0, 0, 0, 0), (int(cut), 0, img.width, img.height))
    img.alpha_composite(layer)


def draw_arrow(draw, e: dict, p: float, seed: int) -> None:
    if p <= 0:
        return
    x0, y0, x1, y1 = e["x0"], e["y0"], e["x1"], e["y1"]
    wobble_line(draw, (x0, y0), (x1, y1), seed, width=4, frac=p)
    if p >= 1:
        dx, dy = x1 - x0, y1 - y0
        norm = max(1.0, (dx * dx + dy * dy) ** 0.5)
        ux, uy = dx / norm, dy / norm
        for sx in (1, -1):
            hx = x1 - ux * 22 + (-uy) * 14 * sx
            hy = y1 - uy * 22 + ux * 14 * sx
            wobble_line(draw, (x1, y1), (hx, hy), seed + sx + 5, width=4)


def token_layout(e: dict):
    enc = encoder(e.get("encoding", "o200k_base"))
    ids = enc.encode(e["text"])
    pieces = [enc.decode([i]) for i in ids]
    size = e.get("size", 46)
    font = hand(size)
    x_left, width = e.get("x0", 40), e.get("width", W - 80)
    pad, gap, line_h = 14, 12, size + 70 + (50 if e.get("ids_at") is not None else 0)
    x, y, placed = x_left, e["y"], []
    for piece, token_id in zip(pieces, ids):
        label = piece.replace(" ", "·")
        w = font.getlength(label) + pad * 2
        if x + w > x_left + width and x > x_left:
            x, y = x_left, y + line_h
        placed.append((label, token_id, x, y, w))
        x += w + gap
    rows_right = {}
    for _, _, px, py, pw in placed:
        rows_right[py] = px + pw
    offset = {py: (x_left + width - right) / 2 for py, right in rows_right.items()}
    return [(lab, tid, px + offset[py], py, pw) for lab, tid, px, py, pw in placed], size


def draw_tokens(img: Image.Image, e: dict, t: float, seed: int) -> None:
    placed, size = token_layout(e)
    draw = ImageDraw.Draw(img)
    at, stagger = e.get("at", 0), e.get("stagger", 0.2)
    ids_at = e.get("ids_at")
    font_small = hand(max(30, int(size * 0.72)))
    for i, (label, token_id, x, y, w) in enumerate(placed):
        p = progress(t, at + i * stagger, 0.3)
        if p <= 0:
            continue
        box = (x, y, x + w, y + size + 22)
        wobble_rect(draw, box, seed + i * 11, fill=CHIP_COLORS[i % len(CHIP_COLORS)], frac=ease(p))
        if p >= 0.6:
            write_text(img, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2 - 2), label, size, INK, "mm", 1.0)
            draw = ImageDraw.Draw(img)
        if ids_at is not None and t >= ids_at + i * e.get("ids_stagger", 0.12):
            draw.text(((box[0] + box[2]) / 2, box[3] + 30), str(token_id), font=font_small, fill="#5A5A5A", anchor="mm")
    if e.get("count_label") and placed:
        last_at = at + (len(placed) - 1) * stagger + 0.4
        n = len(placed)
        bottom = max(y for _, _, _, y, _ in placed) + size + 22
        write_text(img, (W / 2, bottom + 50), f"= {n} token{'s' if n != 1 else ''}", size, "#D1495B", "mm", progress(t, last_at, 0.4))


def draw_bracket(draw, img, e: dict, p: float, seed: int) -> None:
    if p <= 0:
        return
    x, y0, y1 = e["x"], e["y0"], e["y1"]
    wobble_line(draw, (x - 16, y0), (x, y0), seed, width=3)
    wobble_line(draw, (x, y0), (x, y0 + (y1 - y0) * p), seed + 1, width=3)
    if p >= 1:
        wobble_line(draw, (x, y1), (x - 16, y1), seed + 2, width=3)
        label = Image.new("RGBA", (y1 - y0 + 200, 70), (0, 0, 0, 0))
        ImageDraw.Draw(label).text((label.width / 2, 35), e["label"], font=hand(e.get("size", 40)), fill="#D1495B", anchor="mm")
        rotated = label.rotate(-90, expand=True)
        img.alpha_composite(rotated, (int(x + 8), int((y0 + y1) / 2 - rotated.height / 2)))


def render(elements: list[dict], t: float, values: dict | None = None) -> Image.Image:
    values = values or {}
    img = Image.new("RGBA", (W, H), PAPER)
    for index, e in enumerate(elements):
        seed = index * 131 + 7
        kind = e["type"]
        at, dur = e.get("at", 0.0), e.get("dur", 0.5)
        p = ease(progress(t, at, dur))
        draw = ImageDraw.Draw(img)
        if kind == "text":
            text = e["text"].format(**values) if "{" in e["text"] else e["text"]
            write_text(img, (e.get("x", W / 2), e["y"]), text, e.get("size", 56), e.get("color", INK), e.get("anchor", "mm"), p)
        elif kind == "highlight":
            if p > 0:
                layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
                x1 = e["x0"] + (e["x1"] - e["x0"]) * p
                ImageDraw.Draw(layer).rounded_rectangle((e["x0"], e["y0"], x1, e["y1"]), radius=12, fill=e.get("color", "#9FD3FFB0"))
                img.alpha_composite(layer)
        elif kind == "box":
            box = (e["x"], e["y"], e["x"] + e["w"], e["y"] + e["h"])
            wobble_rect(draw, box, seed, fill=e.get("fill"), frac=p)
            if e.get("label") and p >= 1:
                label = e["label"].format(**values) if "{" in e["label"] else e["label"]
                write_text(img, ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), label, e.get("size", 48), INK, "mm", progress(t, at + dur, 0.35))
        elif kind == "arrow":
            draw_arrow(draw, e, p, seed)
        elif kind == "tokens":
            draw_tokens(img, e, t, seed)
        elif kind == "bracket":
            draw_bracket(draw, img, e, p, seed)
        else:
            raise SystemExit(f"unknown notes element {kind}")
    return img.convert("RGB")
