"""Burn a caption, call to action, and optional logo onto a clean clip."""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from clips import ROOT, branded_record, load_json, rebuild_manifest, write_json

DEFAULT_TEMPLATE = ROOT / "templates" / "end-card.json"
FONT_CANDIDATES = [
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial.ttf"),
    Path("/System/Library/Fonts/Helvetica.ttc"),
]


def find_font() -> Path:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return path
    raise SystemExit("No usable font found for ffmpeg drawtext")


def resolve_logo(raw: str | None, template_path: Path) -> Path | None:
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return path
    beside = template_path.parent / raw
    if beside.is_file():
        return beside
    raise SystemExit(f"logo not found: {raw}")


def probe_size(video: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(video),
        ],
        text=True,
    )
    width, height = out.strip().split(",")
    return int(width), int(height)


def render_card(
    width: int,
    height: int,
    font_path: Path,
    caption: str,
    cta: str,
    logo_path: Path | None,
) -> Path:
    if not caption and not cta and logo_path is None:
        raise SystemExit("template needs a caption, a call to action, or a logo")
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if caption:
        font = ImageFont.truetype(str(font_path), 46)
        draw.text(
            (width / 2, height * 0.74),
            caption,
            font=font,
            fill="white",
            anchor="mm",
            stroke_width=3,
            stroke_fill="black",
        )
    if cta:
        font = ImageFont.truetype(str(font_path), 32)
        draw.text(
            (width / 2, height * 0.84),
            cta,
            font=font,
            fill="white",
            anchor="mm",
            stroke_width=3,
            stroke_fill="black",
        )
    if logo_path:
        logo = Image.open(logo_path).convert("RGBA")
        logo.thumbnail((140, 140))
        image.alpha_composite(logo, (width - logo.width - 36, 48))
    handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    handle.close()
    card = Path(handle.name)
    image.save(card)
    return card


def brand(
    source_json: Path,
    template_path: Path,
    caption: str | None,
    cta: str | None,
    logo: str | None,
) -> Path:
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH")
    clip = load_json(source_json)
    if clip.get("kind") not in (None, "clip"):
        raise SystemExit("pass a clip sidecar, not a branded file")
    video = source_json.with_suffix(".mp4")
    if not video.is_file():
        named = source_json.parent / clip.get("video", "")
        if named.is_file():
            video = named
        else:
            raise SystemExit(f"clip video not found next to {source_json}")

    template = load_json(template_path)
    caption_text = caption if caption is not None else template.get("caption", "")
    cta_text = cta if cta is not None else template.get("cta", "")
    logo_path = resolve_logo(logo if logo is not None else template.get("logo"), template_path)
    font = find_font()
    width, height = probe_size(video)
    card = render_card(width, height, font, caption_text, cta_text, logo_path)
    source_id = clip["id"]
    output = source_json.parent / f"{source_id}-branded.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(video),
                "-i",
                str(card),
                "-filter_complex",
                "[0:v][1:v]overlay=0:0",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(output),
            ],
            check=True,
        )
    finally:
        card.unlink(missing_ok=True)
    record = branded_record(source_id, template.get("name") or template_path.stem)
    write_json(source_json.parent / f"{source_id}-branded.json", record)
    if source_json.parent.resolve() == (ROOT / "outputs").resolve():
        rebuild_manifest()
    print(f"Saved {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Add caption, call to action, and optional logo")
    parser.add_argument("clip", type=Path, help="Clip sidecar JSON")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--cta", default=None)
    parser.add_argument("--logo", default=None, help="PNG logo, instead of the template logo")
    args = parser.parse_args()
    brand(args.clip, args.template, args.caption, args.cta, args.logo)


if __name__ == "__main__":
    main()
