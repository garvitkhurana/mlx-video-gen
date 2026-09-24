# video-gen

A local still, then a short clip. Z-Image Turbo draws the frame. Wan 2.2 5B animates it. Each file keeps a JSON sidecar with the prompt and seed.

## Setup

Needs macOS on Apple Silicon, Python 3.11+, [uv](https://docs.astral.sh/uv/), ffmpeg, and about 40 GB free disk.

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv run hf download filipstrand/Z-Image-Turbo-mflux-4bit --local-dir models/Z-Image-Turbo-mflux-4bit
uv run hf download Wan-AI/Wan2.2-TI2V-5B --local-dir models/Wan2.2-TI2V-5B
uv run python -m mlx_video.models.wan_2.convert \
  --checkpoint-dir models/Wan2.2-TI2V-5B \
  --output-dir models/Wan2.2-TI2V-5B-MLX \
  --quantize --bits 4
```

Torch is only for the Wan conversion. After it finishes, `models/Wan2.2-TI2V-5B` can be deleted. The MLX directory stays.

## Generate

Name the subject and its size before the setting. Do not ask the model to draw words, logos, or prices.

A still is 704×1280, nine steps. The seed is chosen up front and saved next to the PNG.

```bash
uv run python still.py "A tiny white origami paper boat, small enough to sit in one hand, floating in a puddle on wet asphalt at night"
uv run python still.py --from outputs/<id>.json
```

The clip is that still, then Wan 2.2 5B at 704×1280, 41 frames, 20 steps. `--still` makes the PNG and animates it. `--image` uses a PNG you already have.

```bash
uv run python generate.py "A tiny white origami paper boat, small enough to sit in one hand, floating in a puddle on wet asphalt at night" --still
uv run python generate.py --prompts prompts.txt --still
uv run python generate.py "..." --image outputs/<id>.png
uv run python generate.py --from outputs/<id>.json
```

`--num-frames` must be 4n+1 (17, 41, 81). A bare `generate.py` call uses Wan 2.2 5B and asks for `--still` or `--image`.

`prompts.txt` is one prompt per line. `#` starts a comment.

A still writes `outputs/<id>.png` and `outputs/<id>.json`. A clip writes `outputs/<id>.mp4` and `outputs/<id>.json`, then refreshes `outputs/manifest.json`.

## Brand

Burn words on after the footage:

```bash
uv run python brand.py outputs/<id>.json
uv run python brand.py outputs/<id>.json --caption "Shop the drop" --cta "Link in bio" --logo logo.png
```

The template is [templates/end-card.json](templates/end-card.json). Pillow draws the words and logo onto a transparent card, and ffmpeg overlays that card. This writes `outputs/<id>-branded.mp4`.

## Feed

From the repo root:

```bash
python -m http.server 8765
```

Open http://127.0.0.1:8765/feed/ . The page plays the branded file when one exists, otherwise the clean clip, and shows the start image when the sidecar has one. It does not start a render.

## Caveats

- The first still download is about 6 GB. The Wan 2.2 download is about 32 GB, then conversion writes the MLX weights.
- A 41-frame clip on this 32 GB Mac takes about 20 minutes.
- 1.3B weights, if they are still on disk, are not the draft. Pass `--model-dir models/Wan2.1-T2V-1.3B-MLX` to use them.
- Wan weights are Apache 2.0. `mlx-video` is MIT. Z-Image Turbo weights are Apache 2.0. `mflux` is MIT.
