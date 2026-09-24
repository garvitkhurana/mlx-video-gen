# video-gen

Local text-to-video clips with Wan 2.1 1.3B on Apple Silicon. One prompt, one mp4, and a JSON file of the settings used.

## Setup

Needs macOS on Apple Silicon, Python 3.11+, [uv](https://docs.astral.sh/uv/), ffmpeg, and about 30 GB free disk.

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv run hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/Wan2.1-T2V-1.3B
uv run python -m mlx_video.models.wan_2.convert \
  --checkpoint-dir models/Wan2.1-T2V-1.3B \
  --output-dir models/Wan2.1-T2V-1.3B-MLX \
  --quantize --bits 4
```

Torch is only for that conversion. After it finishes, `models/Wan2.1-T2V-1.3B` can be deleted. The MLX directory stays; the text encoder in it is about 11 GB.

## Generate

Default clip is 480×832, 17 frames, 10 steps. The seed is chosen up front and saved next to the mp4.

```bash
uv run python generate.py "A paper boat crosses a rain-lit street at night, camera low, no text"
uv run python generate.py --prompts prompts.txt
uv run python generate.py --from outputs/20260924-012200.json
uv run python generate.py "..." --width 832 --height 480 --num-frames 81 --steps 50
```

Wan 2.2 5B is the sharper local model. It wants more steps than 1.3B. 704×1280 is its portrait 720p.

```bash
uv run python generate.py "A paper boat crosses a rain-lit street at night, camera low, no text" \
  --model-dir models/Wan2.2-TI2V-5B-MLX \
  --width 704 --height 1280 --num-frames 17 --steps 40
```

`--num-frames` must be 4n+1 (17, 41, 81). 480×832 and 832×480 are this model's native 480p.

Name the subject and its size before the setting. `--guide-scale 7` holds an unusual object more tightly than the default. Wan 2.2 5B, with no start image, tends to paint the place and drop that object.

`prompts.txt` is one prompt per line. `#` starts a comment.

Each clip writes `outputs/<id>.mp4` and `outputs/<id>.json`, then refreshes `outputs/manifest.json`.

## Brand

Do not ask the model to draw words, logos, or prices. Burn them on after:

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

Open http://127.0.0.1:8765/feed/ . The page plays the branded file when one exists, otherwise the clean clip. It does not start a render.

## Caveats

- First run downloads about 15 GB, then conversion writes the MLX weights.
- A longer clip on this 32 GB Mac takes minutes, not seconds.
- 1.3B at 480p is for prompt tests, not a finished ad.
- Weights are Apache 2.0. `mlx-video` is MIT.
