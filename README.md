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

Say what you are promoting. The still shows that thing, large, with no words. The clip is a slow push-in. The name and the price go on afterwards with `brand.py`.

```bash
uv run python generate.py --promote "a summer sneaker"
uv run python generate.py --promote "a stock chart on a desk"
```

A full prompt still works. `--still` makes the PNG and animates it. `--image` uses a PNG you already have. `--turbo` is the fast clip, about 4 minutes, and `--promote` turns it on.

```bash
uv run python still.py "A tiny white origami paper boat, small enough to sit in one hand, floating in a puddle on wet asphalt at night"
uv run python generate.py "..." --still --turbo --motion "The camera slowly pushes in."
uv run python generate.py --prompts prompts.txt --still --turbo
uv run python generate.py --from outputs/<id>.json
```

`--num-frames` must be 4n+1 (17, 41, 81). A bare `generate.py` call uses Wan 2.2 5B and asks for `--promote`, `--still`, or `--image`.

The Turbo LoRA has to be on disk once:

```bash
uv run hf download Kijai/WanVideo_comfy LoRAs/Wan22-Turbo/Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors --local-dir models/loras
```

`--motion` replaces the push-in. Describe only what moves. Do not restate the subject the still already shows.

`prompts.txt` is one prompt per line. `#` starts a comment. `still prompt || motion prompt` gives the clip its own motion prompt. `--promote` treats each line as the thing being advertised.

A still writes `outputs/<id>.png` and `outputs/<id>.json`. A clip writes `outputs/<id>.mp4` and `outputs/<id>.json`, then refreshes `outputs/manifest.json`.

## Storyboards (finished videos)

A finished short is `storyboards/<name>.json`: a list of shots, each with an optional voiceover line (`vo`) or caption.

```bash
uv run python make.py storyboards/tokenization.json
uv run python make.py storyboards/mu.json storyboards/beach.json --ai-only   # pre-render slow AI shots
uv run python make.py storyboards/mu.json --refresh-data                      # new price data
```

Shot types:
- `ai`: `still` + `motion` prompts, `dur` seconds. Z-Image still, then Wan 2.2 5B `--turbo`. Cached in `outputs/shots/`.
- `notes`: hand-drawn explainer (`notes.py`): `text`, `highlight`, `box`, `arrow`, `tokens` (real `tiktoken` splits and IDs), `bracket`. Each element has `at` (seconds).
- `card`: lines of text on a dark background, with a `source` line.
- `chart`: daily closes from `url`, cached in `cache`; `change_since` sets `{<var>}` (percent change).

`{name}` in any text is filled from `facts` (a JSON of values with sources), the chart, or the tokenizer (`{vocab}`). Only use numbers that come from those.

An `ai` shot with `"continue": true` starts from the previous AI shot's last frame, so the same person keeps moving. Use it for every shot of one character; separate stills give a different person each time.

Voiceover comes from `~/Projects/voice-clone` (your cloned voice), cached in `outputs/voice/`. Digits are spelled out for speech (captions keep the digits); add odd names to the storyboard's `pronounce` map (e.g. `"GPT-4o": "GPT four oh"`). Every line is transcribed with Whisper and must match the script, or it is retried and then the build stops. Write full sentences: very short lines are unstable with this voice model. The voice line sets how long the shot runs. Short-form targets: lifestyle ad 6–8 s, stock snapshot ~9 s, explainer ~30 s. `make.py` warns when a video runs more than 20% over `target_s`.

Output: `outputs/<id>.mp4`, `outputs/<id>.txt` (transcript), `outputs/<id>.json` (sources and values), and a feed entry.

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
