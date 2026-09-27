# mlx-video-gen

The simplest local video generation example: a prompt → Z-Image Turbo draws the first frame → Wan 2.2 5B (Turbo LoRA) animates it. One script, `generate.py`, on Apple Silicon via MLX.

## Run
- `uv sync`, then download the models (commands in README.md).
- `uv run python generate.py "<prompt>" [--motion "<what moves>"] [--seconds 1.7] [--seed N] [--image-only]`
- Output: `outputs/<time>.png` and `.mp4`. A 1.7 s clip takes ~8 min on an M5 32 GB.

## Constraints
- Open models only, running locally (Apple Silicon, 32 GB).
- Image first, then video. Text-to-video alone is not used.
- Keep it to one script. New features go elsewhere unless they make this example better.
- Non-goals: storyboards, voiceover, captions or assembled ads (on the `storyboards` branch); production infrastructure.

## Key decisions
- 2026-09-24 — Rejected — Text-to-video alone (Wan 2.1 1.3B; Wan 2.2 5B without an image). Both dropped unusual subjects. 14B Wan and LTX-2 don't fit 32 GB; SVD takes no text prompt.
- 2026-09-24 — Z-Image Turbo still, then Wan 2.2 5B animates it. The still holds the subject and is the fast prompt test.
- 2026-09-24 — Wan 2.2 5B Turbo LoRA: 4 steps, guide 1, euler, shift 5. Same still and seed: 20 steps took 20 min and flickered; Turbo took 4.3 min and held the first frame. The LoRA has no stated license.
- 2026-09-24 — Rejected — Storyboards with voiceover, captions and drawn graphics. Built for four demos; kept on the `storyboards` branch.
- 2026-09-26 — main is one script, generate.py, with settings fixed. The four demos sit in `examples/`.

## Session
Read STATUS.md first if it exists (local only, not in the repo). Update it before ending a turn that changed files.
