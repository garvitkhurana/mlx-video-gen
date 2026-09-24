# DECISIONS.md

## 2026-09-24 — Rejected — Wan 2.1 1.3B

This machine is a MacBook Pro (M5, 10-core GPU, 32 GB). The goal is to iterate on text prompts for short ads.

**Model:** Wan2.1-T2V-1.3B (Apache 2.0), 4-bit MLX weights via [mlx-video](https://github.com/Blaizzy/mlx-video). The text encoder stays full precision (~11 GB). Only the transformer is quantized.

**Why this size:** 14B Wan and LTX-2 do not fit a 32 GB laptop comfortably. The 1.3B text-to-video model does.

**Loop:** Default frame is 480×832 (9:16). Each clip writes an mp4 and a JSON sidecar with the prompt, seed, and settings. A prompts file batches clips. A static page reads `outputs/manifest.json`. No server starts a render.

**Type and branding:** Captions, logo, and call to action are an ffmpeg pass after the footage. The prompt stays free of words, prices, and logos.

**Quality:** 1.3B at 480p is for testing prompts. It will not look like a top-tier model. Finals can later try Wan 2.2 5B on this Mac, or 14B on a rented GPU. Neither starts until this loop works.

## 2026-09-24 — Wan 2.2 5B, short clips

The loop works, so 5B was tried. 4-bit `Wan2.2-TI2V-5B` rendered 704×1280, 17 frames, 10 steps in about 3.5 minutes on this 32 GB Mac. Text-only, no start image.

1.3B stays the fast prompt test. 5B is the better local model for a short clip. 14B still does not fit this machine.

## Prompt following

Wan 2.2 5B with no start image follows the setting and drops an unusual subject. The paper-boat prompt became an empty wet street for the whole clip.

Wan 2.1 1.3B is the text-to-video model. Name the subject and its size first, then the setting. `--guide-scale 7` holds that subject. A rewritten prompt produced a small white paper boat on wet pavement (`outputs/20260924-091507.mp4`).

## Rejected — SVD (first draft)

**Video Generation Model:** Stable Video Diffusion (SVD) via Hugging Face `diffusers` library
- Free, open-source, runs on consumer GPUs
- Generates 4-8 second videos from images/text
- Reasonable quality for ads/promo content

**Backend:** Python with FastAPI (minimal API for generation + management)

**Frontend:** Simple web UI (React/vanilla JS) to display feed of generated videos

**Infrastructure:** Local/cloud GPU (start local for MVP, scale as needed)

**Why it was picked, then dropped:** SVD was chosen over AnimateDiff for a simpler first run. It does not take a text prompt, and this Mac is not an NVIDIA GPU. A generation API was dropped with it: a clip takes minutes, so the first loop is a prompts file and a static feed.

## 2026-09-24 — Z-Image Turbo still, then Wan 2.2 5B

The local clip is a Z-Image Turbo still, then Wan 2.2 5B from that frame (704×1280, 41 frames, 20 steps). The still is the prompt test. 1.3B is no longer the draft. Its weights stay until a still-then-5B paper-boat clip looks right. 14B stays off this machine.
