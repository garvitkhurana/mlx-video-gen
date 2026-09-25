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

## 2026-09-24 — Wan 2.2 5B Turbo LoRA, 4 steps, CFG off

Clips are a Z-Image still, then Wan 2.2 5B with the Turbo LoRA (`Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors` from Kijai/WanVideo_comfy, extracted from quanhaol/Wan2.2-TI2V-5B-Turbo): 4 steps, guide 1, euler, shift 5, VAE tiling auto. `--turbo` in `generate.py` sets this.

**Why:** same still and seed, 41 frames. The 20-step, guide-5 run (`20260924-125505`) took 20 min and changed her face, crossed her legs, and flickered every 4 frames. Turbo (`20260924-142526`) took 4.3 min (denoising 926 s → 49 s), kept her matching frame 1, and walked with a natural stride. At 121 frames (`20260924-154832`, 5 s) it took 18 min and the 4-frame flicker was gone. The LoRA also restores 300 of the 4-bit layers to bf16, which is part of the quality gain.

**Not chosen:** Wan 2.2 Lightning LoRAs (14B only). FastWan (trained on text-to-video, not image-to-video). Shift 8 (same as shift 5). `tiling none` (out of GPU memory at decode on 32 GB). 8-bit re-conversion (not needed; the LoRA already restores most layers to bf16).

**License:** neither the LoRA nor the quanhaol model card states a license. Clear this before a clip is used in a paid ad.

**Known limit:** faces in full-body shots are small for the VAE and drift. Decode is now most of a 5 s clip's time (851 of 1087 s).

## 2026-09-24 — Voiceover from ~/Projects/voice-clone (Qwen3-TTS clone, lite)

Narration uses the user's voice-clone project (Qwen3-TTS 0.6B 8-bit on mlx-audio), called as a subprocess with `uv run --project ~/Projects/voice-clone`. Reference: `securities_clip.m4a` (the user's own voice) plus `documents/ref_voice_transcript.txt`. One wav per shot, cached by text, voice, and speed.

**Why:** already built and tuned (chunking, ref cache, speed). A consistent brand voice. Same MLX stack, separate environment.

**Not chosen:** Kokoro (stock voice, extra download). Building new TTS here.

## 2026-09-24 — Storyboards with typed shots: AI footage vs code-drawn graphics

A video is `storyboards/<name>.json`. `make.py` renders each shot by type and assembles them with ffmpeg:
- `ai`: Z-Image still, then Wan 2.2 5B `--turbo`.
- `notes`, `card`, `chart`: drawn with PIL and matplotlib.

Shots are cached by hash. Voiceover length sets shot length. AI shots stay 2–3 s.

**Why:** the video models cannot draw readable text, charts, or accurate facts. Code-drawn graphics are exact and instant. Caching means editing a caption or a line never re-renders footage.

**Not chosen:** one AI clip per video (no text or data). Text-rendering image models for diagrams (text errors, facts not checkable).
