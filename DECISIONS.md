# DECISIONS.md

## Tech Stack

**Video Generation Model:** Stable Video Diffusion (SVD) via Hugging Face `diffusers` library
- Free, open-source, runs on consumer GPUs
- Generates 4-8 second videos from images/text
- Reasonable quality for ads/promo content

**Backend:** Python with FastAPI (minimal API for generation + management)

**Frontend:** Simple web UI (React/vanilla JS) to display feed of generated videos

**Infrastructure:** Local/cloud GPU (start local for MVP, scale as needed)

## Initial Stack Rationale
- SVD chosen over AnimateDiff because of simplicity and better out-of-box results
- FastAPI for lightweight API if we want to trigger generation via UI
- Keep frontend minimal to iterate fast on video content/prompts
