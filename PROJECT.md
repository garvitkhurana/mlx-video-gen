# video-gen: Short-form Video Factory

## Goal
Make short vertical videos (ads, promos, concept explainers) locally with free, open models, shown in a feed. Each video is a storyboard of shots: AI footage for photoreal motion, code-drawn graphics for text, data and diagrams, and a cloned voiceover, assembled into one clip.

## Constraints
- Open models only, running locally on this Mac (M5, 32 GB).
- Every fact or number on screen comes from a cited source or is computed.
- No real brands' logos.
- The voice is the user's own.
- Short-form lengths: lifestyle ads 6–8 s, stock snapshots ~9 s, explainers ~30 s (45 s max).
- Iterate fast: cache everything that is slow.

## Non-goals
- Production infrastructure, monetization or deployment.
- Multi-user platform.
- A general-purpose video editor. Assembly is fixed: cuts, captions, voiceover, end card.
