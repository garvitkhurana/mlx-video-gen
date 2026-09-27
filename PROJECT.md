# mlx-video-gen: local prompt → image → video

## Goal
The simplest local example of video generation: type a prompt, an image model draws the first frame, a video model animates it. One script, easy to read.

## Constraints
- Open models only, running locally on this Mac (M5, 32 GB).
- Image first, then video. Text-to-video alone is not used.
- Keep it to one script. New features go elsewhere unless they make this example better.

## Non-goals
- Storyboards, voiceover, captions or assembled ads (they live on the `storyboards` branch).
- Production infrastructure or a multi-user platform.
