"""Prompt -> image (Z-Image Turbo) -> video (Wan 2.2 5B + Turbo LoRA). Runs locally on Apple Silicon."""

import argparse
import gc
import random
from datetime import datetime
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent
IMAGE_MODEL = ROOT / "models" / "Z-Image-Turbo-mflux-4bit"
VIDEO_MODEL = ROOT / "models" / "Wan2.2-TI2V-5B-MLX"
TURBO_LORA = ROOT / "models" / "loras" / "LoRAs" / "Wan22-Turbo" / "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"
WIDTH, HEIGHT, FPS = 704, 1280, 24


def make_image(prompt: str, seed: int, out: Path) -> None:
    from mflux.models.common.config import ModelConfig
    from mflux.models.z_image import ZImage

    model = ZImage(model_config=ModelConfig.z_image_turbo(), model_path=str(IMAGE_MODEL))
    model.generate_image(prompt=prompt, seed=seed, num_inference_steps=9, width=WIDTH, height=HEIGHT).save(str(out))
    del model
    gc.collect()
    mx.clear_cache()  # free the image model before the video model loads


def make_video(image: Path, motion: str, seconds: float, seed: int, out: Path) -> None:
    from mlx_video.models.wan_2.generate import generate_video

    frames = round(seconds * FPS) // 4 * 4 + 1  # Wan needs 4n+1 frames
    generate_video(
        model_dir=str(VIDEO_MODEL),
        image=str(image),
        prompt=motion,
        width=WIDTH,
        height=HEIGHT,
        num_frames=frames,
        seed=seed,
        loras=[(str(TURBO_LORA), 1.0)],
        steps=4,
        guide_scale=1.0,
        scheduler="euler",
        shift=5.0,
        output_path=str(out),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", help="What the image shows: subject and its size first, then the setting")
    parser.add_argument("--motion", default="The camera slowly pushes in.", help="What moves in the video")
    parser.add_argument("--seconds", type=float, default=1.7)
    parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1))
    parser.add_argument("--image-only", action="store_true", help="Stop after the image (fast prompt test)")
    args = parser.parse_args()

    out = ROOT / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.parent.mkdir(exist_ok=True)
    image, video = out.with_suffix(".png"), out.with_suffix(".mp4")

    make_image(args.prompt, args.seed, image)
    print(f"image: {image}  (seed {args.seed})")
    if not args.image_only:
        make_video(image, args.motion, args.seconds, args.seed, video)
        print(f"video: {video}")


if __name__ == "__main__":
    main()
