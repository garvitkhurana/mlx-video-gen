"""One prompt in, one short mp4 out, with the settings saved beside it."""

import argparse
import random
from pathlib import Path

from clips import OUTPUTS, ROOT, clip_record, load_json, new_id, read_prompts, rebuild_manifest, write_json

DEFAULT_MODEL = ROOT / "models" / "Wan2.1-T2V-1.3B-MLX"
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 832
DEFAULT_FRAMES = 17
DEFAULT_STEPS = 10


def choose_seed(seed: int | None) -> int:
    if seed is None or seed < 0:
        return random.randint(0, 2**32 - 1)
    return seed


def check_frames(num_frames: int) -> None:
    if (num_frames - 1) % 4 != 0:
        raise SystemExit("num-frames must be 4n+1, for example 17, 41, or 81")


def generate_one(
    prompt: str,
    model_dir: Path,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    seed: int | None,
    guide_scale: float | None,
    output: Path | None,
) -> Path:
    check_frames(num_frames)
    if not model_dir.is_dir():
        raise SystemExit(
            f"model not found at {model_dir}. Download and convert it first (see README)."
        )
    chosen = choose_seed(seed)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    clip_id = output.stem if output else new_id(OUTPUTS)
    video = output or (OUTPUTS / f"{clip_id}.mp4")
    video.parent.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_video.models.wan_2.generate import generate_video
    except ImportError as exc:
        raise SystemExit("mlx-video is not installed. From the repo root: uv sync") from exc

    generate_video(
        model_dir=str(model_dir),
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        steps=steps,
        seed=chosen,
        guide_scale=guide_scale,
        output_path=str(video),
    )
    record = clip_record(
        clip_id=clip_id,
        prompt=prompt,
        seed=chosen,
        width=width,
        height=height,
        num_frames=num_frames,
        steps=steps,
        guide_scale=guide_scale,
        model=model_dir.name.removesuffix("-MLX"),
    )
    write_json(video.with_suffix(".json"), record)
    if video.parent.resolve() == OUTPUTS.resolve():
        rebuild_manifest(OUTPUTS)
    print(f"Saved {video}")
    print(f"Saved {video.with_suffix('.json')}")
    return video


def settings_from_args(args: argparse.Namespace, base: dict | None = None) -> dict:
    base = base or {}
    return {
        "prompt": args.prompt if args.prompt is not None else base.get("prompt"),
        "width": args.width if args.width is not None else base.get("width", DEFAULT_WIDTH),
        "height": args.height if args.height is not None else base.get("height", DEFAULT_HEIGHT),
        "num_frames": (
            args.num_frames if args.num_frames is not None else base.get("num_frames", DEFAULT_FRAMES)
        ),
        "steps": args.steps if args.steps is not None else base.get("steps", DEFAULT_STEPS),
        "seed": args.seed if args.seed is not None else base.get("seed"),
        "guide_scale": args.guide_scale if args.guide_scale is not None else base.get("guide_scale"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a short Wan 2.1 clip")
    parser.add_argument("prompt", nargs="?", help="Text prompt")
    parser.add_argument("--prompts", type=Path, help="Text file with one prompt per line")
    parser.add_argument("--from", dest="from_json", type=Path, help="Rerun a saved clip JSON")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Omit for a new random seed, saved in the JSON")
    parser.add_argument("--guide-scale", type=float, default=None, help="How tightly to follow the prompt. Higher sticks closer.")
    args = parser.parse_args()

    if args.prompts and (args.prompt or args.from_json):
        parser.error("use either a prompt, --from, or --prompts")
    if args.output and args.prompts:
        parser.error("--output cannot be used with --prompts")

    base = {}
    if args.from_json:
        base = load_json(args.from_json)
        if base.get("kind") not in (None, "clip"):
            parser.error("--from expects a clip sidecar, not a branded file")

    if args.prompts:
        jobs = [
            settings_from_args(argparse.Namespace(**{**vars(args), "prompt": line}), base)
            for line in read_prompts(args.prompts)
        ]
    else:
        job = settings_from_args(args, base)
        if not job["prompt"]:
            parser.error("pass a prompt, --prompts, or --from")
        jobs = [job]

    for job in jobs:
        generate_one(
            prompt=job["prompt"],
            model_dir=args.model_dir,
            width=job["width"],
            height=job["height"],
            num_frames=job["num_frames"],
            steps=job["steps"],
            seed=job["seed"],
            guide_scale=job["guide_scale"],
            output=args.output,
        )


if __name__ == "__main__":
    main()
