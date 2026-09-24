"""One prompt in, one short mp4 out, with the settings saved beside it."""

import argparse
import gc
import random
from pathlib import Path

from clips import OUTPUTS, ROOT, clip_record, load_json, new_id, read_prompts, rebuild_manifest, split_prompt, write_json

DEFAULT_MODEL = ROOT / "models" / "Wan2.2-TI2V-5B-MLX"
DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 1280
DEFAULT_FRAMES = 41
DEFAULT_STEPS = 20
TURBO_LORA = ROOT / "models" / "loras" / "LoRAs" / "Wan22-Turbo" / "Wan22_TI2V_5B_Turbo_lora_rank_64_fp16.safetensors"
TURBO_PRESET = {"steps": 4, "guide_scale": 1.0, "scheduler": "euler", "shift": 5.0}


def release_memory() -> None:
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


def image_ref(image: Path) -> str:
    return path_ref(image)


def path_ref(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


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
    image: Path | None = None,
    motion_prompt: str | None = None,
    loras: list | None = None,
    scheduler: str | None = None,
    shift: float | None = None,
    tiling: str | None = None,
    trim_first_frames: int | None = None,
) -> Path:
    check_frames(num_frames)
    if not model_dir.is_dir():
        raise SystemExit(
            f"model not found at {model_dir}. Download and convert it first (see README)."
        )
    if image is not None and not image.is_file():
        raise SystemExit(f"start image not found: {image}")
    loras = [(Path(path) if Path(path).is_absolute() else ROOT / path, strength) for path, strength in loras or []]
    for lora_path, _ in loras:
        if not Path(lora_path).is_file():
            raise SystemExit(f"LoRA not found: {lora_path}. Download it first (see README).")
    chosen = choose_seed(seed)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    clip_id = output.stem if output else new_id(OUTPUTS)
    video = output or (OUTPUTS / f"{clip_id}.mp4")
    video.parent.mkdir(parents=True, exist_ok=True)

    try:
        from mlx_video.models.wan_2.generate import generate_video
    except ImportError as exc:
        raise SystemExit("mlx-video is not installed. From the repo root: uv sync") from exc

    extra = {
        "loras": [(str(path), float(strength)) for path, strength in loras] if loras else None,
        "scheduler": scheduler,
        "shift": shift,
        "tiling": tiling,
        "trim_first_frames": trim_first_frames,
    }
    generate_video(
        model_dir=str(model_dir),
        prompt=motion_prompt or prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        steps=steps,
        seed=chosen,
        guide_scale=guide_scale,
        image=str(image) if image else None,
        output_path=str(video),
        **{key: value for key, value in extra.items() if value is not None},
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
        image=image_ref(image) if image else None,
        motion_prompt=motion_prompt,
        loras=[[path_ref(path), float(strength)] for path, strength in loras] if loras else None,
        scheduler=scheduler,
        shift=shift,
        tiling=tiling,
        trim_first_frames=trim_first_frames,
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
        "image": args.image if args.image is not None else (Path(base["image"]) if base.get("image") else None),
        "motion_prompt": args.motion if args.motion is not None else base.get("motion_prompt"),
        "loras": args.lora if args.lora is not None else base.get("loras"),
        "scheduler": args.scheduler if args.scheduler is not None else base.get("scheduler"),
        "shift": args.shift if args.shift is not None else base.get("shift"),
        "tiling": args.tiling if args.tiling is not None else base.get("tiling"),
        "trim_first_frames": (
            args.trim_first_frames if args.trim_first_frames is not None else base.get("trim_first_frames")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a short Wan 2.2 clip from a still")
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
    parser.add_argument("--image", type=Path, default=None, help="Start frame. Skips making a still.")
    parser.add_argument("--still", action="store_true", help="Make a Z-Image Turbo still, then animate it")
    parser.add_argument("--motion", default=None, help="Video-stage prompt: only what moves. The still uses the main prompt.")
    parser.add_argument("--turbo", action="store_true", help="Turbo LoRA preset: 4 steps, CFG off. Explicit flags override.")
    parser.add_argument("--lora", nargs=2, action="append", metavar=("PATH", "STRENGTH"), default=None)
    parser.add_argument("--scheduler", choices=["euler", "dpm++", "unipc"], default=None)
    parser.add_argument("--shift", type=float, default=None, help="Noise schedule shift")
    parser.add_argument("--tiling", default=None, help="VAE decode tiling: auto, none, default, aggressive, conservative, spatial, temporal")
    parser.add_argument("--trim-first-frames", type=int, default=None, help="Extra latent frames to generate and drop at the start (x4 frames)")
    args = parser.parse_args()

    if args.turbo:
        for key, value in TURBO_PRESET.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        if args.lora is None:
            args.lora = [[str(TURBO_LORA), "1.0"]]

    if args.prompts and (args.prompt or args.from_json):
        parser.error("use either a prompt, --from, or --prompts")
    if args.output and args.prompts:
        parser.error("--output cannot be used with --prompts")
    if args.still and args.image:
        parser.error("--still makes the start image; do not also pass --image")

    base = {}
    if args.from_json:
        base = load_json(args.from_json)
        if base.get("kind") not in (None, "clip"):
            parser.error("--from expects a clip sidecar, not a branded file")

    if args.prompts:
        jobs = []
        for line in read_prompts(args.prompts):
            still_prompt, motion = split_prompt(line)
            overrides = {"prompt": still_prompt, "motion": motion or args.motion}
            jobs.append(settings_from_args(argparse.Namespace(**{**vars(args), **overrides}), base))
    else:
        job = settings_from_args(args, base)
        if not job["prompt"]:
            parser.error("pass a prompt, --prompts, or --from")
        jobs = [job]

    text_only = "T2V-1.3B" in args.model_dir.name
    for job in jobs:
        image = job["image"]
        if args.still:
            from still import generate_still

            image = generate_still(prompt=job["prompt"], width=job["width"], height=job["height"])
            release_memory()
        elif image is None and not text_only:
            raise SystemExit("This model needs a start image. Pass --still or --image.")
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
            image=image,
            motion_prompt=job["motion_prompt"],
            loras=job["loras"],
            scheduler=job["scheduler"],
            shift=job["shift"],
            tiling=job["tiling"],
            trim_first_frames=job["trim_first_frames"],
        )


if __name__ == "__main__":
    main()
