"""One prompt in, one still out, with the seed saved beside it."""

import argparse
import random
from datetime import datetime
from pathlib import Path

from clips import OUTPUTS, ROOT, load_json, new_id, read_prompts, write_json

DEFAULT_MODEL = ROOT / "models" / "Z-Image-Turbo-mflux-4bit"
HF_MODEL = "filipstrand/Z-Image-Turbo-mflux-4bit"
DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 1280
DEFAULT_STEPS = 9
MODEL_NAME = "Z-Image-Turbo"


def choose_seed(seed: int | None) -> int:
    if seed is None or seed < 0:
        return random.randint(0, 2**32 - 1)
    return seed


def still_record(
    still_id: str,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    steps: int,
    image_name: str,
) -> dict:
    return {
        "id": still_id,
        "kind": "still",
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "steps": steps,
        "model": MODEL_NAME,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "image": image_name,
    }


def generate_still(
    prompt: str,
    model_dir: Path | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    seed: int | None = None,
    output: Path | None = None,
) -> Path:
    model_dir = model_dir or DEFAULT_MODEL
    if not model_dir.is_dir():
        raise SystemExit(
            f"still model not found at {model_dir}. Download it first:\n"
            f"uv run hf download {HF_MODEL} --local-dir {DEFAULT_MODEL}"
        )
    chosen = choose_seed(seed)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    still_id = output.stem if output else new_id(OUTPUTS)
    image_path = output or (OUTPUTS / f"{still_id}.png")
    image_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from mflux.models.common.config import ModelConfig
        from mflux.models.z_image import ZImage
    except ImportError as exc:
        raise SystemExit("mflux is not installed. From the repo root: uv sync") from exc

    model = ZImage(
        model_config=ModelConfig.z_image_turbo(),
        model_path=str(model_dir),
    )
    image = model.generate_image(
        seed=chosen,
        prompt=prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
    )
    image.save(str(image_path))
    del image
    del model

    try:
        image_name = image_path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        image_name = str(image_path)
    record = still_record(
        still_id=still_id,
        prompt=prompt,
        seed=chosen,
        width=width,
        height=height,
        steps=steps,
        image_name=image_name,
    )
    write_json(image_path.with_suffix(".json"), record)
    print(f"Saved {image_path}")
    print(f"Saved {image_path.with_suffix('.json')}")
    return image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Z-Image Turbo still")
    parser.add_argument("prompt", nargs="?", help="Text prompt")
    parser.add_argument("--prompts", type=Path, help="Text file with one prompt per line")
    parser.add_argument("--from", dest="from_json", type=Path, help="Rerun a saved still JSON")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Omit for a new random seed, saved in the JSON")
    args = parser.parse_args()

    if args.prompts and (args.prompt or args.from_json):
        parser.error("use either a prompt, --from, or --prompts")
    if args.output and args.prompts:
        parser.error("--output cannot be used with --prompts")

    base = {}
    if args.from_json:
        base = load_json(args.from_json)
        if base.get("kind") not in (None, "still"):
            parser.error("--from expects a still sidecar")

    if args.prompts:
        prompts = read_prompts(args.prompts)
    else:
        prompt = args.prompt if args.prompt is not None else base.get("prompt")
        if not prompt:
            parser.error("pass a prompt, --prompts, or --from")
        prompts = [prompt]

    width = args.width if args.width is not None else base.get("width", DEFAULT_WIDTH)
    height = args.height if args.height is not None else base.get("height", DEFAULT_HEIGHT)
    steps = args.steps if args.steps is not None else base.get("steps", DEFAULT_STEPS)
    seed = args.seed if args.seed is not None else base.get("seed")

    for prompt in prompts:
        generate_still(
            prompt=prompt,
            model_dir=args.model_dir,
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            output=args.output,
        )


if __name__ == "__main__":
    main()
