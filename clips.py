"""Clip files, sidecars, and the feed manifest."""

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODEL_NAME = "Wan2.1-T2V-1.3B"


def new_id(directory: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = stamp
    n = 2
    while any((directory / f"{candidate}{suffix}").exists() for suffix in (".mp4", ".png", ".json")):
        candidate = f"{stamp}-{n}"
        n += 1
    return candidate


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def clip_record(
    clip_id: str,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    num_frames: int,
    steps: int,
    guide_scale: float | None = None,
    model: str | None = None,
    created_at: str | None = None,
    image: str | None = None,
    motion_prompt: str | None = None,
    loras: list | None = None,
    scheduler: str | None = None,
    shift: float | None = None,
    tiling: str | None = None,
    trim_first_frames: int | None = None,
) -> dict:
    record = {
        "id": clip_id,
        "kind": "clip",
        "prompt": prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "steps": steps,
        "model": model or MODEL_NAME,
        "created_at": created_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "video": f"{clip_id}.mp4",
    }
    if guide_scale is not None:
        record["guide_scale"] = guide_scale
    optional = {
        "image": image,
        "motion_prompt": motion_prompt,
        "loras": loras,
        "scheduler": scheduler,
        "shift": shift,
        "tiling": tiling,
        "trim_first_frames": trim_first_frames,
    }
    record.update({key: value for key, value in optional.items() if value})
    return record


def split_prompt(line: str) -> tuple[str, str | None]:
    """`still prompt || motion prompt` -> (still, motion). Motion is None without `||`."""
    still, sep, motion = line.partition("||")
    return still.strip(), (motion.strip() or None) if sep else None


def branded_record(source_id: str, template_name: str, created_at: str | None = None) -> dict:
    return {
        "id": f"{source_id}-branded",
        "kind": "branded",
        "source_id": source_id,
        "template": template_name,
        "created_at": created_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "video": f"{source_id}-branded.mp4",
    }


def load_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Could not read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path} is not a JSON object")
    return data


def rebuild_manifest(directory: Path | None = None) -> Path:
    directory = directory or OUTPUTS
    directory.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    branded: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "manifest.json":
            continue
        data = load_json(path)
        if data.get("kind") == "branded" and data.get("source_id"):
            branded[data["source_id"]] = data
        elif data.get("kind") in ("clip", "video"):
            clips.append(data)
    clips.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    items = []
    for clip in clips:
        item = dict(clip)
        brand = branded.get(clip["id"])
        item["playback"] = brand["video"] if brand else clip["video"]
        if brand:
            item["branded"] = brand["video"]
            item["template"] = brand.get("template")
        items.append(item)
    manifest_path = directory / "manifest.json"
    write_json(manifest_path, {"clips": items})
    return manifest_path


def read_prompts(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"prompts file not found: {path}")
    prompts = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    if not prompts:
        raise SystemExit(f"No prompts in {path}")
    return prompts
