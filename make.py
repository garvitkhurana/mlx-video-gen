"""Storyboard in, finished short video out. Slow parts (AI footage, voice) are cached by hash."""

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageColor, ImageDraw, ImageFont

from clips import OUTPUTS, ROOT, load_json, new_id, rebuild_manifest, write_json

W, H, FPS = 704, 1280, 24
SR = 24000
SHOTS = OUTPUTS / "shots"
VOICE = OUTPUTS / "voice"
VOICE_CLONE = Path.home() / "Projects" / "voice-clone"
BOLD = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
VO_LEAD, VO_TAIL = 0.1, 0.35


def digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def fit(draw: ImageDraw.ImageDraw, text: str, size: int, margin: int = 48) -> ImageFont.FreeTypeFont:
    while size > 20 and draw.textlength(text, font=font(BOLD, size)) > W - 2 * margin:
        size -= 2
    return font(BOLD, size)


def fill(text: str | None, values: dict) -> str | None:
    return text.format(**values) if text and "{" in text else text


# ---------- data ----------

def load_prices(shot: dict, refresh: bool) -> list[tuple[date, float]]:
    cache = ROOT / shot["cache"]
    if refresh or not cache.is_file():
        request = urllib.request.Request(shot["url"], headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(response.read())
    result = json.loads(cache.read_text())["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    return [
        (datetime.fromtimestamp(ts, timezone.utc).date(), close)
        for ts, close in zip(result["timestamp"], closes)
        if close is not None
    ]


def build_values(board: dict, refresh: bool) -> tuple[dict, list]:
    values, sources = {}, []
    if board.get("facts"):
        facts = load_json(ROOT / board["facts"])
        values.update(facts.get("values", {}))
        sources += facts.get("sources", [])
    if board.get("tokenizer"):
        import tiktoken

        enc = tiktoken.get_encoding(board["tokenizer"])
        values["vocab"] = f"{round(enc.n_vocab, -3):,}"
        values["vocab_exact"] = f"{enc.n_vocab:,}"
    for shot in board["shots"]:
        if shot["type"] == "chart":
            prices = load_prices(shot, refresh)
            since = date.fromisoformat(shot["change_since"])
            base = [p for d, p in prices if d <= since][-1]
            change = (prices[-1][1] / base - 1) * 100
            values[shot.get("var", "change")] = f"{change:,.0f}"
            values["last_date"] = prices[-1][0].strftime("%b %-d, %Y")
            shot["_prices"] = prices
            sources.append({"what": "daily closes", "url": shot["url"].split("?")[0]})
    return values, sources


# ---------- slow, cached parts ----------

ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
STT_MODEL = "mlx-community/whisper-large-v3-turbo-asr-fp16"
VOICE_TEMPERATURES = (0.45, 0.35, 0.6)
MIN_MATCH = 0.8


def number_words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + number_words(n % 100) if n % 100 else "")
    for size, name in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= size:
            return number_words(n // size) + " " + name + (" " + number_words(n % size) if n % size else "")
    raise ValueError(n)


def spoken(text: str, pronounce: dict) -> str:
    """Speech text: pronunciations applied, digits spelled out (TTS misreads digits)."""
    for written, said in pronounce.items():
        text = text.replace(written, said)
    text = re.sub(r"(\d[\d,]*)\s*%", r"\1 percent", text)
    text = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: ordinal_words(int(m.group(1))), text)
    return re.sub(r"\d[\d,]*", lambda m: number_words(int(m.group().replace(",", ""))), text)


def ordinal_words(n: int) -> str:
    head, _, last = number_words(n).rpartition(" ")
    irregular = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
    last = irregular.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return f"{head} {last}".strip()


def words(text: str, pronounce: dict) -> list[str]:
    return re.findall(r"[a-z]+", spoken(text, pronounce).lower())


def transcribe(wav: Path) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stt"
        subprocess.run(
            ["uv", "run", "python", "-m", "mlx_audio.stt.generate", "--model", STT_MODEL, "--audio", str(wav), "--output-path", str(out)],
            cwd=ROOT, capture_output=True, check=True,
        )
        return out.with_suffix(".txt").read_text().strip()


def voice_line(text: str, voice: dict, pronounce: dict) -> Path:
    say = spoken(text, pronounce)
    key = digest({"say": say, "voice": voice})
    out, check = VOICE / f"{key}.wav", VOICE / f"{key}.check.json"
    if out.is_file() and check.is_file():
        return out
    VOICE.mkdir(parents=True, exist_ok=True)
    work = VOICE / "work"
    work.mkdir(exist_ok=True)
    script = work / f"{key}.txt"
    script.write_text(say)
    heard, score = "", 0.0
    for temperature in VOICE_TEMPERATURES:
        cmd = [
            "uv", "run", "--project", str(VOICE_CLONE), "python", str(VOICE_CLONE / "scripts" / "qwen3_clone.py"),
            "--ref-audio", voice["ref_audio"], "--ref-text-file", voice["ref_text_file"],
            "--text-file", str(script), "--quality", voice.get("quality", "lite"),
            "--speed", str(voice.get("speed", 1.08)), "--temperature", str(temperature),
            "--output-format", "wav", "--output-dir", str(work), "--no-stats", "--quiet",
        ]
        result = subprocess.run(cmd, cwd=VOICE_CLONE, capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(f"voice-clone failed:\n{result.stderr[-2000:]}")
        produced = Path(result.stdout.strip().splitlines()[-1])
        if not produced.is_absolute():
            produced = VOICE_CLONE / produced
        shutil.move(str(produced), out)
        heard = transcribe(out)
        score = SequenceMatcher(None, words(say, pronounce), words(heard, pronounce)).ratio()
        print(f"  voice {score:.2f} t={temperature}: {say[:50]!r} -> {heard[:50]!r}")
        if score >= MIN_MATCH:
            write_json(check, {"say": say, "heard": heard, "score": round(score, 3), "temperature": temperature})
            return out
    raise SystemExit(f"voice line failed the transcript check (best {score:.2f}).\nwant:  {say}\nheard: {heard}")


def level(wav: np.ndarray, target_db: float = -20.0, peak: float = 0.95) -> np.ndarray:
    """Same loudness for every line (lines are generated separately), without clipping."""
    speech = wav[np.abs(wav) > 0.01]
    if speech.size == 0:
        return wav
    gain = 10 ** (target_db / 20) / np.sqrt(np.mean(speech**2))
    gain = min(gain, peak / np.abs(wav).max())
    return (wav * gain).astype(np.float32)


def trim_silence(wav: np.ndarray, threshold: float = 0.01, margin: float = 0.05) -> np.ndarray:
    loud = np.flatnonzero(np.abs(wav) > threshold)
    if loud.size == 0:
        return wav
    pad = int(margin * SR)
    return wav[max(0, loud[0] - pad): loud[-1] + pad]


def frames_for(seconds: float) -> int:
    n = math.ceil(seconds * FPS)
    return n + (1 - n) % 4


def ai_clip(shot: dict) -> Path:
    spec = {k: shot.get(k) for k in ("still", "motion", "dur", "seed")}
    if shot.get("image"):
        spec["image"] = shot["image"]
    key = digest({"ai": spec, "turbo": 1})
    video = SHOTS / f"{key}.mp4"
    if video.is_file():
        return video
    from generate import DEFAULT_MODEL, TURBO_LORA, TURBO_PRESET, generate_one, release_memory
    from still import generate_still

    SHOTS.mkdir(parents=True, exist_ok=True)
    seed = shot.get("seed", int(key[:8], 16))
    prompt = shot.get("still") or shot["motion"]
    print(f"  ai shot {key}: {prompt[:60]}")
    if shot.get("image"):
        image = ROOT / shot["image"]
    else:
        image = generate_still(prompt=shot["still"], width=W, height=H, seed=seed, output=SHOTS / f"{key}.png")
        release_memory()
    generate_one(
        prompt=prompt,
        model_dir=DEFAULT_MODEL,
        width=W,
        height=H,
        num_frames=frames_for(shot["dur"]),
        steps=TURBO_PRESET["steps"],
        seed=seed,
        guide_scale=TURBO_PRESET["guide_scale"],
        output=video,
        image=image,
        motion_prompt=shot["motion"],
        loras=[(TURBO_LORA, 1.0)],
        scheduler=TURBO_PRESET["scheduler"],
        shift=TURBO_PRESET["shift"],
    )
    release_memory()
    return video


def last_frame(video: Path) -> Path:
    """Start image for a continuation shot, so the same person keeps moving."""
    out = video.with_name(f"{video.stem}_last.png")
    if not out.is_file():
        subprocess.run(["ffmpeg", "-v", "error", "-sseof", "-0.2", "-i", str(video), "-update", "1", "-frames:v", "1", str(out)], check=True)
    return out


def read_frames(video: Path) -> list[np.ndarray]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True,
        check=True,
    ).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3))


# ---------- drawn shots ----------

def card_frame(shot: dict, t: float, values: dict) -> Image.Image:
    img = Image.new("RGB", (W, H), shot.get("bg", "#0E1116"))
    draw = ImageDraw.Draw(img)
    y = shot.get("top", 430)
    for i, line in enumerate(shot["lines"]):
        at = line.get("at", i * 0.25)
        if t < at:
            continue
        alpha = min(1.0, (t - at) / 0.25)
        size = line.get("size", 64)
        color = tuple(int(c * alpha) for c in ImageColor.getrgb(line.get("color", "#FFFFFF")))
        text = fill(line["text"], values)
        draw.text((W / 2, y + (1 - alpha) * 20), text, font=fit(draw, text, size), fill=color, anchor="mm")
        y += int(size * 1.5)
    if shot.get("source"):
        draw.text((W / 2, H - 70), fill(shot["source"], values), font=font(BOLD, 22), fill="#8A93A0", anchor="mm")
    return img


def chart_image(shot: dict, values: dict) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prices = shot["_prices"]
    fig = plt.figure(figsize=(W / 100, 620 / 100), dpi=100)
    fig.patch.set_facecolor("#0E1116")
    ax = fig.add_axes([0.15, 0.1, 0.8, 0.84])
    ax.set_facecolor("#0E1116")
    xs = [d for d, _ in prices]
    ys = [p for _, p in prices]
    ax.plot(xs, ys, color="#4ADE80", linewidth=3.5)
    ax.fill_between(xs, ys, min(ys), color="#4ADE80", alpha=0.12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#39414D")
    ax.tick_params(colors="#8A93A0", labelsize=12)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    fig.canvas.draw()
    chart = Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB")
    plt.close(fig)
    return chart


def chart_frame(shot: dict, t: float, values: dict, cache: dict) -> Image.Image:
    if "chart" not in cache:
        cache["chart"] = chart_image(shot, values)
    chart = cache["chart"]
    img = Image.new("RGB", (W, H), "#0E1116")
    draw = ImageDraw.Draw(img)
    title, headline = fill(shot["title"], values), fill(shot["headline"], values)
    draw.text((W / 2, 250), title, font=fit(draw, title, 60), fill="white", anchor="mm")
    draw.text((W / 2, 340), headline, font=fit(draw, headline, 88), fill="#4ADE80", anchor="mm")
    reveal = min(1.0, t / shot.get("reveal", 1.2))
    visible = chart.crop((0, 0, max(1, int(chart.width * reveal)), chart.height))
    img.paste(visible, (0, 440))
    if shot.get("source"):
        draw.text((W / 2, H - 70), fill(shot["source"], values), font=font(BOLD, 22), fill="#8A93A0", anchor="mm")
    return img


# ---------- overlays ----------

def phrases(text: str, max_words: int = 4, max_chars: int = 20) -> list[str]:
    words_, out, cur = text.split(), [], []
    for word in words_:
        if cur and len(" ".join(cur + [word])) > max_chars:
            out.append(" ".join(cur))
            cur = []
        cur.append(word)
        if len(cur) >= max_words or word[-1] in ".,!?:":
            out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def draw_caption(img: Image.Image, text: str, y: float) -> None:
    draw = ImageDraw.Draw(img)
    size = 46
    while size > 24 and draw.textlength(text, font=font(BOLD, size)) > W - 64:
        size -= 2
    draw.text((W / 2, y), text, font=font(BOLD, size), fill="white", anchor="mm", stroke_width=5, stroke_fill="black")


def caption_at(vo: str, vo_dur: float, t: float) -> str | None:
    parts = phrases(vo)
    total = sum(len(p) for p in parts)
    clock = VO_LEAD
    for part in parts:
        span = vo_dur * len(part) / total
        if clock <= t < clock + span:
            return part
        clock += span
    return parts[-1] if t >= clock else None


def draw_end_card(img: Image.Image, card: dict) -> None:
    band = Image.new("RGBA", (W, 300), (0, 0, 0, 150))
    img.paste(band, (0, int(H * 0.62)), band)
    draw = ImageDraw.Draw(img)
    draw.text((W / 2, H * 0.62 + 110), card["caption"], font=font(BOLD, 58), fill="white", anchor="mm")
    if card.get("cta"):
        draw.text((W / 2, H * 0.62 + 190), card["cta"], font=font(BOLD, 34), fill="#FFD84D", anchor="mm")


def draw_label(img: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(img)
    draw.text((W / 2, 150), label, font=font(BOLD, 54), fill="white", anchor="mm", stroke_width=5, stroke_fill="black")


# ---------- assembly ----------

def build(board_path: Path, ai_only: bool, refresh: bool) -> Path | None:
    board = load_json(board_path)
    shots = board["shots"]

    previous = None
    for shot in shots:
        if shot["type"] == "ai":
            if shot.get("continue"):
                if previous is None:
                    raise SystemExit("a shot with continue needs an earlier ai shot")
                shot["image"] = last_frame(previous).relative_to(ROOT).as_posix()
            shot["_video"] = previous = ai_clip(shot)
    if ai_only:
        return None
    values, sources = build_values(board, refresh)

    voice = board.get("voice")
    if voice:
        voice = {**voice, "ref_audio": str(VOICE_CLONE / voice["ref_audio"]), "ref_text_file": str(VOICE_CLONE / voice["ref_text_file"])}
    audio_parts, transcript, total = [], [], 0.0
    for shot in shots:
        vo = fill(shot.get("vo"), values)
        shot["_vo"] = vo
        if vo and voice:
            wav, sr = sf.read(voice_line(vo, voice, board.get("pronounce", {})), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != SR:
                raise SystemExit(f"voice sample rate {sr}, expected {SR}")
            wav = level(trim_silence(wav))
            shot["_vo_dur"] = len(wav) / SR
            shot["_dur"] = max(shot.get("min_dur", 0), VO_LEAD + shot["_vo_dur"] + VO_TAIL)
        else:
            wav = np.zeros(0, np.float32)
            shot["_dur"] = shot.get("dur", shot.get("min_dur", 2.5))
        n = round(shot["_dur"] * FPS)
        shot["_frames"] = n
        track = np.zeros(int(n / FPS * SR), np.float32)
        lead = int(VO_LEAD * SR)
        track[lead:lead + len(wav)] = wav[: len(track) - lead]
        audio_parts.append(track)
        transcript.append(vo or fill(shot.get("caption"), values) or "")
        total += n / FPS

    target = board.get("target_s")
    print(f"length {total:.1f} s" + (f" (target {target} s)" if target else ""))
    if target and total > target * 1.2:
        print(f"WARNING: {total - target:.1f} s over target")

    OUTPUTS.mkdir(exist_ok=True)
    video_id = new_id(OUTPUTS)
    out = OUTPUTS / f"{video_id}.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        sf.write(wav_path, np.concatenate(audio_parts), SR)
        enc = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
             "-i", "-", "-i", str(wav_path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             "-c:a", "aac", "-ar", "48000", "-b:a", "192k", "-shortest", str(out)],
            stdin=subprocess.PIPE,
        )
        for index, shot in enumerate(shots):
            last = index == len(shots) - 1
            cache: dict = {}
            footage = read_frames(shot["_video"]) if shot["type"] == "ai" else None
            for f in range(shot["_frames"]):
                t = f / FPS
                if footage is not None:
                    img = Image.fromarray(footage[min(f, len(footage) - 1)])
                elif shot["type"] == "notes":
                    from notes import render as render_notes

                    img = render_notes(shot["elements"], t, values)
                elif shot["type"] == "card":
                    img = card_frame(shot, t, values)
                elif shot["type"] == "chart":
                    img = chart_frame(shot, t, values, cache)
                else:
                    raise SystemExit(f"unknown shot type {shot['type']}")
                if shot.get("label"):
                    draw_label(img, fill(shot["label"], values))
                if board.get("captions", True):
                    if shot.get("_vo") and shot.get("_vo_dur"):
                        text = caption_at(shot["_vo"], shot["_vo_dur"], t)
                        if text:
                            draw_caption(img, text, shot.get("caption_y", H * 0.86))
                    elif shot.get("caption"):
                        draw_caption(img, fill(shot["caption"], values), shot.get("caption_y", H * 0.80))
                end = board.get("end_card")
                if last and end and t >= shot["_frames"] / FPS - end.get("dur", 1.0):
                    draw_end_card(img, end)
                enc.stdin.write(img.convert("RGB").tobytes())
        enc.stdin.close()
        if enc.wait() != 0:
            raise SystemExit("ffmpeg encode failed")

    out.with_suffix(".txt").write_text("\n".join(line for line in transcript if line) + "\n")
    write_json(
        out.with_suffix(".json"),
        {
            "id": video_id,
            "kind": "video",
            "prompt": board["title"],
            "title": board["title"],
            "storyboard": board_path.resolve().relative_to(ROOT).as_posix(),
            "duration": round(total, 2),
            "shots": [{"type": s["type"], "seconds": round(s["_frames"] / FPS, 2), **({"footage": s["_video"].name} if s["type"] == "ai" else {})} for s in shots],
            "values": values,
            "sources": sources,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "video": out.name,
            "transcript": out.with_suffix(".txt").name,
        },
    )
    rebuild_manifest(OUTPUTS)
    print(f"Saved {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a storyboard into a finished short video")
    parser.add_argument("storyboards", nargs="+", type=Path)
    parser.add_argument("--ai-only", action="store_true", help="Only render (and cache) the AI shots")
    parser.add_argument("--refresh-data", action="store_true", help="Re-download chart data")
    args = parser.parse_args()
    for board in args.storyboards:
        print(f"== {board}")
        build(board, args.ai_only, args.refresh_data)


if __name__ == "__main__":
    main()
