const feed = document.querySelector("#feed");

function card(clip) {
  const article = document.createElement("article");
  const video = document.createElement("video");
  video.controls = true;
  video.playsInline = true;
  video.preload = "metadata";
  video.src = `../outputs/${clip.playback}`;
  if (clip.width && clip.height) {
    video.style.aspectRatio = `${clip.width} / ${clip.height}`;
  }
  const meta = document.createElement("p");
  meta.className = "meta";
  const prompt = document.createElement("strong");
  prompt.textContent = clip.prompt;
  meta.append(
    prompt,
    `${clip.width}×${clip.height} · seed ${clip.seed} · ${clip.num_frames} frames · ${clip.steps} steps`,
  );
  article.append(video, meta);
  return article;
}

function show(message) {
  const p = document.createElement("p");
  p.className = "empty";
  p.textContent = message;
  feed.append(p);
}

fetch("../outputs/manifest.json")
  .then((response) => {
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  })
  .then((manifest) => {
    const clips = manifest.clips || [];
    if (!clips.length) {
      show("No clips yet. Generate one, then reload.");
      return;
    }
    clips.forEach((clip) => feed.append(card(clip)));
  })
  .catch(() => {
    show("No manifest yet. Serve this folder from the repo root, then generate a clip.");
  });
