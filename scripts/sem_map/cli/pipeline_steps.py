# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Pipeline step functions for end-to-end segmentation layer creation.

Each step is a standalone function that yields progress updates and returns
results. No TUI dependency — can be called from CLI, notebooks, or the TUI.

Pipeline:
    1. MCAP or video (mp4/mov) → RGB images
    2. FAISS K-Means selection + train/val split
    3. SAM3 auto-labeling
    4. Train segmentation head
    5. Export to ONNX
    6. Build TensorRT engine
"""

import os
import sys
import time
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator, Optional, Sequence

import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_video_file(path: str) -> bool:
    """True if path has a known video container extension."""
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: str) -> bool:
    """True if path has a known image extension."""
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


# ── Workspace resolution ─────────────────────────────────────────────────────

def _find_sem_map_root() -> Path:
    """Find the sem_map repo root (the directory containing package.xml)."""
    env = os.environ.get("SEM_MAP_WS_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "package.xml").exists():
            return parent
    return Path.cwd()


def _find_workspace_root() -> Path:
    """Working root for data/outputs; defaults to the repo root."""
    return _find_sem_map_root()


def _ensure_scripts_path():
    """Add scripts dir to sys.path if needed.

    Also prevents sem_map.__init__ from running (it sets multiprocessing
    start method which conflicts with Textual threads).
    """
    scripts = str(_find_sem_map_root() / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)

    # Block sem_map.__init__ from executing if not already imported
    if "sem_map" not in sys.modules:
        import types
        pkg_dir = _find_sem_map_root() / "scripts" / "sem_map"
        dummy = types.ModuleType("sem_map")
        dummy.__path__ = [str(pkg_dir)]
        dummy.__file__ = str(pkg_dir / "__init__.py")
        dummy.__package__ = "sem_map"
        sys.modules["sem_map"] = dummy


def _artifacts_dir() -> Path:
    """Get artifacts directory."""
    return _find_sem_map_root() / "artifacts"


def _seg_layers_dir() -> Path:
    """Get seg_layers config directory."""
    return _find_sem_map_root() / "config" / "seg_layers"


# ── Progress update type ─────────────────────────────────────────────────────

@dataclass
class StepProgress:
    current: int = 0
    total: int = 0
    message: str = ""
    done: bool = False
    result: dict = field(default_factory=dict)


# ── Utility functions ────────────────────────────────────────────────────────

def list_mcap_topics(mcap_path: str, image_only: bool = True) -> list[dict]:
    """List available topics in an MCAP file with metadata.

    Args:
        mcap_path: Path to MCAP file.
        image_only: If True, only return topics with Image or CompressedImage message types.

    Returns:
        List of dicts with keys: topic, schema, message_count, duration_s, fps
    """
    from mcap.reader import make_reader

    IMAGE_TYPES = {"sensor_msgs/msg/Image", "sensor_msgs/Image",
                   "sensor_msgs/msg/CompressedImage", "sensor_msgs/CompressedImage"}

    results = []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = getattr(reader, "get_summary", lambda: None)()
        if not summary:
            return results

        channels = getattr(summary, "channels", None) or {}
        schemas = getattr(summary, "schemas", None) or {}
        statistics = getattr(summary, "statistics", None)

        # Get per-channel message counts from statistics
        channel_msg_counts = {}
        if statistics and hasattr(statistics, "channel_message_counts"):
            channel_msg_counts = statistics.channel_message_counts or {}

        # Get overall time range
        start_ns = getattr(statistics, "message_start_time", 0) if statistics else 0
        end_ns = getattr(statistics, "message_end_time", 0) if statistics else 0
        total_duration_s = (end_ns - start_ns) / 1e9 if end_ns > start_ns else 0

        if isinstance(channels, dict):
            for ch_id, ch in channels.items():
                schema = schemas.get(ch.schema_id)
                schema_name = schema.name if schema else "unknown"

                if image_only and schema_name not in IMAGE_TYPES:
                    continue

                count = channel_msg_counts.get(ch_id, 0)
                fps = count / total_duration_s if total_duration_s > 0 else 0

                results.append({
                    "topic": ch.topic,
                    "schema": schema_name,
                    "message_count": count,
                    "duration_s": round(total_duration_s, 1),
                    "fps": round(fps, 1),
                })

    return sorted(results, key=lambda r: r["topic"])


def list_seg_layer_yamls() -> list[str]:
    """List available seg_layer YAML config files."""
    seg_dir = _seg_layers_dir()
    if not seg_dir.exists():
        return []
    return sorted(f.name for f in seg_dir.glob("*.yaml"))


def load_seg_layer_config(yaml_name: str) -> dict:
    """Load a seg_layer YAML config."""
    import yaml
    yaml_path = _seg_layers_dir() / yaml_name
    if not yaml_path.exists():
        yaml_path = Path(yaml_name)  # absolute path fallback
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_seg_layer_classes(yaml_name: str) -> list[dict]:
    """Load categories from a seg_layer YAML."""
    cfg = load_seg_layer_config(yaml_name)
    return sorted(cfg.get("categories", []), key=lambda c: c["id"])


def _collect_dataset_images(
    dataset_root: Path,
    splits: Optional[Sequence[str]] = None,
) -> list[Path]:
    """Resolve image paths from split files, falling back to images/ directly.

    Args:
        dataset_root: Dataset root with ``images/`` and ``<split>_images.txt``.
        splits: Explicit split base-names (e.g. ``["train_kmeans_k400",
            "val_kmeans_k400"]``). If None, picks up every ``*_images.txt`` in
            the dataset root.

    Behaviour:
        - If split files are found (or specified) → union of their stems.
        - If none are found → every image in ``images/`` directly.
    """
    images_dir = dataset_root / "images"
    all_stems: set[str] = set()
    if splits:
        # Strip any trailing slashes / .txt suffixes typed by users.
        cleaned = [s.rstrip("/").removesuffix("_images.txt").removesuffix(".txt")
                   for s in splits]
        split_files = [dataset_root / f"{s}_images.txt" for s in cleaned]
    else:
        split_files = sorted(dataset_root.glob("*_images.txt"))
    for sf in split_files:
        if not sf.exists():
            continue
        for line in sf.read_text().splitlines():
            stem = line.strip()
            if stem:
                all_stems.add(stem)

    img_paths: list[Path] = []
    if all_stems:
        for stem in sorted(all_stems):
            for ext in (".png", ".jpg", ".jpeg"):
                p = images_dir / f"{stem}{ext}"
                if p.exists():
                    img_paths.append(p)
                    break
    elif images_dir.is_dir():
        for ext in (".png", ".jpg", ".jpeg"):
            img_paths.extend(sorted(images_dir.glob(f"*{ext}")))
    return img_paths


# ═════════════════════════════════════════════════════════════════════════════
# STEP 1: MCAP → RGB Images
# ═════════════════════════════════════════════════════════════════════════════

def step_mcap_extract(
    mcap_path: str,
    topic: str,
    output_dir: str,
    skip: int = 0,
    rotate_180: bool = False,
) -> Generator[StepProgress, None, None]:
    """Extract RGB frames from MCAP file.

    Yields StepProgress updates. Final yield has done=True with result dict.
    """
    sys.path.insert(0, str(_find_sem_map_root() / "tools" / "mcap"))
    from generate_rgb_from_mcap import (
        iterate_image_messages,
        compressed_image_to_rgb_array,
        ros_image_to_rgb_array,
        save_image,
    )
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    yield StepProgress(0, 0, "Opening MCAP file...")

    saved = 0
    processed = 0
    for timestamp_ns, ros_msg in iterate_image_messages(mcap_path, topic):
        is_compressed = hasattr(ros_msg, "format") and not hasattr(ros_msg, "step")

        if processed % (skip + 1) == 0:
            rgb = (compressed_image_to_rgb_array(ros_msg) if is_compressed
                   else ros_image_to_rgb_array(ros_msg))
            if rotate_180:
                rgb = np.flipud(np.fliplr(rgb))
            save_image(saved, rgb, output_dir)
            saved += 1

        processed += 1
        if processed % 10 == 0:
            yield StepProgress(processed, 0, f"Extracted {saved} frames...")

    yield StepProgress(
        processed, processed, f"Done: {saved} frames",
        done=True,
        result={"output_dir": output_dir, "num_frames": saved},
    )


def step_video_extract(
    video_path: str,
    output_dir: str,
    skip: int = 0,
    rotate_180: bool = False,
) -> Generator[StepProgress, None, None]:
    """Extract RGB frames from a video file (mp4/mov/avi/mkv/webm).

    Uses OpenCV to decode; frames are saved as PNGs in output_dir, numbered
    sequentially with 6-digit zero-padding to match MCAP extraction output.

    Yields StepProgress updates. Final yield has done=True with result dict.
    """
    import cv2

    # Reuse save_image helper from the mcap utils for naming consistency
    sys.path.insert(0, str(_find_sem_map_root() / "tools" / "mcap"))
    from generate_rgb_from_mcap import save_image

    os.makedirs(output_dir, exist_ok=True)
    yield StepProgress(0, 0, f"Opening video: {Path(video_path).name}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    # Apply container rotation metadata (e.g. .mov from phones/action cams),
    # so decoded frames match how players display them.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    yield StepProgress(0, total_frames,
                       f"Video opened: {total_frames} frames @ {fps:.1f} fps")

    saved = 0
    processed = 0
    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if processed % (skip + 1) == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if rotate_180:
                    rgb = np.flipud(np.fliplr(rgb))
                save_image(saved, rgb, output_dir)
                saved += 1

            processed += 1
            if processed % 10 == 0:
                yield StepProgress(processed, total_frames,
                                   f"Extracted {saved} frames ({processed}/{total_frames})...")
    finally:
        cap.release()

    yield StepProgress(
        processed, total_frames or processed, f"Done: {saved} frames",
        done=True,
        result={"output_dir": output_dir, "num_frames": saved},
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2: FAISS K-Means Selection
# ═════════════════════════════════════════════════════════════════════════════

def step_faiss_select(
    images_dir: str,
    k: int = 100,
    dino_variant: str = "dinov3s",
    split_ratio: float = 0.8,
    seed: int = 42,
) -> Generator[StepProgress, None, None]:
    """Run DINOv3 feature extraction + K-Means selection + train/val split.

    Yields StepProgress updates. Final yield has done=True with result dict.
    """
    _ensure_scripts_path()
    import torch
    from PIL import Image as PILImage
    from sklearn.cluster import MiniBatchKMeans
    import faiss

    from sem_map.perception.models.dinov import load_dinovx_torch, DINOV_CONFIGS
    from sem_map.perception.common import prepare_image, get_device

    images_path = Path(images_dir)
    dataset_root = images_path.parent
    img_exts = {".png", ".jpg", ".jpeg"}
    img_paths = sorted(p for p in images_path.iterdir() if p.suffix.lower() in img_exts)

    if not img_paths:
        raise FileNotFoundError(f"No images in {images_dir}")

    yield StepProgress(0, len(img_paths), f"Found {len(img_paths)} images")

    # Feature cache
    cache_path = dataset_root / "faiss_cls_features.npy"
    paths_cache = dataset_root / "faiss_cls_paths.npy"

    features = None
    if cache_path.exists() and paths_cache.exists():
        cached_paths = np.load(paths_cache, allow_pickle=True).tolist()
        if cached_paths == [str(p) for p in img_paths]:
            features = np.load(cache_path)
            yield StepProgress(len(img_paths), len(img_paths), "Loaded cached features")

    if features is None:
        yield StepProgress(0, len(img_paths), "Loading DINOv3 backbone...")
        device = get_device()
        model, patch_size = load_dinovx_torch(dino_variant)
        model = model.to(device).eval()
        cfg = DINOV_CONFIGS[dino_variant]
        skip_tokens = cfg["skip_tokens"]

        from sem_map.perception.backends.pytorch import run_vision_transformer

        feats_list = []
        for i, img_path in enumerate(img_paths):
            image = PILImage.open(img_path).convert("RGB")
            _, img_batch, _, _ = prepare_image(image, 32, 20, patch_size, device)
            with torch.no_grad():
                all_tokens, _ = run_vision_transformer(model, img_batch, skip_tokens)
            cls = all_tokens[:, 0, :].cpu().numpy().flatten()
            feats_list.append(cls)

            if (i + 1) % 5 == 0 or i == len(img_paths) - 1:
                yield StepProgress(i + 1, len(img_paths),
                                   f"Extracting features: {i+1}/{len(img_paths)}")

        features = np.stack(feats_list, axis=0).astype(np.float32)
        np.save(cache_path, features)
        np.save(paths_cache, np.array([str(p) for p in img_paths], dtype=object))
        del model
        torch.cuda.empty_cache()

    # K-Means
    yield StepProgress(0, 1, f"Running K-Means (K={k})...")
    actual_k = min(k, len(features))
    km = MiniBatchKMeans(n_clusters=actual_k, random_state=seed, n_init=5, batch_size=512)
    labels = km.fit_predict(features)
    centroids = km.cluster_centers_.astype(np.float32)

    idx = faiss.IndexFlatL2(features.shape[1])
    idx.add(features.astype(np.float32))
    dists, nn_idx = idx.search(centroids, 1)
    sel_idx = list(dict.fromkeys(nn_idx[:, 0].tolist()))

    # Train/val split
    selected_stems = [img_paths[i].stem for i in sel_idx]
    random.Random(seed).shuffle(selected_stems)
    split_point = int(len(selected_stems) * split_ratio)
    train_stems = sorted(selected_stems[:split_point])
    val_stems = sorted(selected_stems[split_point:])

    # Tag split files by K so multiple selection runs (e.g. k100 vs k2000) coexist.
    train_split = f"train_kmeans_k{actual_k}"
    val_split = f"val_kmeans_k{actual_k}"
    train_file = dataset_root / f"{train_split}_images.txt"
    val_file = dataset_root / f"{val_split}_images.txt"
    train_file.write_text("\n".join(train_stems) + "\n")
    val_file.write_text("\n".join(val_stems) + "\n")

    yield StepProgress(
        1, 1,
        f"Selected {len(sel_idx)} images → {len(train_stems)} train, {len(val_stems)} val",
        done=True,
        result={
            "dataset_dir": str(dataset_root),
            "train_file": str(train_file),
            "val_file": str(val_file),
            "train_split": train_split,
            "val_split": val_split,
            "num_selected": len(sel_idx),
            "num_train": len(train_stems),
            "num_val": len(val_stems),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 2.5: LLM Class-Presence Pre-Filter
# ═════════════════════════════════════════════════════════════════════════════

def step_llm_class_filter(
    dataset_dir: str,
    yaml_path: str,
    llm_base_url: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    overwrite: bool = False,
    splits: Optional[Sequence[str]] = None,
) -> Generator[StepProgress, None, None]:
    """Ask a vision LLM which classes from the seg_layer YAML are present per image.

    Writes per-image JSON sidecars to ``<dataset_dir>/llm_present_classes/<stem>.json``
    so ``step_sam3_label`` can subset its text prompts and avoid hallucinated
    masks for absent classes. Resumable: existing sidecars are skipped unless
    ``overwrite=True``.

    Provider config is resolved by ``LLMConfig.from_env`` — pass explicit args
    or set ``SEM_MAP_LLM_BASE_URL`` / ``SEM_MAP_LLM_MODEL`` /
    ``SEM_MAP_LLM_API_KEY``. See ``sem_map.perception.llm_client`` docstring
    for provider examples.

    Yields StepProgress updates. Final yield has done=True with result dict.
    """
    _ensure_scripts_path()
    import json

    from sem_map.perception.llm_client import LLMConfig, check_class_presence

    dataset_root = Path(dataset_dir)
    out_dir = dataset_root / "llm_present_classes"
    out_dir.mkdir(parents=True, exist_ok=True)

    categories = load_seg_layer_classes(yaml_path)
    non_bg_ids = [c["id"] for c in categories if c.get("id", 0) != 0]
    if not non_bg_ids:
        raise ValueError(f"No non-background categories in {yaml_path}")

    config = LLMConfig.from_env(
        base_url=llm_base_url, model=llm_model, api_key=llm_api_key)
    config.require()

    img_paths = _collect_dataset_images(dataset_root, splits=splits)
    if not img_paths:
        raise FileNotFoundError(
            f"No images found under {dataset_root}/images for the dataset splits")

    yield StepProgress(
        0, len(img_paths),
        f"Filtering {len(img_paths)} images with {config.model}...")

    num_processed = 0
    num_cached = 0
    for i, img_path in enumerate(img_paths):
        sidecar = out_dir / f"{img_path.stem}.json"
        if sidecar.exists() and not overwrite:
            num_cached += 1
            yield StepProgress(
                i + 1, len(img_paths),
                f"[cached] {img_path.stem} ({num_cached} cached)")
            continue

        decisions = check_class_presence(str(img_path), categories, config)
        present_ids = sorted(d["id"] for d in decisions if d.get("present"))
        sidecar.write_text(json.dumps({
            "present_class_ids": present_ids,
            "class_decisions": decisions,
            "model": config.model,
            "version": 2,
        }, indent=2))
        num_processed += 1

        # Detect fail-open (every reason starts with "fail-open:") and surface
        # the underlying error in the TUI log so the user can debug.
        fail_open_reasons = {
            d.get("reason", "") for d in decisions
            if str(d.get("reason", "")).startswith("fail-open:")
        }
        if fail_open_reasons and len(fail_open_reasons) == 1 and len(decisions) == len(non_bg_ids):
            err_msg = next(iter(fail_open_reasons))
            yield StepProgress(
                i + 1, len(img_paths),
                f"[red]{img_path.stem}: {err_msg}[/red]")
        else:
            yield StepProgress(
                i + 1, len(img_paths),
                f"{img_path.stem}: {len(present_ids)}/{len(non_bg_ids)} classes "
                f"({num_processed} processed)")

    yield StepProgress(
        len(img_paths), len(img_paths),
        f"Done: {num_processed} processed, {num_cached} cached",
        done=True,
        result={
            "dataset_dir": str(dataset_root),
            "present_classes_dir": str(out_dir),
            "num_processed": num_processed,
            "num_cached": num_cached,
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 3: SAM3 Auto-Labeling
# ═════════════════════════════════════════════════════════════════════════════

def step_sam3_label(
    dataset_dir: str,
    yaml_path: str,
    prompts: Optional[dict[int, str]] = None,
    threshold: float = 0.5,
    present_classes_dir: Optional[str] = None,
    splits: Optional[Sequence[str]] = None,
) -> Generator[StepProgress, None, None]:
    """Label all images in dataset using SAM3 text prompts.

    Args:
        dataset_dir: Dataset root with images/ and train_images.txt / val_images.txt
        yaml_path: Path to seg_layer YAML config
        prompts: Optional {class_id: text_prompt} override. If None, uses category names.
        threshold: Confidence threshold for mask binarization.
        present_classes_dir: Optional dir of per-image JSON sidecars produced by
            ``step_llm_class_filter``. When set, SAM3 only runs prompts whose
            class id appears in ``<dir>/<stem>.json``'s ``present_class_ids``.
            Missing sidecars fall through to the full prompt list (fail-open).

    Yields StepProgress updates. Final yield has done=True with result dict.
    """
    _ensure_scripts_path()
    import json
    import torch
    from PIL import Image as PILImage

    from sem_map.perception.models.sam3 import (
        load_sam3_model,
        process_multiple_prompts,
        get_semantic_segmentation_tensor,
    )

    dataset_root = Path(dataset_dir)
    seg_dir = dataset_root / "segmentations"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    categories = load_seg_layer_classes(yaml_path)
    non_bg = [c for c in categories if c["id"] != 0]
    bg_id = next((c["id"] for c in categories if c["id"] == 0), 0)

    # Build prompts
    if prompts is None:
        prompts = {c["id"]: c["concept"] for c in non_bg if c.get("concept", "").strip()}
    prompts_to_class_ids = {text: cid for cid, text in prompts.items() if text.strip()}

    img_paths = _collect_dataset_images(dataset_root, splits=splits)
    if not img_paths:
        raise FileNotFoundError(f"No images found in {dataset_root / 'images'}")

    presence_root = Path(present_classes_dir) if present_classes_dir else None

    yield StepProgress(0, len(img_paths), "Loading SAM3 model...")
    predictor = load_sam3_model()

    all_text_prompts = list(prompts_to_class_ids.keys())
    class_id_priority = sorted(prompts_to_class_ids.values(), reverse=True)

    for i, img_path in enumerate(img_paths):
        image = PILImage.open(img_path).convert("RGB")
        w, h = image.size

        # Per-image prompt subset from LLM presence sidecar (fail-open).
        text_prompts = all_text_prompts
        filter_note = ""
        if presence_root is not None:
            sidecar = presence_root / f"{img_path.stem}.json"
            if sidecar.exists():
                try:
                    present_ids = set(json.loads(sidecar.read_text())
                                      .get("present_class_ids", []))
                    text_prompts = [
                        p for p in all_text_prompts
                        if prompts_to_class_ids[p] in present_ids
                    ]
                    filter_note = f" [{len(text_prompts)}/{len(all_text_prompts)} prompts]"
                except (json.JSONDecodeError, ValueError) as e:
                    yield StepProgress(
                        i, len(img_paths),
                        f"[yellow]Bad sidecar {sidecar.name}: {e} — using all prompts[/yellow]")
            else:
                yield StepProgress(
                    i, len(img_paths),
                    f"[yellow]Missing sidecar {sidecar.name} — using all prompts[/yellow]")

        # If LLM said nothing is present, skip the SAM3 call entirely.
        if not text_prompts:
            segmentation = np.full((h, w), bg_id, dtype=np.uint8)
            PILImage.fromarray(segmentation, mode="L").save(seg_dir / f"{img_path.stem}.png")
            yield StepProgress(
                i + 1, len(img_paths),
                f"Labeled {i+1}/{len(img_paths)}: {img_path.stem} [skipped: no classes]")
            continue

        results = process_multiple_prompts(predictor, image, text_prompts)

        # Combine into segmentation mask
        segmentation = np.full((h, w), bg_id, dtype=np.uint8)
        priority_map = {cid: idx for idx, cid in enumerate(class_id_priority)}
        prompts_sorted = sorted(
            results.keys(),
            key=lambda p: priority_map.get(prompts_to_class_ids.get(p, float("inf")), float("inf")),
            reverse=True,
        )
        for prompt in prompts_sorted:
            result = results[prompt]
            class_id = prompts_to_class_ids[prompt]
            mask = get_semantic_segmentation_tensor(result["semantic_seg"], h, w, threshold=threshold)
            segmentation[(mask > 0.5).cpu().numpy()] = class_id

        # Save
        PILImage.fromarray(segmentation, mode="L").save(seg_dir / f"{img_path.stem}.png")

        yield StepProgress(i + 1, len(img_paths),
                           f"Labeled {i+1}/{len(img_paths)}: {img_path.stem}{filter_note}")

    yield StepProgress(
        len(img_paths), len(img_paths),
        f"Done: {len(img_paths)} images labeled",
        done=True,
        result={
            "dataset_dir": str(dataset_root),
            "segmentations_dir": str(seg_dir),
            "num_labeled": len(img_paths),
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 4: Train Segmentation Head
# ═════════════════════════════════════════════════════════════════════════════

def step_train(
    dataset_dir: str,
    seg_tag: str,
    seg_layer_config: str,
    num_epochs: int = 50,
    lr: float = 5e-4,
    batch_size: int = 4,
    early_stopping_patience: int = 5,
    seg_scale: int = 4,
    img_size: str = "320,512",
    model_family: str = "dinov3s",
    use_wandb: bool = False,
    wandb_entity: Optional[str] = None,
    wandb_project: str = "sem_map_seg",
    wandb_group: Optional[str] = None,
    train_split: str = "train",
    val_split: str = "val",
    upsampler_type: str = "jafar",
) -> Generator[StepProgress, None, None]:
    """Train segmentation head via subprocess (reuses train_segmentation.py).

    Yields StepProgress updates. Final yield has done=True with result dict.
    """
    import subprocess

    _ensure_scripts_path()
    scripts_dir = _find_sem_map_root() / "scripts"
    train_script = scripts_dir / "sem_map" / "training" / "train_segmentation.py"

    cmd = [
        sys.executable, str(train_script),
        "--train_dataset", dataset_dir,
        "--train_split", train_split,
        "--val_split", val_split,
        "--seg_tag", seg_tag,
        "--seg_layer_config", seg_layer_config,
        "--model_family", model_family,
        "--num_epochs", str(num_epochs),
        "--lr", str(lr),
        "--train_batch_size", str(batch_size),
        "--seg_scale", str(seg_scale),
        "--img_size", img_size,
        "--early_stopping_patience", str(early_stopping_patience),
        "--early_stopping_metric", "iou",
        "--upsampler_type", upsampler_type,
    ]

    if use_wandb:
        cmd.extend(["--use_wandb",
                     "--wandb_project", wandb_project])
        if wandb_entity:
            cmd.extend(["--wandb_entity", wandb_entity])
        if wandb_group:
            cmd.extend(["--wandb_group", wandb_group])

    yield StepProgress(0, num_epochs, "Starting training...")

    env = {**os.environ, "PYTHONPATH": str(scripts_dir)}
    # Disable Rich progress bars in subprocess so output is line-based
    env["TERM"] = "dumb"
    env["NO_COLOR"] = "1"
    # Force a wide console so Rich doesn't wrap long file paths across lines
    # (otherwise the checkpoint-path parser below would see a broken path).
    env["COLUMNS"] = "9999"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line buffered
        cwd=str(_find_workspace_root()),
        env=env,
    )

    checkpoint_path = None
    epoch = 0
    for line in process.stdout:
        line = line.rstrip()
        if not line:
            continue

        # Parse epoch progress
        if "Epoch" in line and "/" in line:
            try:
                parts = line.split("Epoch")[1].split("/")
                epoch = int(parts[0].strip().split()[-1])
            except (IndexError, ValueError):
                pass

        # Capture checkpoint path
        if ".pth" in line:
            for word in line.split():
                w = word.strip("'\",:()[]")
                if w.endswith(".pth") and os.path.sep in w:
                    checkpoint_path = w

        # Yield every line so user sees full output
        yield StepProgress(epoch, num_epochs, line)

    process.wait()

    # Check if training failed
    if process.returncode != 0:
        raise RuntimeError(f"Training failed with exit code {process.returncode}")

    # Validate captured path; fall back to filesystem search if missing/broken.
    # Rich-wrapped paths in stdout could be captured as truthy-but-broken
    # strings, so we always re-verify the captured path actually exists.
    if not checkpoint_path or not Path(checkpoint_path).exists():
        runs_dir = _find_sem_map_root() / "scripts" / "sem_map" / "training" / "runs"
        if runs_dir.exists():
            # Prefer *.best.pth (early-stopping best); fall back to *.pth.
            candidates = sorted(
                runs_dir.glob("*/checkpoints/*.best.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            ) or sorted(
                runs_dir.glob("*/checkpoints/*.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                checkpoint_path = str(candidates[0])

    if not checkpoint_path or not Path(checkpoint_path).exists():
        searched = _find_sem_map_root() / "scripts" / "sem_map" / "training" / "runs"
        raise RuntimeError(
            f"Training completed but no checkpoint found (searched: {searched})"
        )

    # Copy best checkpoint to artifacts
    artifacts_torch = _artifacts_dir() / "torch"
    artifacts_torch.mkdir(parents=True, exist_ok=True)
    artifact_name = f"{model_family}_sem_seg_{seg_tag}.pth"
    dest = artifacts_torch / artifact_name
    shutil.copy2(checkpoint_path, dest)
    checkpoint_path = str(dest)

    yield StepProgress(
        num_epochs, num_epochs,
        f"Training complete. Checkpoint: {checkpoint_path}",
        done=True,
        result={
            "checkpoint_path": checkpoint_path,
            "return_code": 0,
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 5: Export to ONNX
# ═════════════════════════════════════════════════════════════════════════════

def step_onnx_export(
    checkpoint_path: str,
    seg_layer_config: str,
    model_family: str = "dinov3s",
    seg_tag: str = "",
    opset: int = 19,
    dynamic: bool = True,
) -> Generator[StepProgress, None, None]:
    """Export segmentation head to ONNX.

    Yields StepProgress updates. Final yield has done=True with result dict.
    """
    _ensure_scripts_path()

    from sem_map.cli.export_to_onnx import export_segmentation_head_to_onnx
    from sem_map.perception.models.dinov import DINOV_CONFIGS

    cfg = load_seg_layer_config(seg_layer_config)
    num_classes = cfg["num_classes"]
    feat_dim = DINOV_CONFIGS[model_family]["feat_dims"]

    # Feature shape for dummy tracing — dynamic=True makes spatial dims flexible
    feature_shape = (1, feat_dim, 20, 32)

    onnx_dir = _artifacts_dir() / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_name = f"{model_family}_sem_seg_{seg_tag}.onnx"
    onnx_path = onnx_dir / onnx_name

    yield StepProgress(0, 1, f"Exporting to ONNX: {onnx_name}...")

    export_segmentation_head_to_onnx(
        feature_shape=feature_shape,
        num_classes=num_classes,
        checkpoint_path=checkpoint_path,
        output_path=onnx_path,
        opset=opset,
        dynamic=dynamic,
    )

    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    yield StepProgress(
        1, 1,
        f"ONNX exported: {onnx_path.name} ({size_mb:.1f} MB)",
        done=True,
        result={
            "onnx_path": str(onnx_path),
            "size_mb": size_mb,
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 6: Build TensorRT Engine
# ═════════════════════════════════════════════════════════════════════════════

def step_trt_build(
    onnx_path: str,
    seg_tag: str = "",
    model_family: str = "dinov3s",
    seg_scale: int = 4,
    fp16: bool = True,
    workspace_gb: float = 4.0,
    max_batch_size: int = 8,
) -> Generator[StepProgress, None, None]:
    """Build TensorRT engine from ONNX model.

    Yields StepProgress updates. Final yield has done=True with result dict.
    """
    _ensure_scripts_path()
    from sem_map.cli.build_trt_engine import build_engine

    trt_dir = _artifacts_dir() / "trt"
    trt_dir.mkdir(parents=True, exist_ok=True)
    engine_name = f"{model_family}_sem_seg_{seg_tag}.plan"
    engine_path = trt_dir / engine_name

    yield StepProgress(0, 1, f"Building TensorRT engine: {engine_name}...")

    # Seg head input: (1, feat_dim, H, W) — static shape for faster build
    from sem_map.perception.models.dinov import DINOV_CONFIGS
    feat_dim = DINOV_CONFIGS[model_family]["feat_dims"]
    # seg_scale=4 on base 20x32 = 80x128
    h, w = 20 * seg_scale, 32 * seg_scale
    min_shape = (1, feat_dim, h, w)
    opt_shape = (1, feat_dim, h, w)
    max_shape = (1, feat_dim, h, w)

    build_engine(
        onnx_path=onnx_path,
        engine_path=str(engine_path),
        fp16=fp16,
        workspace_gb=workspace_gb,
        batch=None,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
        input_name="features",
    )

    size_mb = engine_path.stat().st_size / (1024 * 1024) if engine_path.exists() else 0

    yield StepProgress(
        1, 1,
        f"TRT engine built: {engine_path.name} ({size_mb:.1f} MB)",
        done=True,
        result={
            "engine_path": str(engine_path),
            "size_mb": size_mb,
        },
    )


# ═════════════════════════════════════════════════════════════════════════════
# STEP 7: Inference (model → rgb / segmentation / traversability videos)
# ═════════════════════════════════════════════════════════════════════════════

def list_model_families() -> list[str]:
    """List configured model families from models.yaml (best-effort)."""
    _ensure_scripts_path()
    try:
        from sem_map.perception.models import load_models_yaml
        return sorted((load_models_yaml().get("families") or {}).keys())
    except Exception:
        return []


def list_seg_layers_for_family(family: str) -> list[str]:
    """List seg_layer keys configured under a family in models.yaml."""
    _ensure_scripts_path()
    try:
        from sem_map.perception.models import load_models_yaml
        fam = load_models_yaml()["families"][family]
        return sorted((fam.get("seg_layers") or {}).keys())
    except Exception:
        return []


def iter_frames_from_source(
    source: str,
    topic: str = "",
    every_nth: int = 1,
    rotate_180: bool = False,
):
    """Yield ``(out_idx, PIL.Image RGB, total_estimate, source_fps)``.

    Source autodetection:
      - glob (contains ``*?[``) → image glob
      - directory → all images inside
      - ``*.mcap`` file → frames from ``topic``
      - video extension → OpenCV decode
      - image extension → single image
    ``total_estimate`` is 0 when unknown.
    ``source_fps`` is set for videos; None otherwise (consumer should default).
    """
    from PIL import Image
    import glob as _glob

    step = max(1, int(every_nth))

    # 1. Glob pattern
    if any(ch in source for ch in ("*", "?", "[")):
        paths = sorted(p for p in _glob.glob(source) if is_image_file(p))
        total = (len(paths) + step - 1) // step
        out_idx = 0
        for i, p in enumerate(paths):
            if i % step:
                continue
            img = Image.open(p).convert("RGB")
            if rotate_180:
                img = img.transpose(Image.ROTATE_180)
            yield out_idx, img, total, None
            out_idx += 1
        return

    src_path = Path(source)

    # 2. Directory
    if src_path.is_dir():
        paths: list[Path] = []
        for ext in IMAGE_EXTENSIONS:
            paths.extend(src_path.glob(f"*{ext}"))
        paths.sort()
        total = (len(paths) + step - 1) // step
        out_idx = 0
        for i, p in enumerate(paths):
            if i % step:
                continue
            img = Image.open(p).convert("RGB")
            if rotate_180:
                img = img.transpose(Image.ROTATE_180)
            yield out_idx, img, total, None
            out_idx += 1
        return

    # 3. File
    if not src_path.is_file():
        raise FileNotFoundError(f"Source not found: {source}")

    suffix = src_path.suffix.lower()

    # 3a. Single image
    if suffix in IMAGE_EXTENSIONS:
        img = Image.open(src_path).convert("RGB")
        if rotate_180:
            img = img.transpose(Image.ROTATE_180)
        yield 0, img, 1, None
        return

    # 3b. Video
    if suffix in VIDEO_EXTENSIONS:
        import cv2
        cap = cv2.VideoCapture(str(src_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {source}")
        # Apply container rotation metadata (e.g. .mov from phones/action cams),
        # so decoded frames match how players display them.
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        total = (total_frames + step - 1) // step if total_frames else 0
        try:
            i = 0
            out_idx = 0
            while True:
                ret, bgr = cap.read()
                if not ret:
                    break
                if i % step == 0:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    if rotate_180:
                        rgb = np.ascontiguousarray(np.flipud(np.fliplr(rgb)))
                    yield out_idx, Image.fromarray(rgb), total, fps
                    out_idx += 1
                i += 1
        finally:
            cap.release()
        return

    # 3c. MCAP
    if suffix == ".mcap":
        if not topic:
            raise ValueError("MCAP input requires a topic")
        sys.path.insert(0, str(_find_sem_map_root() / "tools" / "mcap"))
        from generate_rgb_from_mcap import (
            iterate_image_messages,
            compressed_image_to_rgb_array,
            ros_image_to_rgb_array,
        )
        try:
            infos = list_mcap_topics(str(src_path))
            total_msgs = next(
                (t["message_count"] for t in infos if t["topic"] == topic), 0)
        except Exception:
            total_msgs = 0
        total = (total_msgs + step - 1) // step if total_msgs else 0

        i = 0
        out_idx = 0
        for _ts, ros_msg in iterate_image_messages(str(src_path), topic):
            if i % step == 0:
                is_compressed = (hasattr(ros_msg, "format")
                                 and not hasattr(ros_msg, "step"))
                rgb = (compressed_image_to_rgb_array(ros_msg) if is_compressed
                       else ros_image_to_rgb_array(ros_msg))
                if rotate_180:
                    rgb = np.ascontiguousarray(np.flipud(np.fliplr(rgb)))
                yield out_idx, Image.fromarray(rgb), total, None
                out_idx += 1
            i += 1
        return

    raise ValueError(f"Unsupported source extension: {suffix} ({source})")


def _build_seg_color_lut(categories: list[dict]) -> np.ndarray:
    """Build a (256,3) RGB uint8 LUT from seg_layer categories."""
    import hashlib
    lut = np.zeros((256, 3), dtype=np.uint8)
    for cat in categories:
        cid = int(cat["id"])
        if not (0 <= cid < 256):
            continue
        color = cat.get("color")
        if color is None:
            h = int(hashlib.md5(str(cid).encode()).hexdigest()[:6], 16)
            color = [(h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF]
        lut[cid] = color
    return lut


def step_inference(
    source: str,
    output_dir: str,
    *,
    family: str = "dinov3s",
    backbone_backend: str = "trt",
    upsampler_mode: str = "jafar_full",
    upsampler_backend: str = "trt",
    upsample_partial_factor: int = 16,
    seg_layer: str = "",
    seg_backend: str = "trt",
    temperature: float = 0.5,
    latent_w: int = 32,
    latent_h: int = 20,
    topic: str = "",
    every_nth: int = 1,
    rotate_180: bool = False,
    fps: float = 10.0,
    overlay_alpha: float = 0.0,
) -> Generator[StepProgress, None, None]:
    """Run inference over a source and write rgb / segmentation / traversability mp4s.

    Streams frame-by-frame; no intermediate disk writes besides the output videos.

    Yields StepProgress updates. Final yield has done=True with result dict
    containing output paths and frame count.
    """
    _ensure_scripts_path()

    if not seg_layer:
        raise ValueError("seg_layer is required")
    if backbone_backend not in ("torch", "trt"):
        raise ValueError(f"backbone_backend: torch|trt, got {backbone_backend}")
    if upsampler_mode not in ("bilinear", "jafar_full", "jafar_partial"):
        raise ValueError(f"upsampler_mode: bilinear|jafar_full|jafar_partial, "
                         f"got {upsampler_mode}")
    if upsampler_backend not in ("torch", "onnx", "trt"):
        raise ValueError(f"upsampler_backend: torch|onnx|trt, got {upsampler_backend}")
    if seg_backend not in ("torch", "trt"):
        raise ValueError(f"seg_backend: torch|trt, got {seg_backend}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    import torch

    from sem_map.perception import prepare_image, get_device
    from sem_map.perception.models import (
        load_models_yaml, resolve_artifact_path, get_num_classes, get_categories,
    )
    from sem_map.perception.models.dinov import (
        load_dinovx_torch, compute_dinovx_feats_from_batch_torch,
        compute_dinovx_bilinear_upsampled_feats_from_batch,
        load_dinovx_trt, compute_dinovx_feats_from_batch_trt,
    )
    from sem_map.perception.models.jafar import (
        load_jafar_torch, compute_jafar_upscaled_feats_from_batch_torch,
        load_jafar_trt, compute_jafar_upscaled_feats_from_batch_trt,
    )
    from sem_map.perception.models.segmentation import (
        load_seg_head_torch, run_seg_from_batch_torch,
        load_seg_head_trt, run_seg_from_batch_trt,
        traversability_from_probs,
    )

    yield StepProgress(0, 1, "Loading models.yaml...")
    models_cfg = load_models_yaml()
    if family not in models_cfg.get("families", {}):
        raise ValueError(f"Unknown family '{family}'. "
                         f"Available: {sorted(models_cfg.get('families', {}).keys())}")
    fam_cfg = models_cfg["families"][family]
    patch_size = int(fam_cfg["backbone"]["patch_size"])
    feature_dim = int(fam_cfg["backbone"]["feat_dims"])

    # Resolve artifact paths
    backbone_path = None
    if backbone_backend == "trt":
        backbone_path = resolve_artifact_path(family, "backbone", "trt")
        if backbone_path is None or not Path(backbone_path).exists():
            raise FileNotFoundError(
                f"Backbone TRT engine not found for family '{family}': {backbone_path}")

    jafar_path = None
    if upsampler_mode in ("jafar_full", "jafar_partial") and upsampler_backend == "trt":
        f_factor = 16 if upsampler_mode == "jafar_full" else int(upsample_partial_factor)
        jafar_path = resolve_artifact_path(
            family, "upsampler", "trt",
            upsampler="jafar", upsample_factor=f_factor,
        )
        if jafar_path is None or not Path(jafar_path).exists():
            raise FileNotFoundError(
                f"JAFAR TRT engine x{f_factor} not found for family '{family}'")

    seg_path = resolve_artifact_path(family, "seg_layer", seg_backend, seg_layer=seg_layer)
    if seg_path is None or not Path(seg_path).exists():
        raise FileNotFoundError(
            f"Seg head ({seg_backend}) not found for '{seg_layer}': {seg_path}")

    device = get_device()

    # Load backbone
    yield StepProgress(0, 1, f"Loading backbone {family} ({backbone_backend})...")
    dino_model = trt_back = None
    if backbone_backend == "torch":
        dino_model, _ = load_dinovx_torch(family)
    else:
        trt_back = load_dinovx_trt(str(backbone_path))

    # Load upsampler (if jafar_*)
    jafar_model = trt_jafar = None
    if upsampler_mode in ("jafar_full", "jafar_partial"):
        yield StepProgress(0, 1, f"Loading JAFAR upsampler ({upsampler_backend})...")
        if upsampler_backend == "torch":
            jafar_model, _ = load_jafar_torch(
                fam_cfg["upsamplers"]["jafar"]["torch_variant"])
        elif upsampler_backend == "trt":
            trt_jafar = load_jafar_trt(str(jafar_path))
        else:
            raise NotImplementedError("upsampler_backend=onnx not wired here")

    # Load seg head
    yield StepProgress(0, 1, f"Loading seg head '{seg_layer}' ({seg_backend})...")
    seg_head_torch = seg_trt = None
    if seg_backend == "torch":
        n_classes = int(get_num_classes(family, seg_layer) or 27)
        seg_head_torch = load_seg_head_torch(
            model_feature_dim=feature_dim,
            num_classes=n_classes,
            model_name=family,
            checkpoint_dir=str(seg_path),
        )
    else:
        seg_trt = load_seg_head_trt(str(seg_path))

    # Color LUT for segmentation visualization
    categories = get_categories(family, seg_layer)
    lut = _build_seg_color_lut(categories)

    # Lazy-init video writers on the first frame (need known H, W)
    rgb_path = out_dir / "rgb.mp4"
    seg_path_out = out_dir / "segmentation.mp4"
    trav_path = out_dir / "traversability.mp4"
    rgb_writer = seg_writer = trav_writer = None
    write_overlays = float(overlay_alpha) > 0.0
    alpha = float(max(0.0, min(1.0, overlay_alpha)))
    seg_overlay_path = out_dir / "segmentation_overlay.mp4"
    trav_overlay_path = out_dir / "traversability_overlay.mp4"
    seg_overlay_writer = trav_overlay_writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    yield StepProgress(0, 0, f"Starting inference: {Path(source).name}")
    n_frames = 0
    last_total = 0
    try:
        for out_idx, img, total, src_fps in iter_frames_from_source(
            source, topic=topic, every_nth=every_nth, rotate_180=rotate_180,
        ):
            last_total = total
            unorm, batch, in_h, in_w = prepare_image(
                image=img,
                latent_w=latent_w,
                latent_h=latent_h,
                patch_size=patch_size,
                device=device,
            )

            with torch.inference_mode():
                # Backbone
                if backbone_backend == "trt":
                    lr_feats = compute_dinovx_feats_from_batch_trt(
                        trt_back, batch, latent_w, latent_h)
                else:
                    lr_feats = compute_dinovx_feats_from_batch_torch(
                        image_batch=batch, model=dino_model,
                        latent_w=latent_w, latent_h=latent_h,
                    )

                # Upsampler — produces hr_feats at partial resolution for
                # bilinear (factor<patch_size) and jafar_partial. Seg head
                # runs at hr_feats resolution; outputs are upsampled to
                # (in_h, in_w) afterwards (mirrors SemanticPipeline).
                if upsampler_mode == "bilinear":
                    f = max(1, int(upsample_partial_factor))
                    scale = max(1, int(round(float(patch_size) / float(f))))
                    tgt_h = max(1, in_h // scale)
                    tgt_w = max(1, in_w // scale)
                    feats = compute_dinovx_bilinear_upsampled_feats_from_batch(
                        lr_feats, tgt_h, tgt_w)
                else:
                    if upsampler_mode == "jafar_full":
                        scale = 1
                    else:
                        f_factor = int(upsample_partial_factor)
                        scale = max(1, int(round(
                            float(patch_size) / float(max(1, f_factor)))))
                    if scale > 1:
                        batch_ds = torch.nn.functional.interpolate(
                            batch,
                            size=(max(1, in_h // scale), max(1, in_w // scale)),
                            mode="bilinear", align_corners=False,
                        )
                    else:
                        batch_ds = batch

                    if trt_jafar is not None:
                        feats = compute_jafar_upscaled_feats_from_batch_trt(
                            trt_jafar, batch_ds, lr_feats)
                    else:
                        feats = compute_jafar_upscaled_feats_from_batch_torch(
                            image_batch=batch_ds, model=jafar_model, lr_feats=lr_feats)

                # Seg head (runs at hr_feats resolution — possibly partial)
                if seg_backend == "trt":
                    seg, _logits, probs = run_seg_from_batch_trt(
                        seg_trt,
                        feats.unsqueeze(0) if feats.ndim == 3 else feats,
                        temperature,
                        return_probs=True,
                    )
                else:
                    if isinstance(feats, torch.Tensor) and seg_head_torch is not None:
                        try:
                            t_dt = next(seg_head_torch.parameters()).dtype
                            if feats.dtype != t_dt:
                                feats = feats.to(dtype=t_dt)
                        except StopIteration:
                            pass
                    seg, logits = run_seg_from_batch_torch(
                        features=feats, seg_head=seg_head_torch, temperature=temperature,
                    )
                    t_val = float(temperature) if temperature else 1.0
                    probs = torch.nn.functional.softmax(logits / t_val, dim=1)

                trav = traversability_from_probs(
                    probs, family, seg_layer)[0].clamp(0, 1)

                # Upsample seg/trav to full input resolution when seg ran
                # at partial resolution.
                if upsampler_mode in ("jafar_partial", "bilinear") and \
                        feats.shape[-2:] != (in_h, in_w):
                    seg = torch.nn.functional.interpolate(
                        seg.unsqueeze(1).float(),
                        size=(in_h, in_w),
                        mode="nearest-exact",
                    ).squeeze(1).to(seg.dtype)
                    trav = torch.nn.functional.interpolate(
                        trav.unsqueeze(0).unsqueeze(0),
                        size=(in_h, in_w),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0).squeeze(0)

            # Build frames (BGR for cv2.VideoWriter)
            rgb_u8 = (unorm.detach().cpu().clamp(0, 1).float().numpy() * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

            seg_ids = seg[0].detach().cpu().to(torch.uint8).numpy()
            seg_rgb = lut[seg_ids]
            seg_bgr = cv2.cvtColor(seg_rgb, cv2.COLOR_RGB2BGR)

            trav_np = trav.detach().cpu().float().numpy()
            trav_u8 = (np.clip(trav_np, 0.0, 1.0) * 255).astype(np.uint8)
            trav_color = cv2.applyColorMap(trav_u8, cv2.COLORMAP_VIRIDIS)

            if rgb_writer is None:
                H, W = rgb_bgr.shape[:2]
                rgb_writer = cv2.VideoWriter(str(rgb_path), fourcc, fps, (W, H))
                sH, sW = seg_bgr.shape[:2]
                seg_writer = cv2.VideoWriter(str(seg_path_out), fourcc, fps, (sW, sH))
                tH, tW = trav_color.shape[:2]
                trav_writer = cv2.VideoWriter(str(trav_path), fourcc, fps, (tW, tH))
                if write_overlays:
                    seg_overlay_writer = cv2.VideoWriter(
                        str(seg_overlay_path), fourcc, fps, (W, H))
                    trav_overlay_writer = cv2.VideoWriter(
                        str(trav_overlay_path), fourcc, fps, (W, H))

            rgb_writer.write(rgb_bgr)
            seg_writer.write(seg_bgr)
            trav_writer.write(trav_color)
            if write_overlays:
                seg_over = cv2.addWeighted(seg_bgr, alpha, rgb_bgr, 1.0 - alpha, 0.0)
                trav_over = cv2.addWeighted(trav_color, alpha, rgb_bgr, 1.0 - alpha, 0.0)
                seg_overlay_writer.write(seg_over)
                trav_overlay_writer.write(trav_over)

            n_frames += 1
            if n_frames % 5 == 0:
                msg = f"Inferred {n_frames}" + (f"/{total}" if total else "")
                yield StepProgress(n_frames, total, msg)
    finally:
        for w in (rgb_writer, seg_writer, trav_writer,
                  seg_overlay_writer, trav_overlay_writer):
            if w is not None:
                w.release()

    result = {
        "output_dir": str(out_dir),
        "rgb_path": str(rgb_path),
        "seg_path": str(seg_path_out),
        "trav_path": str(trav_path),
        "num_frames": n_frames,
    }
    if write_overlays:
        result["seg_overlay_path"] = str(seg_overlay_path)
        result["trav_overlay_path"] = str(trav_overlay_path)
    yield StepProgress(
        n_frames, last_total or n_frames,
        f"Done: {n_frames} frames → {out_dir}",
        done=True,
        result=result,
    )
