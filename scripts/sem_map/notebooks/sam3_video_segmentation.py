# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Run SAM3 streaming video segmentation on a video file and write an overlay MP4.

Usage:
    python sam3_video_segmentation.py \
        --input  /path/to/in.mp4 \
        --output /path/to/out.mp4 \
        --prompt "bike robot" \
        --duration 30
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def _add_scripts_to_path() -> None:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        if (parent / "package.xml").exists():
            scripts_root = parent / "scripts"
            if scripts_root.exists() and str(scripts_root) not in sys.path:
                sys.path.insert(0, str(scripts_root))
            return


_add_scripts_to_path()

from sem_map.perception.models.sam3 import (  # noqa: E402
    init_video_streaming_session,
    load_sam3_video_model,
    process_streaming_frame,
    process_video,
)


_PALETTE = np.array(
    [
        (255, 56, 56), (56, 255, 56), (56, 56, 255),
        (255, 255, 56), (255, 56, 255), (56, 255, 255),
        (255, 165, 0), (160, 80, 255), (0, 200, 120), (240, 120, 60),
    ],
    dtype=np.uint8,
)


def _color_for(obj_id: int) -> np.ndarray:
    return _PALETTE[int(obj_id) % len(_PALETTE)]


def _to_numpy_mask(m) -> np.ndarray:
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    return np.asarray(m).astype(bool)


def _overlay(frame_bgr: np.ndarray, masks, object_ids, boxes, scores, alpha: float = 0.5) -> np.ndarray:
    if masks is None or len(masks) == 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()

    masks_np = _to_numpy_mask(masks)
    if masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    if masks_np.shape[-2:] != (h, w):
        resized = np.zeros((masks_np.shape[0], h, w), dtype=bool)
        for i, m in enumerate(masks_np):
            resized[i] = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        masks_np = resized

    obj_ids = list(object_ids) if object_ids is not None else list(range(masks_np.shape[0]))
    boxes_np = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else (
        np.asarray(boxes) if boxes is not None else None
    )
    scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else (
        np.asarray(scores) if scores is not None else None
    )

    for i in range(masks_np.shape[0]):
        m = masks_np[i]
        if not m.any():
            continue
        obj_id = int(obj_ids[i]) if i < len(obj_ids) else i
        color = _color_for(obj_id)  # RGB
        color_bgr = (int(color[2]), int(color[1]), int(color[0]))

        colored = np.zeros_like(out)
        colored[m] = color_bgr
        out = np.where(m[..., None], (out * (1 - alpha) + colored * alpha).astype(np.uint8), out)

        if boxes_np is not None and i < len(boxes_np):
            x1, y1, x2, y2 = [int(v) for v in boxes_np[i][:4]]
            cv2.rectangle(out, (x1, y1), (x2, y2), color_bgr, 2)
            label = f"id={obj_id}"
            if scores_np is not None and i < len(scores_np):
                label += f" {float(scores_np[i]):.2f}"
            cv2.putText(out, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2, cv2.LINE_AA)

    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt", default="bike robot")
    p.add_argument("--start", type=float, default=0.0,
                   help="Start time in seconds (default 0).")
    p.add_argument("--end", type=float, default=0.0,
                   help="End time in seconds. <=0 means use --duration "
                        "(or run to end of video if duration<=0).")
    p.add_argument("--duration", type=float, default=30.0,
                   help="Seconds to process from --start. <=0 means run to end "
                        "of video. Ignored if --end is set.")
    p.add_argument("--backend", choices=("streaming", "preloaded"), default="streaming",
                   help="streaming = init_video_streaming_session + per-frame call "
                        "(constant memory). preloaded = collect all frames in memory "
                        "and call process_video (better tracking, holds everything in RAM).")
    p.add_argument("--stride", type=int, default=1,
                   help="Process every Nth frame (1 = every frame). Output fps is "
                        "scaled to input_fps/stride so playback stays real-time.")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--device", default=None)
    p.add_argument("--image-size", type=int, default=0,
                   help="If >0, override SAM3's internal input size (default 1008) "
                        "via Sam3VideoConfig.image_size. Lower = less memory/compute, "
                        "less accuracy. Try 560 if you OOM. This is the official "
                        "knob — prefer it over --infer-width.")
    p.add_argument("--infer-width", type=int, default=0,
                   help="Pre-resize frames to this width before handing to the "
                        "processor (does NOT change SAM3's internal resolution). "
                        "Mostly cosmetic — use --image-size for real memory savings. "
                        "0 = no pre-resize.")
    p.add_argument("--output-scale", type=float, default=1.0,
                   help="Scale factor for the output video relative to input "
                        "(e.g. 0.5 = half-res output). Independent of --infer-width.")
    args = p.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"Failed to open: {in_path}", file=sys.stderr)
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(round(args.start * fps)))
    if start_frame >= total:
        print(f"--start {args.start}s is past end of video ({total/fps:.1f}s)", file=sys.stderr)
        cap.release()
        return 1
    if args.end > 0:
        end_frame = min(total, int(round(args.end * fps)))
    elif args.duration > 0:
        end_frame = min(total, start_frame + int(round(args.duration * fps)))
    else:
        end_frame = total
    max_frames = max(0, end_frame - start_frame)
    if max_frames == 0:
        print("Empty range (end <= start).", file=sys.stderr)
        cap.release()
        return 1

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Input  : {in_path}  ({w}x{h} @ {fps:.2f}fps, {total} frames)")
    print(f"Output : {out_path}")
    print(f"Range  : frames [{start_frame}, {end_frame})  "
          f"({args.start:.2f}s -> {end_frame/fps:.2f}s, {max_frames} frames)")
    print(f"Prompt : {args.prompt!r}")

    print(f"Backend: {args.backend}")
    if args.image_size > 0:
        print(f"Model image_size override: {args.image_size}  (default 1008)")
    print("Loading SAM3 video model ...")
    predictor = load_sam3_video_model(
        device=args.device,
        image_size=args.image_size if args.image_size > 0 else None,
    )

    out_w = max(1, int(round(w * args.output_scale)))
    out_h = max(1, int(round(h * args.output_scale)))
    if args.infer_width > 0:
        infer_w = args.infer_width
        infer_h = max(1, int(round(h * infer_w / w)))
        print(f"Inference resolution: {infer_w}x{infer_h}  (native {w}x{h})")
    else:
        infer_w = infer_h = 0
    if (out_w, out_h) != (w, h):
        print(f"Output resolution: {out_w}x{out_h}")

    stride = max(1, int(args.stride))
    out_fps = fps / stride
    if stride > 1:
        print(f"Stride: processing every {stride} frames -> output fps {out_fps:.2f}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        print("Failed to open VideoWriter", file=sys.stderr)
        return 1

    def _scale_boxes(boxes, src_w, src_h):
        if boxes is None:
            return None
        sx = out_w / src_w
        sy = out_h / src_h
        if sx == 1.0 and sy == 1.0:
            return boxes
        if isinstance(boxes, torch.Tensor):
            b = boxes.detach().cpu().float().clone()
        else:
            b = np.asarray(boxes, dtype=np.float32).copy()
        b[:, [0, 2]] *= sx
        b[:, [1, 3]] *= sy
        return b

    def _render_and_write(draw_frame_bgr, out_dict, src_box_w, src_box_h):
        boxes = _scale_boxes(out_dict.get("boxes"), src_box_w, src_box_h)
        overlay = _overlay(
            draw_frame_bgr,
            masks=out_dict.get("masks"),
            object_ids=out_dict.get("object_ids"),
            boxes=boxes,
            scores=out_dict.get("scores"),
            alpha=args.alpha,
        )
        writer.write(overlay)

    def _prep_infer_and_draw(frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb_infer = cv2.resize(rgb, (infer_w, infer_h), interpolation=cv2.INTER_AREA) \
            if infer_w > 0 else rgb
        if (out_w, out_h) != (w, h):
            draw_frame = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            draw_frame = frame_bgr
        return rgb_infer, draw_frame

    t0 = time.time()
    try:
        if args.backend == "streaming":
            session = init_video_streaming_session(predictor, text_prompts=args.prompt)
            processed = 0
            for idx in range(max_frames):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % stride != 0:
                    continue

                rgb_infer, draw_frame = _prep_infer_and_draw(frame_bgr)
                out_dict = process_streaming_frame(predictor, session, Image.fromarray(rgb_infer))

                src_box_w = infer_w if infer_w > 0 else w
                src_box_h = infer_h if infer_h > 0 else h
                _render_and_write(draw_frame, out_dict, src_box_w, src_box_h)
                processed += 1

                if processed % 30 == 0 or idx + 1 == max_frames:
                    elapsed = time.time() - t0
                    rate = processed / max(elapsed, 1e-6)
                    remaining_to_process = max(0, (max_frames - idx - 1)) // stride
                    eta = remaining_to_process / max(rate, 1e-6)
                    n_objs = 0 if out_dict.get("object_ids") is None else len(out_dict["object_ids"])
                    print(f"  src {idx+1}/{max_frames}  processed={processed}  objs={n_objs}  "
                          f"{rate:.2f} infer-fps  eta={eta:5.1f}s")

        else:  # preloaded
            print("Decoding inference frames into memory ...")
            infer_frames: list[np.ndarray] = []
            for idx in range(max_frames):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % stride != 0:
                    continue
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                rgb_infer = cv2.resize(rgb, (infer_w, infer_h), interpolation=cv2.INTER_AREA) \
                    if infer_w > 0 else rgb
                infer_frames.append(rgb_infer)

            n = len(infer_frames)
            if n == 0:
                print("No frames decoded.", file=sys.stderr)
                return 1
            ih, iw = infer_frames[0].shape[:2]
            mem_mb = sum(f.nbytes for f in infer_frames) / 1e6
            print(f"Decoded {n} frames ({iw}x{ih} for inference); ~{mem_mb:.0f} MB in RAM "
                  "(draw frames re-read from video at render time)")
            print("Running process_video ...")
            outputs_per_frame = process_video(
                predictor,
                video_frames=infer_frames,
                text_prompts=args.prompt,
                show_progress_bar=True,
            )
            # Free inference frames before render pass.
            del infer_frames
            print(f"Inference done in {time.time()-t0:.1f}s; rendering ...")

            src_box_w = iw
            src_box_h = ih
            cap.release()
            cap = cv2.VideoCapture(str(in_path))
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            rendered = 0
            for idx in range(max_frames):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if idx % stride != 0:
                    continue
                if (out_w, out_h) != (w, h):
                    draw_frame = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
                else:
                    draw_frame = frame_bgr

                out_dict = outputs_per_frame.get(rendered)
                if out_dict is None:
                    writer.write(draw_frame)
                else:
                    _render_and_write(draw_frame, out_dict, src_box_w, src_box_h)
                rendered += 1
                if rendered % 30 == 0 or rendered == n:
                    n_objs = 0 if (out_dict is None or out_dict.get("object_ids") is None) \
                        else len(out_dict["object_ids"])
                    print(f"  render {rendered}/{n}  objs={n_objs}")
    finally:
        cap.release()
        writer.release()

    print(f"Done in {time.time()-t0:.1f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
