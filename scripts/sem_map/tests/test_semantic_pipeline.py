# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import time
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

from sem_map.parameter import Parameter
from sem_map.semantic_pipeline import SemanticPipeline


def make_blob_raw(image_path: str | None = None) -> tuple[bytes, int, int, int]:
    if image_path and Path(image_path).exists():
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        arr = np.asarray(img, dtype=np.uint8)
        return arr.tobytes(), h, w, 3

    # Fallback: synthetic horizontal gradient (any reasonable HxW)
    h, w = 640, 960
    x = np.linspace(0, 255, w, dtype=np.uint8)
    img = np.tile(x, (h, 1))
    img = np.stack([img] * 3, axis=-1)
    return img.tobytes(), h, w, 3


def run_dataset(param: Parameter, images_dir: str, start: int, end: int, step: int):
    pipeline = SemanticPipeline(param)

    indices = range(start, end, step)
    for frame_idx in tqdm(indices, desc="SemanticPipeline", unit="frame"):
        idx_with_zeros = f"{frame_idx:05d}"
        image_path = f"{images_dir}/{idx_with_zeros}.png"
        blob, h, w, c = make_blob_raw(image_path=image_path)
        now_ns = int(time.time_ns())
        pipeline.process_frame(blob, h, w, c, sensor_name="test_cam", stamp_ns=now_ns)


def run_perf(param: Parameter, num_iters: int = 100):
    param.set_value('enable_pca', False)

    pipeline = SemanticPipeline(param)

    # Prepare one synthetic frame and reuse it to avoid I/O and allocation overheads
    blob, h, w, c = make_blob_raw(image_path=None)

    latencies_ns: list[int] = []
    t_begin_ns = time.perf_counter_ns()
    for _ in range(num_iters):
        t0 = time.perf_counter_ns()
        now_ns = int(time.time_ns())
        pipeline.process_frame(blob, h, w, c, sensor_name="test_cam", stamp_ns=now_ns)
        t1 = time.perf_counter_ns()
        latencies_ns.append(t1 - t0)
    t_end_ns = time.perf_counter_ns()

    total_ns = t_end_ns - t_begin_ns
    avg_latency_ms = (sum(latencies_ns) / max(len(latencies_ns), 1)) / 1e6
    fps = (num_iters / (total_ns / 1e9)) if total_ns > 0 else float("inf")

    # Single concise line: average latency and overall throughput
    print(f"process_frame avg latency: {avg_latency_ms:.3f} ms; throughput: {fps:.2f} FPS")


def main():
    p = argparse.ArgumentParser(description="SemanticPipeline dataset smoke test (viz_seg-like)")
    p.add_argument("--images_dir", type=str, default="data/custom_datasets/example/images")
    p.add_argument("--start", type=int, default=400)
    p.add_argument("--end", type=int, default=2000)
    p.add_argument("--step", type=int, default=15)
    p.add_argument("--perf", action="store_true", help="Run process_frame performance test with synthetic frames")
    p.add_argument("--iters", type=int, default=100, help="Number of frames to process in perf test")
    args = p.parse_args()

    param = Parameter()
    param.set_value('enable_pca', False)
    if args.perf:
        run_perf(param, num_iters=args.iters)
    else:
        run_dataset(param, images_dir=args.images_dir, start=args.start, end=args.end, step=args.step)


if __name__ == "__main__":
    main()
    print("\u2713 semantic_pipeline dataset test completed")
