#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
Generate label-budget k-means training splits for the trail dataset, matching
the Rellis label-efficiency protocol: the **validation set is fixed** and only
the **training set size varies** with k.

For each k, we run MiniBatchKMeans over the CLS features of the **training pool
only** (the images in ``train_images.txt``), pick the image nearest each
centroid (deduplicated), and write ``train_kmeans_k{k}_images.txt``. The
existing ``val_images.txt`` is never touched — every budget validates on the
same held-out set, so mIoU is comparable across k.

(This differs from ``pipeline_steps.step_faiss_select``, which selects k images
total and re-splits *those* into train/val — that would shrink val with k.)

Usage
─────
    python make_trail_kmeans_splits.py \
        --dataset_root data/custom_datasets/trail_dataset/vae_trail \
        --k 20 50 100 200 500 --seed 42
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True,
                    help="Dataset dir with images/, train_images.txt, val_images.txt, "
                         "faiss_cls_features.npy, faiss_cls_paths.npy")
    ap.add_argument("--k", type=int, nargs="+", default=[20, 50, 100, 200, 500])
    ap.add_argument("--seed", type=int, default=42,
                    help="K-means seed (matches Rellis selection determinism)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Rewrite split files even if they already exist")
    args = ap.parse_args()

    from sklearn.cluster import MiniBatchKMeans
    import faiss

    root = Path(args.dataset_root)
    feats = np.load(root / "faiss_cls_features.npy").astype(np.float32)
    paths = np.load(root / "faiss_cls_paths.npy", allow_pickle=True).tolist()
    stems = [Path(p).stem for p in paths]
    stem_to_row = {s: i for i, s in enumerate(stems)}

    # Restrict to the TRAIN POOL only (so we never select a val image into train).
    train_pool = [l.strip() for l in (root / "train_images.txt").read_text().splitlines() if l.strip()]
    val_pool = [l.strip() for l in (root / "val_images.txt").read_text().splitlines() if l.strip()]
    rows = [stem_to_row[s] for s in train_pool if s in stem_to_row]
    missing = len(train_pool) - len(rows)
    pool_feats = feats[rows]
    pool_stems = [train_pool[i] for i, s in enumerate(train_pool) if s in stem_to_row]

    print(f"Dataset: {root}")
    print(f"  train pool: {len(train_pool)} (features found for {len(rows)}, missing {missing})")
    print(f"  val (fixed, untouched): {len(val_pool)}")
    print(f"  feature dim: {pool_feats.shape[1]}")

    for k in args.k:
        out = root / f"train_kmeans_k{k}_images.txt"
        if out.exists() and not args.overwrite:
            print(f"  k={k}: exists, skip ({out.name})")
            continue
        actual_k = min(k, len(pool_feats))
        km = MiniBatchKMeans(n_clusters=actual_k, random_state=args.seed,
                             n_init=5, batch_size=512)
        km.fit(pool_feats)
        centroids = km.cluster_centers_.astype(np.float32)
        index = faiss.IndexFlatL2(pool_feats.shape[1])
        index.add(pool_feats)
        _, nn_idx = index.search(centroids, 1)
        sel = list(dict.fromkeys(nn_idx[:, 0].tolist()))   # dedup, keep order
        sel_stems = sorted(pool_stems[i] for i in sel)
        out.write_text("\n".join(sel_stems) + "\n")
        print(f"  k={k}: selected {len(sel_stems)} train images -> {out.name}")

    print("\nDone. Val set left as-is (train_split=train_kmeans_kN, val_split=val).")


if __name__ == "__main__":
    main()
