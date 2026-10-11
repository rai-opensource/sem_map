# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Streamlit app for SAM3 semantic segmentation with configurable class labels and parameters."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
import streamlit as st
import plotly.express as px

# Detect workspace root and add scripts to path
try:
    current_file = Path(__file__).resolve()
except NameError:
    current_file = Path.cwd()

repo_root = None
for parent in current_file.parents:
    if (parent / "package.xml").exists():
        repo_root = parent
        break
if repo_root is None:
    repo_root = Path(os.environ.get("SEM_MAP_WS_ROOT", Path.cwd()))

SCRIPTS_ROOT = str(repo_root / "scripts")
if not (Path(SCRIPTS_ROOT) / "sem_map" / "parameter.py").exists():
    raise RuntimeError(
        f"Could not find sem_map scripts directory (looked in {SCRIPTS_ROOT}); "
        "set SEM_MAP_WS_ROOT to the sem_map repo root"
    )
if SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, SCRIPTS_ROOT)

from sem_map.perception.models.sam3 import (
    load_sam3_model,
    process_multiple_prompts,
    process_with_exemplars,
    get_semantic_segmentation_tensor,
    overlay_mask_on_image_numpy,
    create_combined_overlay,
)


@st.cache_resource(show_spinner=False)
def init_sam3_model():
    predictor = load_sam3_model()
    return predictor


def parse_class_labels(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if len(lines) == 1 and "," in lines[0]:
        return [p.strip() for p in lines[0].split(",") if p.strip()]
    return lines


def parse_boxes_per_prompt(
    boxes_text: str, num_classes: int,
) -> tuple[list | None, list | None]:
    if not boxes_text or not boxes_text.strip():
        return None, None
    boxes_per_class = [[] for _ in range(num_classes)]
    labels_per_class = [[] for _ in range(num_classes)]
    for line in boxes_text.strip().split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            part1, part2 = line.split(":", 1)
            class_idx = int(part1.strip())
            coords_str, label_str = part2.strip().rsplit(":", 1)
            x1, y1, x2, y2 = (float(x.strip()) for x in coords_str.split(","))
            label = int(label_str.strip())
            if label not in (0, 1) or not (0 <= class_idx < num_classes):
                continue
            boxes_per_class[class_idx].append([x1, y1, x2, y2])
            labels_per_class[class_idx].append(label)
        except (ValueError, AttributeError):
            continue
    if all(not b for b in boxes_per_class):
        return None, None
    return (
        [b if b else None for b in boxes_per_class],
        [l if l else None for l in labels_per_class],
    )


def load_image(image_path: str | None, uploaded_file) -> np.ndarray | None:
    if uploaded_file is not None:
        import cv2
        arr = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr[:, :, ::-1] if bgr is not None else None
    if image_path and os.path.exists(image_path):
        import cv2
        bgr = cv2.imread(image_path)
        return bgr[:, :, ::-1] if bgr is not None else None
    return None


def exemplar_box_drawer(image_rgb, w, h, key_prefix):
    """Click-and-drag box drawing on image. Returns (bboxes, labels) or (None, None)."""
    from streamlit_image_coordinates import streamlit_image_coordinates
    import cv2 as _cv2

    boxes_key = f"{key_prefix}_drawn_boxes"
    coords_key = f"{key_prefix}_last_coords"
    if boxes_key not in st.session_state:
        st.session_state[boxes_key] = []
    if coords_key not in st.session_state:
        st.session_state[coords_key] = None

    display_w = min(w, 450)
    display_h = int(h * display_w / w)
    scale_x, scale_y = w / display_w, h / display_h

    display_img = _cv2.resize(image_rgb, (display_w, display_h))
    for i, box in enumerate(st.session_state[boxes_key]):
        x1d, y1d = int(box[0] / scale_x), int(box[1] / scale_y)
        x2d, y2d = int(box[2] / scale_x), int(box[3] / scale_y)
        _cv2.rectangle(display_img, (x1d, y1d), (x2d, y2d), (0, 255, 0), 2)
        _cv2.putText(display_img, f"#{i+1}", (x1d + 4, y1d + 16),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, _cv2.LINE_AA)

    coords = streamlit_image_coordinates(
        display_img, width=display_w, height=display_h,
        click_and_drag=True, cursor="crosshair", key=f"{key_prefix}_img_coords",
    )

    if coords and "x1" in coords and "x2" in coords:
        ct = (coords["x1"], coords["y1"], coords["x2"], coords["y2"])
        if ct != st.session_state[coords_key]:
            st.session_state[coords_key] = ct
            x1 = min(ct[0], ct[2]) * scale_x
            y1 = min(ct[1], ct[3]) * scale_y
            x2 = max(ct[0], ct[2]) * scale_x
            y2 = max(ct[1], ct[3]) * scale_y
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                st.session_state[boxes_key].append([x1, y1, x2, y2])
                st.rerun()

    drawn = st.session_state[boxes_key]
    if drawn:
        col_info, col_clear = st.columns([3, 1])
        with col_info:
            st.caption(f"{len(drawn)} box(es)")
        with col_clear:
            if st.button("Clear", key=f"{key_prefix}_clear"):
                st.session_state[boxes_key] = []
                st.session_state["processed"] = False
                st.session_state["exemplar_result"] = None
                st.rerun()
        return list(drawn), [1] * len(drawn)
    return None, None


# ─────────────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="SAM3 Segmentation", layout="wide")
    st.title("SAM3 Semantic Segmentation")

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.header("Input")
    image_option = st.sidebar.radio("Image source:", ["Custom Path", "Upload"])

    image_path, uploaded = None, None
    if image_option == "Custom Path":
        image_path = st.sidebar.text_input(
            "Image path",
            value="data/custom_datasets/example/images/000000.png",
        )
    else:
        uploaded = st.sidebar.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

    prompt_mode = st.sidebar.radio(
        "Prompt mode", ["Text prompts", "Image exemplars"],
        help="**Text prompts**: text per class.  **Image exemplars**: draw boxes on same image.",
    )

    class_labels_text, boxes_text = "", ""
    if prompt_mode == "Text prompts":
        class_labels_text = st.sidebar.text_area(
            "Class labels", value="black floor\nwhite floor", height=100,
        )
        with st.sidebar.expander("Bounding boxes (optional)", expanded=False):
            boxes_text = st.text_area(
                "Boxes", value="", height=60,
                placeholder="0: 100,200,300,400:1", key="text_boxes",
            )

    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    overlay_alpha = st.sidebar.slider("Overlay alpha", 0.1, 1.0, 0.4, 0.05)
    run_btn = st.sidebar.button("Run segmentation", type="primary")

    # ── Load image ───────────────────────────────────────────────────────────
    image_rgb = load_image(image_path if image_option == "Custom Path" else None, uploaded)
    if image_rgb is None:
        st.info("Provide an image path or upload an image to begin.")
        st.stop()

    from PIL import Image
    pil_image = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    # Clear stale results on image change
    img_key = f"{image_path}_{w}x{h}"
    if st.session_state.get("_img_key") != img_key:
        st.session_state["_img_key"] = img_key
        st.session_state["processed"] = False
        st.session_state["results"] = {}
        st.session_state["exemplar_result"] = None

    # ═════════════════════════════════════════════════════════════════════════
    # TEXT PROMPTS
    # ═════════════════════════════════════════════════════════════════════════
    if prompt_mode == "Text prompts":
        class_labels = parse_class_labels(class_labels_text)
        box_prompts, box_labels = (
            parse_boxes_per_prompt(boxes_text, len(class_labels))
            if class_labels else (None, None)
        )
        if not class_labels:
            col_o, col_p = st.columns(2)
            with col_o:
                st.image(image_rgb, caption=f"{w}x{h}", width="stretch")
            with col_p:
                st.warning("Enter at least one class label.")
            st.stop()

        if run_btn:
            with st.spinner("Loading SAM3..."):
                predictor = init_sam3_model()
            with st.spinner(f"Segmenting {len(class_labels)} classes..."):
                results = process_multiple_prompts(
                    predictor, pil_image, class_labels,
                    input_boxes_per_prompt=box_prompts,
                    input_boxes_labels_per_prompt=box_labels,
                )
            st.session_state["results"] = results
            st.session_state["processed"] = True

        if st.session_state.get("processed") and st.session_state.get("results"):
            results = st.session_state["results"]
            masks = {p: get_semantic_segmentation_tensor(r["semantic_seg"], h, w, threshold=confidence_threshold)
                     for p, r in results.items()}
            combined = create_combined_overlay(pil_image, masks, alpha=overlay_alpha)

            # Row 1: original + combined
            c1, c2 = st.columns(2)
            with c1:
                st.image(image_rgb, caption=f"Original ({w}x{h})", width="stretch")
            with c2:
                st.image(combined, caption="Combined", width="stretch")

            # Row 2: per-class
            cols = st.columns(min(len(masks), 3))
            for i, (p, m) in enumerate(masks.items()):
                with cols[i % 3]:
                    st.image(overlay_mask_on_image_numpy(pil_image, m, alpha=overlay_alpha),
                             caption=p, width="stretch")

            # Heatmaps collapsed
            with st.expander("Heatmaps", expanded=False):
                hcols = st.columns(min(len(results), 3))
                for i, (p, r) in enumerate(results.items()):
                    raw = get_semantic_segmentation_tensor(r["semantic_seg"], h, w, threshold=None)
                    with hcols[i % 3]:
                        fig = px.imshow(torch.sigmoid(raw).cpu().numpy(),
                                        color_continuous_scale="turbo", origin="upper",
                                        aspect="equal", zmin=0, zmax=1)
                        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=220,
                                          title=dict(text=p, font=dict(size=10)))
                        st.plotly_chart(fig, width="stretch")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(image_rgb, caption=f"{w}x{h}", width="stretch")
            with c2:
                st.info("Click **Run segmentation**.")

    # ═════════════════════════════════════════════════════════════════════════
    # IMAGE EXEMPLARS (same-image)
    # ═════════════════════════════════════════════════════════════════════════
    elif prompt_mode == "Image exemplars":
        col_draw, col_result = st.columns(2, gap="medium")

        with col_draw:
            st.caption("Click and drag to draw exemplar boxes:")
            ex_bboxes, ex_labels = exemplar_box_drawer(image_rgb, w, h, "ex")

        if run_btn and ex_bboxes:
            with st.spinner("Loading SAM3..."):
                predictor = init_sam3_model()
            with st.spinner("Running exemplar segmentation..."):
                result = process_with_exemplars(
                    predictor, pil_image, bboxes=ex_bboxes, labels=ex_labels,
                )
            st.session_state["exemplar_result"] = result
            st.session_state["processed"] = True

        with col_result:
            if st.session_state.get("processed") and st.session_state.get("exemplar_result"):
                result = st.session_state["exemplar_result"]
                mask = get_semantic_segmentation_tensor(
                    result["semantic_seg"], h, w, threshold=confidence_threshold,
                )
                overlay = overlay_mask_on_image_numpy(
                    pil_image, mask, color=(0, 255, 0), alpha=overlay_alpha,
                )
                st.caption("Result")
                st.image(overlay, width="stretch")
            else:
                st.caption("Result")
                st.image(image_rgb, caption=f"{w}x{h}", width="stretch")

        if st.session_state.get("processed") and st.session_state.get("exemplar_result"):
            with st.expander("Heatmap", expanded=False):
                raw = get_semantic_segmentation_tensor(
                    st.session_state["exemplar_result"]["semantic_seg"], h, w, threshold=None)
                fig = px.imshow(torch.sigmoid(raw).cpu().numpy(),
                                color_continuous_scale="turbo", origin="upper",
                                aspect="equal", zmin=0, zmax=1)
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=220)
                st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
