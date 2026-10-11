# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""Streamlit app for interactive SAM3 dataset labeling.

Browse images from a split file, run SAM3 with per-image editable text prompts
and/or per-class image exemplars, and save accepted masks.

Run:
    streamlit run scripts/sem_map/notebooks/sam3_dataset_labeling_app.py
"""

import html
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
import streamlit as st
from PIL import Image

# ── Workspace / path setup ───────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def init_sam3_model():
    return load_sam3_model()


def load_seg_layer(yaml_path: str) -> list[dict]:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return sorted(cfg.get("categories", []), key=lambda c: c["id"])


def load_present_classes(dataset_root: Path, image_stem: str):
    """Load LLM class-presence sidecar for one image, if it exists.

    Returns:
        (present_ids, reasons, model_name) where:
            present_ids: set[int] of class ids the LLM marked visible
            reasons:     dict[int, str] mapping class id → per-class reason
                         (empty dict for v1 sidecars without reasons)
            model_name:  string identifier of the model that produced the sidecar
        Or (None, None, None) if no sidecar exists / parse fails. Sidecars
        are written by ``step_llm_class_filter`` to
        ``<dataset_root>/llm_present_classes/<stem>.json``.
    """
    import json
    sidecar = dataset_root / "llm_present_classes" / f"{image_stem}.json"
    if not sidecar.exists():
        return None, None, None
    try:
        data = json.loads(sidecar.read_text())
        ids = set(int(x) for x in data.get("present_class_ids", []))
        reasons: dict[int, str] = {}
        for d in data.get("class_decisions", []) or []:
            try:
                reasons[int(d["id"])] = str(d.get("reason", "")).strip()
            except (KeyError, ValueError, TypeError):
                continue
        return ids, reasons, data.get("model", "")
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, None, None


def load_split_images(dataset_root: Path, split_filename: str) -> list[Path]:
    split_path = dataset_root / split_filename
    if not split_path.exists():
        split_path = Path(split_filename)
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    images_dir = dataset_root / "images"
    paths = []
    for line in split_path.read_text().splitlines():
        stem = line.strip()
        if not stem:
            continue
        for ext in (".png", ".jpg", ".jpeg"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                paths.append(p)
                break
    return paths


def combine_masks_to_segmentation(masks_dict, prompts_to_class_ids, class_id_priority,
                                   background_class_id, image_size, threshold=0.5):
    """Combine SAM3 masks into a single uint8 segmentation mask.

    Multiple prompts/exemplars for the same class are unioned first,
    then classes are painted in priority order (highest priority last = wins).
    """
    h, w = image_size[1], image_size[0]
    segmentation = np.full((h, w), background_class_id, dtype=np.uint8)

    # Step 1: Merge all masks per class (union)
    per_class_masks = {}  # {class_id: np.ndarray bool}
    for prompt, result in masks_dict.items():
        class_id = prompts_to_class_ids.get(prompt)
        if class_id is None:
            continue
        mask_tensor = get_semantic_segmentation_tensor(
            result["semantic_seg"], h, w, threshold=threshold)
        mask_binary = (mask_tensor > 0.5).cpu().numpy()
        if class_id in per_class_masks:
            per_class_masks[class_id] = per_class_masks[class_id] | mask_binary
        else:
            per_class_masks[class_id] = mask_binary

    # Step 2: Paint in priority order (lowest priority first, highest last)
    priority_map = {cid: idx for idx, cid in enumerate(class_id_priority)}
    classes_sorted = sorted(
        per_class_masks.keys(),
        key=lambda cid: priority_map.get(cid, float("inf")),
        reverse=True,
    )
    for class_id in classes_sorted:
        segmentation[per_class_masks[class_id]] = class_id

    return segmentation


def colorize_mask(mask_np, categories, image_rgb, alpha=0.5):
    overlay = image_rgb.astype(np.float32).copy()
    for cat in categories:
        color = np.array(cat.get("color", [128, 128, 128]), dtype=np.float32)
        region = mask_np == cat["id"]
        overlay[region] = overlay[region] * (1 - alpha) + color * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


def color_swatch_html(color):
    r, g, b = color
    return (
        f'<span style="display:inline-block;width:14px;height:14px;border-radius:3px;'
        f'background:rgb({r},{g},{b});vertical-align:middle;margin-right:4px;'
        f'border:1px solid #888"></span>'
    )


def exemplar_box_drawer(image_rgb, w, h, key_prefix, selected_class_idx=None,
                         non_bg_categories=None, is_positive=True):
    """Click-and-drag box drawing on image. Boxes tagged with class index and pos/neg label.

    Storage: {class_idx: [{"box": [x1,y1,x2,y2], "label": 1|0}, ...]}
    Returns (boxes_per_class, labels_per_class):
        boxes_per_class: {class_idx: [[x1,y1,x2,y2], ...]}
        labels_per_class: {class_idx: [1|0, ...]}
    """
    from streamlit_image_coordinates import streamlit_image_coordinates
    import cv2 as _cv2

    boxes_key = f"{key_prefix}_drawn_boxes"
    coords_key = f"{key_prefix}_last_coords"
    if boxes_key not in st.session_state:
        st.session_state[boxes_key] = {}
    if coords_key not in st.session_state:
        st.session_state[coords_key] = None

    display_w = min(w, 450)
    display_h = int(h * display_w / w)
    scale_x, scale_y = w / display_w, h / display_h

    # Draw existing boxes: positive=class color solid, negative=class color dashed + "NEG"
    display_img = _cv2.resize(image_rgb, (display_w, display_h))
    all_entries = st.session_state[boxes_key]
    for cls_idx, entries in all_entries.items():
        if non_bg_categories and 0 <= cls_idx < len(non_bg_categories):
            cls_color = tuple(non_bg_categories[cls_idx].get("color", [0, 255, 0]))
            cls_name = non_bg_categories[cls_idx]["name"]
        else:
            cls_color = (0, 255, 0)
            cls_name = str(cls_idx)
        for entry in entries:
            box = entry["box"]
            label = entry["label"]
            x1d, y1d = int(box[0] / scale_x), int(box[1] / scale_y)
            x2d, y2d = int(box[2] / scale_x), int(box[3] / scale_y)
            if label == 1:
                _cv2.rectangle(display_img, (x1d, y1d), (x2d, y2d), cls_color, 2)
                tag = f"+{cls_name}"
            else:
                for pt1, pt2 in [((x1d, y1d), (x2d, y1d)), ((x2d, y1d), (x2d, y2d)),
                                  ((x2d, y2d), (x1d, y2d)), ((x1d, y2d), (x1d, y1d))]:
                    _draw_dashed_line(display_img, pt1, pt2, cls_color, 2, 8)
                tag = f"-{cls_name}"
            # Label tag with background
            (tw, th), _ = _cv2.getTextSize(tag, _cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            _cv2.rectangle(display_img, (x1d, y1d - th - 4), (x1d + tw + 4, y1d), cls_color, -1)
            _cv2.putText(display_img, tag, (x1d + 2, y1d - 2),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, _cv2.LINE_AA)

    coords = streamlit_image_coordinates(
        display_img, width=display_w, height=display_h,
        click_and_drag=True, cursor="crosshair", key=f"{key_prefix}_img_coords",
    )

    if coords and "x1" in coords and "x2" in coords and selected_class_idx is not None:
        ct = (coords["x1"], coords["y1"], coords["x2"], coords["y2"])
        if ct != st.session_state[coords_key]:
            st.session_state[coords_key] = ct
            x1 = min(ct[0], ct[2]) * scale_x
            y1 = min(ct[1], ct[3]) * scale_y
            x2 = max(ct[0], ct[2]) * scale_x
            y2 = max(ct[1], ct[3]) * scale_y
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                if selected_class_idx not in st.session_state[boxes_key]:
                    st.session_state[boxes_key][selected_class_idx] = []
                st.session_state[boxes_key][selected_class_idx].append({
                    "box": [x1, y1, x2, y2],
                    "label": 1 if is_positive else 0,
                })
                st.rerun()

    drawn = st.session_state[boxes_key]
    total = sum(len(v) for v in drawn.values())
    if total > 0:
        n_pos = sum(1 for entries in drawn.values() for e in entries if e["label"] == 1)
        n_neg = total - n_pos
        info_parts = []
        if n_pos:
            info_parts.append(f"{n_pos} pos")
        if n_neg:
            info_parts.append(f"{n_neg} neg")
        col_info, col_clear = st.columns([3, 1])
        with col_info:
            st.caption(f"{', '.join(info_parts)} box(es)")
        with col_clear:
            if st.button("Clear", key=f"{key_prefix}_clear"):
                st.session_state[boxes_key] = {}
                st.session_state["ds_processed"] = False
                st.session_state["ds_results"] = None
                st.session_state["ds_last_mask"] = None
                st.rerun()

    # Extract into separate dicts for downstream use
    boxes_per_class = {}
    labels_per_class = {}
    for cls_idx, entries in drawn.items():
        if entries:
            boxes_per_class[cls_idx] = [e["box"] for e in entries]
            labels_per_class[cls_idx] = [e["label"] for e in entries]

    return boxes_per_class, labels_per_class


def _draw_dashed_line(img, pt1, pt2, color, thickness, dash_len):
    """Draw a dashed line on an image."""
    import cv2 as _cv2
    import math
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length == 0:
        return
    dx, dy = dx / length, dy / length
    pos = 0
    draw = True
    while pos < length:
        seg_end = min(pos + dash_len, length)
        if draw:
            p1 = (int(pt1[0] + dx * pos), int(pt1[1] + dy * pos))
            p2 = (int(pt1[0] + dx * seg_end), int(pt1[1] + dy * seg_end))
            _cv2.line(img, p1, p2, color, thickness)
        pos = seg_end
        draw = not draw


# ─────────────────────────────────────────────────────────────────────────────


def main():
    st.set_page_config(page_title="SAM3 Dataset Labeling", layout="wide")
    st.title("SAM3 Dataset Labeling")

    # ── Sidebar: dataset config (collapsible) ──────────────────────────────
    with st.sidebar.expander("Dataset", expanded="ds_images" not in st.session_state):
        default_root = str(repo_root / "data/custom_datasets/my_scene/001")
        dataset_root_str = st.text_input("Root", value=default_root, key="sb_root")
        split_file = st.text_input("Split", value="train_images.txt", key="sb_split")
        default_yaml = str(repo_root / "config/seg_layers/coarse_3.yaml")
        yaml_path = st.text_input("YAML", value=default_yaml, key="sb_yaml")
        if st.button("Load", type="primary", width="stretch"):
            try:
                dataset_root = Path(dataset_root_str).resolve()
                imgs = load_split_images(dataset_root, split_file)
                cats = load_seg_layer(yaml_path)
                if not imgs:
                    st.error("No images found.")
                else:
                    st.session_state["ds_root"] = dataset_root
                    st.session_state["ds_images"] = imgs
                    st.session_state["ds_categories"] = cats
                    st.session_state["ds_current_idx"] = 0
                    st.session_state["ds_processed"] = False
                    st.session_state["ds_results"] = None
                    st.session_state["ds_last_mask"] = None
                    # Clear per-YAML state so new categories take effect
                    st.session_state["ds_prompts"] = {}
                    st.session_state["ds_ex_drawn_boxes"] = {}
                    st.session_state["ds_ex_last_coords"] = None
                    # Drop cached per-class prompt text_input widget values
                    for k in list(st.session_state.keys()):
                        if k.startswith("prompt_input_"):
                            del st.session_state[k]
                    st.success(f"{len(imgs)} imgs, {len(cats)} cls")
            except Exception as e:
                st.error(str(e))

    if "ds_images" not in st.session_state:
        st.info("Configure dataset in sidebar and click **Load**.")
        st.stop()

    images = st.session_state["ds_images"]
    categories = st.session_state["ds_categories"]
    dataset_root = st.session_state["ds_root"]
    n_images = len(images)
    non_bg_categories = [c for c in categories if c["id"] != 0]

    # ── Sidebar: navigation (compact) ────────────────────────────────────────
    def _nav_change():
        st.session_state["ds_processed"] = False
        st.session_state["ds_results"] = None
        st.session_state["ds_last_mask"] = None

    idx = st.session_state.get("ds_current_idx", 0)
    c1, c2, c3 = st.sidebar.columns([1, 2, 1])
    with c1:
        if st.button("◀", width="stretch") and idx > 0:
            st.session_state["ds_current_idx"] = idx - 1
            _nav_change(); st.rerun()
    with c2:
        st.markdown(f"<div style='text-align:center;padding-top:6px'><b>{idx+1}</b>/{n_images}</div>",
                     unsafe_allow_html=True)
    with c3:
        if st.button("▶", width="stretch") and idx < n_images - 1:
            st.session_state["ds_current_idx"] = idx + 1
            _nav_change(); st.rerun()

    jump = st.sidebar.number_input("Go to", 1, n_images, idx + 1, 1, label_visibility="collapsed")
    if int(jump) - 1 != idx:
        st.session_state["ds_current_idx"] = int(jump) - 1
        _nav_change(); st.rerun()

    # ── Sidebar: parameters (compact) ────────────────────────────────────────
    confidence_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.5, 0.05)
    overlay_alpha = st.sidebar.slider("Alpha", 0.1, 1.0, 0.4, 0.05)
    skip_llm_absent = st.sidebar.checkbox(
        "Skip LLM-rejected classes on Run",
        value=True,
        help="When the LLM class-presence sidecar exists for an image, skip "
             "classes the LLM marked absent (❌) instead of running SAM3 on "
             "them. Has no effect on images without a sidecar.",
    )

    # ── Load image ───────────────────────────────────────────────────────────
    idx = st.session_state["ds_current_idx"]
    image_path = images[idx]
    image_stem = image_path.stem
    seg_out_path = dataset_root / "segmentations" / f"{image_stem}.png"

    pil_image = Image.open(image_path).convert("RGB")
    image_rgb = np.array(pil_image)
    h, w = image_rgb.shape[:2]
    already_saved = seg_out_path.exists()

    # LLM class-presence sidecar (None if step_llm_class_filter never ran for this image)
    llm_present_ids, llm_reasons, llm_model_name = load_present_classes(dataset_root, image_stem)

    # ── Three-column layout: prompts | canvas | results ──────────────────────
    col_prompts, col_canvas, col_results = st.columns([2, 3, 3], gap="small")

    # ── Left: text prompts + class selector for exemplars ────────────────────
    with col_prompts:
        saved_badge = " ✅" if already_saved else ""
        st.markdown(
            f"**{image_stem}** ({idx+1}/{n_images}){saved_badge}",
        )

        # Text prompts per class
        st.markdown("**Text prompts**")

        # LLM filter summary line
        if llm_present_ids is not None:
            n_present = sum(1 for c in non_bg_categories if c["id"] in llm_present_ids)
            n_total = len(non_bg_categories)
            model_tag = f" · `{llm_model_name}`" if llm_model_name else ""
            st.caption(f"🤖 LLM filter: **{n_present}/{n_total}** classes present{model_tag}")
            if llm_reasons:
                with st.expander("LLM reasoning per class", expanded=False):
                    for cat in non_bg_categories:
                        cid = cat["id"]
                        reason = llm_reasons.get(cid, "—")
                        mark = "✅" if cid in llm_present_ids else "❌"
                        st.markdown(f"{mark} **{cat['name']}** — {reason}")
        else:
            st.caption("🤖 LLM filter: no sidecar — run `step_llm_class_filter` to populate")

        current_prompts = st.session_state.get("ds_prompts", {})
        updated_prompts = {}
        for cat in non_bg_categories:
            cid = cat["id"]
            yaml_default = cat.get("concept", "").strip() or cat["name"]
            default_prompt = current_prompts.get(cid, yaml_default)
            color = cat.get("color", [128, 128, 128])
            swatch = color_swatch_html(color)

            # LLM presence indicator: ✅ present, ❌ absent, faded · if no sidecar.
            # Reason (if available) is exposed as the HTML title for hover tooltip.
            reason_text = (llm_reasons or {}).get(cid, "") if llm_reasons is not None else ""
            if llm_present_ids is None:
                tooltip = "no LLM sidecar"
                mark = "·"
                mark_style = "color:#888;"
                name_style = ""
            elif cid in llm_present_ids:
                tooltip = reason_text or "LLM: present"
                mark = "✅"
                mark_style = ""
                name_style = ""
            else:
                tooltip = reason_text or "LLM: absent"
                mark = "❌"
                mark_style = ""
                name_style = "opacity:0.45;text-decoration:line-through;"
            tooltip_attr = html.escape(tooltip, quote=True)
            presence_html = (
                f'<span style="{mark_style}margin-left:6px" title="{tooltip_attr}">{mark}</span>'
            )

            st.markdown(
                f'<div style="{name_style}">{swatch} <b>{cat["name"]}</b>{presence_html}</div>',
                unsafe_allow_html=True,
            )
            prompt_val = st.text_input(
                f"prompt_{cid}", value=default_prompt,
                label_visibility="collapsed", key=f"prompt_input_{cid}",
            )
            updated_prompts[cid] = prompt_val
        st.session_state["ds_prompts"] = updated_prompts

        prompts_to_class_ids = {p: cid for cid, p in updated_prompts.items() if p.strip()}

        # Exemplar class selector + pos/neg toggle
        st.divider()
        st.markdown("**Exemplar class**")
        class_options = [f"{cat['name']}" for cat in non_bg_categories]
        selected_class_name = st.radio(
            "Draw boxes for:",
            class_options,
            label_visibility="collapsed",
            key="ds_exemplar_class_select",
        )
        selected_class_idx = class_options.index(selected_class_name)

        box_mode = st.radio(
            "Box type:",
            ["Positive (include)", "Negative (exclude)"],
            horizontal=True,
            key="ds_box_mode",
        )
        is_positive = box_mode.startswith("Positive")

        # Buttons. The exemplar boxes are drawn further down (in col_canvas), so
        # read them from the drawer's session state rather than from its return
        # value, which does not exist yet on this pass.
        drawn_boxes = st.session_state.get("ds_ex_drawn_boxes") or {}
        can_run = bool(prompts_to_class_ids) or any(drawn_boxes.values())
        b_run, b_save, b_clear = st.columns(3)
        with b_run:
            run_btn = st.button("Run", type="primary", disabled=not can_run, width="stretch")
        with b_save:
            can_save = (st.session_state.get("ds_processed")
                        and st.session_state.get("ds_last_mask") is not None)
            save_btn = st.button("Save", type="secondary", disabled=not can_save, width="stretch")
        with b_clear:
            clear_btn = st.button("Clear", width="stretch")

    # ── Middle: exemplar canvas ──────────────────────────────────────────────
    with col_canvas:
        sel_cat = non_bg_categories[selected_class_idx]
        mode_label = "+" if is_positive else "-"
        st.caption(f"Draw **{mode_label}** boxes for **{sel_cat['name']}**:")
        exemplar_per_class, exemplar_labels_per_class = exemplar_box_drawer(
            image_rgb, w, h, "ds_ex",
            selected_class_idx=selected_class_idx,
            non_bg_categories=non_bg_categories,
            is_positive=is_positive,
        )

    # ── Clear results ────────────────────────────────────────────────────────
    if clear_btn:
        st.session_state["ds_processed"] = False
        st.session_state["ds_results"] = None
        st.session_state["ds_last_mask"] = None
        st.session_state["ds_ex_drawn_boxes"] = {}
        st.session_state["ds_ex_last_coords"] = None
        st.rerun()

    # ── Run segmentation ─────────────────────────────────────────────────────
    if run_btn:
        with st.spinner("Loading SAM3..."):
            predictor = init_sam3_model()

        # Build unified per-class inputs: text + exemplar boxes combined
        class_results = {}  # {class_id: result}
        result_class_ids = {}  # {key: class_id} for combine_masks

        # Decide which classes to actually run. If the LLM filter sidecar exists
        # and the user enabled "Skip LLM-rejected classes", drop classes the LLM
        # marked absent. Exemplar boxes always override (user-drawn signal wins).
        llm_skipped_names: list[str] = []
        if skip_llm_absent and llm_present_ids is not None:
            llm_skipped_names = [
                cat["name"] for cat in non_bg_categories
                if cat["id"] not in llm_present_ids
                and not exemplar_per_class.get(non_bg_categories.index(cat))
            ]
        if llm_skipped_names:
            st.caption(
                f"🤖 Skipping {len(llm_skipped_names)} LLM-rejected "
                f"class(es): {', '.join(llm_skipped_names)}"
            )

        with st.spinner(f"Segmenting {len(non_bg_categories)} class(es)..."):
            for cls_idx, cat in enumerate(non_bg_categories):
                cid = cat["id"]
                text_prompt = updated_prompts.get(cid, "").strip() or None
                boxes = exemplar_per_class.get(cls_idx)

                # Need at least one positive signal (text or positive box) to detect
                box_labels = exemplar_labels_per_class.get(cls_idx, [1] * len(boxes)) if boxes else []
                has_positive_box = any(l == 1 for l in box_labels)

                if not text_prompt and not has_positive_box:
                    continue

                # Honor the LLM filter: if the sidecar marks this class absent
                # and the user opted in to skipping, drop it — UNLESS the user
                # has drawn a positive exemplar box, which always wins.
                if (skip_llm_absent
                        and llm_present_ids is not None
                        and cid not in llm_present_ids
                        and not has_positive_box):
                    continue

                # Separate positive and negative boxes
                pos_boxes = [b for b, l in zip(boxes, box_labels) if l == 1] if boxes else []
                neg_boxes = [b for b, l in zip(boxes, box_labels) if l == 0] if boxes else []

                # Run with text + positive boxes only
                if pos_boxes:
                    result = process_with_exemplars(
                        predictor, pil_image,
                        bboxes=pos_boxes,
                        labels=[1] * len(pos_boxes),
                        text=[text_prompt] if text_prompt else None,
                    )
                else:
                    results_dict = process_multiple_prompts(
                        predictor, pil_image, [text_prompt],
                    )
                    result = results_dict[text_prompt]

                # Post-process: zero out semantic_seg inside negative box regions
                if neg_boxes:
                    seg = result["semantic_seg"]
                    for nb in neg_boxes:
                        x1, y1, x2, y2 = nb
                        # Scale box coords to semantic_seg resolution
                        seg_h, seg_w = seg.shape[-2], seg.shape[-1]
                        sx, sy = seg_w / w, seg_h / h
                        bx1 = int(x1 * sx)
                        by1 = int(y1 * sy)
                        bx2 = int(x2 * sx)
                        by2 = int(y2 * sy)
                        seg[..., by1:by2, bx1:bx2] = -32.0  # large negative = no detection
                    result["semantic_seg"] = seg

                key = f"class_{cid}"
                class_results[key] = result
                result_class_ids[key] = cid

        st.session_state["ds_results"] = class_results
        st.session_state["ds_prompts_to_class_ids"] = result_class_ids
        st.session_state["ds_processed"] = True
        st.session_state["ds_last_mask"] = None

    # ── Right: results ───────────────────────────────────────────────────────
    with col_results:
        if st.session_state.get("ds_processed") and st.session_state.get("ds_results"):
            results = st.session_state["ds_results"]
            p2c = st.session_state.get("ds_prompts_to_class_ids", {})
            cat_color_by_id = {c["id"]: tuple(c.get("color", [255, 0, 0])) for c in categories}

            # One result per class — build overlay
            masks_for_overlay = {}
            overlay_colors = []
            for key, result in results.items():
                cid = p2c.get(key, -1)
                masks_for_overlay[key] = get_semantic_segmentation_tensor(
                    result["semantic_seg"], h, w, threshold=confidence_threshold)
                overlay_colors.append(cat_color_by_id.get(cid, (255, 0, 0)))

            combined = create_combined_overlay(
                pil_image, masks_for_overlay, colors=overlay_colors, alpha=overlay_alpha)
            st.markdown("**Combined**")
            st.image(combined, width="stretch")

            # Per-class overlays
            n = len(results)
            if n > 0:
                cols = st.columns(min(n, 3))
                for i, (key, result) in enumerate(results.items()):
                    cid = p2c.get(key, -1)
                    color = cat_color_by_id.get(cid, (255, 0, 0))
                    mask_t = get_semantic_segmentation_tensor(
                        result["semantic_seg"], h, w, threshold=confidence_threshold)
                    overlay = overlay_mask_on_image_numpy(
                        pil_image, mask_t, color=color, alpha=overlay_alpha)
                    name = next((c["name"] for c in categories if c["id"] == cid), key)
                    with cols[i % 3]:
                        st.image(overlay, caption=name, width="stretch")

            # Final mask with priority painting (higher class ID wins)
            bg_id = next((c["id"] for c in categories if c["id"] == 0), 0)
            class_id_priority = sorted(p2c.values(), reverse=True)
            final_mask = combine_masks_to_segmentation(
                results, p2c, class_id_priority, bg_id, pil_image.size,
                threshold=confidence_threshold,
            )
            st.session_state["ds_last_mask"] = final_mask

        elif already_saved:
            saved_mask_np = np.array(Image.open(seg_out_path))
            colored = colorize_mask(saved_mask_np, categories, image_rgb, alpha=overlay_alpha)
            st.markdown("**Saved mask**")
            st.image(colored, width="stretch")
            st.session_state["ds_last_mask"] = saved_mask_np
        else:
            st.info("Configure prompts and/or draw exemplar boxes, then click **Run**.")

    if run_btn:
        st.rerun()

    # ── Handle save ──────────────────────────────────────────────────────────
    if save_btn and st.session_state.get("ds_last_mask") is not None:
        final_mask = st.session_state["ds_last_mask"]
        seg_out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(final_mask, mode="L").save(seg_out_path)
        st.toast(f"Saved {image_stem}", icon="✅")
        if idx + 1 < n_images:
            st.session_state["ds_current_idx"] = idx + 1
            st.session_state["ds_processed"] = False
            st.session_state["ds_results"] = None
            st.session_state["ds_last_mask"] = None
            st.rerun()


if __name__ == "__main__":
    main()
