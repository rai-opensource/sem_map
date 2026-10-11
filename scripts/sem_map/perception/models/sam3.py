# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""SAM3 model utilities for promptable concept segmentation.

This module provides utilities for loading and using SAM3 models for semantic
and instance segmentation with text prompts and image exemplars (box prompts).

Uses HuggingFace transformers (Sam3Model / Sam3Processor) as the inference backend.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

try:
    from transformers import Sam3Processor, Sam3Model
except ImportError:
    Sam3Processor = None
    Sam3Model = None

try:
    from transformers import Sam3VideoProcessor, Sam3VideoModel
except ImportError:
    Sam3VideoProcessor = None
    Sam3VideoModel = None


class SAM3Predictor:
    """Thin wrapper around HF Sam3Model + Sam3Processor for a unified API."""

    def __init__(self, model: "Sam3Model", processor: "Sam3Processor", device: str):
        self.model = model
        self.processor = processor
        self.device = device


def load_sam3_model(
    model_name: str = "facebook/sam3",
    device: Optional[str] = None,
) -> SAM3Predictor:
    """Load SAM3 model and processor from HuggingFace.

    Args:
        model_name: HuggingFace model identifier. Defaults to "facebook/sam3".
        device: Device to load model on ("cuda", "cpu", or None for auto-detect).

    Returns:
        SAM3Predictor wrapping the model and processor.
    """
    if Sam3Model is None or Sam3Processor is None:
        raise ImportError(
            "transformers library is required. Install with: pip install transformers"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Sam3Model.from_pretrained(model_name).to(device)
    processor = Sam3Processor.from_pretrained(model_name)

    return SAM3Predictor(model, processor, device)


def process_single_prompt(
    predictor: SAM3Predictor,
    image: Image.Image,
    text_prompt: Optional[str] = None,
    input_boxes: Optional[List[List[float]]] = None,
    input_boxes_labels: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Process a single prompt on an image (text, boxes, or both).

    Args:
        predictor: SAM3Predictor instance.
        image: PIL Image to process.
        text_prompt: Optional text prompt describing the concept.
        input_boxes: Optional list of boxes in xyxy format [x1, y1, x2, y2].
        input_boxes_labels: Optional labels per box (0=negative, 1=positive).

    Returns:
        Dictionary with 'instance_masks' and 'semantic_seg' tensors.
    """
    model, processor, device = predictor.model, predictor.processor, predictor.device

    # Build processor kwargs
    proc_kwargs = {"images": image, "return_tensors": "pt"}
    if text_prompt is not None:
        proc_kwargs["text"] = text_prompt
    if input_boxes is not None and input_boxes_labels is not None:
        if len(input_boxes) != len(input_boxes_labels):
            raise ValueError(
                f"input_boxes ({len(input_boxes)}) and input_boxes_labels "
                f"({len(input_boxes_labels)}) must have same length"
            )
        proc_kwargs["input_boxes"] = [input_boxes]
        proc_kwargs["input_boxes_labels"] = [input_boxes_labels]

    inputs = processor(**proc_kwargs).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    return {
        "instance_masks": torch.sigmoid(outputs.pred_masks),
        "semantic_seg": outputs.semantic_seg,
    }


def process_multiple_prompts(
    predictor: SAM3Predictor,
    image: Image.Image,
    text_prompts: List[str],
    input_boxes_per_prompt: Optional[List[Optional[List[List[float]]]]] = None,
    input_boxes_labels_per_prompt: Optional[List[Optional[List[int]]]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Process multiple text prompts on an image.

    Args:
        predictor: SAM3Predictor instance.
        image: PIL Image to process.
        text_prompts: List of text prompts.
        input_boxes_per_prompt: Optional boxes per prompt.
        input_boxes_labels_per_prompt: Optional labels per prompt's boxes.

    Returns:
        Dictionary keyed by prompt text, each with 'instance_masks' and 'semantic_seg'.
    """
    if input_boxes_per_prompt is not None and len(input_boxes_per_prompt) != len(text_prompts):
        raise ValueError("input_boxes_per_prompt length must match text_prompts")
    if input_boxes_labels_per_prompt is not None and len(input_boxes_labels_per_prompt) != len(text_prompts):
        raise ValueError("input_boxes_labels_per_prompt length must match text_prompts")

    results = {}
    for i, text_prompt in enumerate(text_prompts):
        boxes = input_boxes_per_prompt[i] if input_boxes_per_prompt else None
        labels = input_boxes_labels_per_prompt[i] if input_boxes_labels_per_prompt else None
        results[text_prompt] = process_single_prompt(
            predictor, image, text_prompt=text_prompt,
            input_boxes=boxes, input_boxes_labels=labels,
        )

    return results


def process_with_exemplars(
    predictor: SAM3Predictor,
    image: Image.Image,
    bboxes: List[List[float]],
    labels: Optional[List[int]] = None,
    text: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Process image using bounding box exemplars (same-image visual prompts).

    Boxes define regions in the image that represent the visual concept.
    SAM3 will find all instances matching that visual appearance.

    Args:
        predictor: SAM3Predictor instance.
        image: PIL Image to process.
        bboxes: List of exemplar boxes in xyxy format [x1, y1, x2, y2].
        labels: Optional labels per box (0=negative, 1=positive). Defaults to all positive.
        text: Optional text prompts to combine with exemplars.

    Returns:
        Dictionary with 'instance_masks' and 'semantic_seg' tensors.
    """
    if labels is None:
        labels = [1] * len(bboxes)

    text_prompt = text[0] if text else None

    return process_single_prompt(
        predictor, image,
        text_prompt=text_prompt,
        input_boxes=bboxes,
        input_boxes_labels=labels,
    )


# ── Tensor utilities ─────────────────────────────────────────────────────────


def resize_mask_tensor(
    mask: torch.Tensor,
    target_h: int,
    target_w: int,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """Resize a mask tensor to target dimensions."""
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(target_h, target_w), mode=mode, align_corners=align_corners)
        return mask.squeeze(0).squeeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
        mask = F.interpolate(mask, size=(target_h, target_w), mode=mode, align_corners=align_corners)
        return mask.squeeze(0)
    elif mask.ndim == 4:
        return F.interpolate(mask, size=(target_h, target_w), mode=mode, align_corners=align_corners)
    else:
        raise ValueError(f"Mask must be 2D, 3D, or 4D, got {mask.ndim}D")


def get_semantic_segmentation_tensor(
    semantic_seg: torch.Tensor,
    target_h: int,
    target_w: int,
    threshold: Optional[float] = None,
) -> torch.Tensor:
    """Extract and resize semantic segmentation mask.

    Handles both [batch, 1, H, W] and [1, H, W] or [H, W] shapes.
    """
    if semantic_seg.ndim == 4:
        seg = semantic_seg[0, 0]
    elif semantic_seg.ndim == 3:
        seg = semantic_seg[0]
    else:
        seg = semantic_seg

    seg_resized = resize_mask_tensor(seg, target_h, target_w)
    if threshold is not None:
        seg_resized = (seg_resized > threshold).float()
    return seg_resized


def get_instance_masks_tensor(
    instance_masks: torch.Tensor,
    target_h: int,
    target_w: int,
    threshold: Optional[float] = None,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Extract and resize instance masks."""
    if instance_masks.ndim == 4:
        masks = instance_masks[0]
    else:
        masks = instance_masks
    if top_k is not None:
        masks = masks[:top_k]
    masks_resized = resize_mask_tensor(masks, target_h, target_w)
    if threshold is not None:
        masks_resized = (masks_resized > threshold).float()
    return masks_resized


# ── Visualization utilities ──────────────────────────────────────────────────


def overlay_mask_on_image(
    image: Image.Image,
    mask: torch.Tensor,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> torch.Tensor:
    """Overlay a mask on an image, returning (3, H, W) tensor."""
    import numpy as np
    image_array = np.array(image)
    if image_array.ndim != 3:
        raise ValueError(f"Expected 3-channel image, got {image_array.ndim}D")
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()

    mask = mask.cpu()
    if mask.shape != image_tensor.shape[1:]:
        mask = resize_mask_tensor(mask, image_tensor.shape[1], image_tensor.shape[2])

    mask_3ch = mask.unsqueeze(0).expand(3, -1, -1)
    color_tensor = torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
    overlay = image_tensor * (1 - mask_3ch * alpha) + mask_3ch * color_tensor * alpha
    return torch.clamp(overlay, 0, 255)


def create_combined_overlay_tensor(
    image: Image.Image,
    masks: Dict[str, torch.Tensor],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.4,
) -> torch.Tensor:
    """Create combined overlay of multiple masks, returning (3, H, W) tensor."""
    import numpy as np

    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    image_array = np.array(image)
    if image_array.ndim != 3:
        raise ValueError(f"Expected 3-channel image, got {image_array.ndim}D")
    overlay = torch.from_numpy(image_array).permute(2, 0, 1).float()

    for idx, (prompt, mask) in enumerate(masks.items()):
        color = colors[idx % len(colors)]
        mask = mask.cpu()
        if mask.shape != overlay.shape[1:]:
            mask = resize_mask_tensor(mask, overlay.shape[1], overlay.shape[2])
        mask_3ch = mask.unsqueeze(0).expand(3, -1, -1)
        color_tensor = torch.tensor(color, dtype=torch.float32).view(3, 1, 1)
        overlay = overlay * (1 - mask_3ch) + mask_3ch * (color_tensor * alpha + overlay * (1 - alpha))

    return torch.clamp(overlay, 0, 255)


# ── Numpy convenience functions ──────────────────────────────────────────────


def get_semantic_segmentation_numpy(semantic_seg, image_size, threshold=0.5):
    import numpy as np
    return get_semantic_segmentation_tensor(
        semantic_seg, image_size[1], image_size[0], threshold=threshold
    ).cpu().numpy()


def get_instance_masks_numpy(instance_masks, image_size, threshold=0.5, top_k=None):
    import numpy as np
    return get_instance_masks_tensor(
        instance_masks, image_size[1], image_size[0], threshold=threshold, top_k=top_k
    ).cpu().numpy()


def overlay_mask_on_image_numpy(image, mask, color=(255, 0, 0), alpha=0.5):
    import numpy as np
    t = overlay_mask_on_image(image, mask, color, alpha)
    return t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def create_combined_overlay(image, masks, colors=None, alpha=0.4):
    import numpy as np
    t = create_combined_overlay_tensor(image, masks, colors, alpha)
    return t.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


# ── SAM3 Video predictor ─────────────────────────────────────────────────────


class SAM3VideoPredictor:
    """Thin wrapper around HF Sam3VideoModel + Sam3VideoProcessor for video PCS.

    Supports both pre-loaded video inference (all frames available upfront) and
    streaming inference (frame-by-frame as they arrive). Tracks object identities
    across frames using text prompts.
    """

    def __init__(self, model: "Sam3VideoModel", processor: "Sam3VideoProcessor", device: str):
        self.model = model
        self.processor = processor
        self.device = device


def load_sam3_video_model(
    model_name: str = "facebook/sam3",
    device: Optional[str] = None,
    image_size: Optional[int] = None,
) -> SAM3VideoPredictor:
    """Load SAM3 video model and processor from HuggingFace.

    Args:
        model_name: HuggingFace model identifier. Defaults to "facebook/sam3".
        device: Device to load model on ("cuda", "cpu", or None for auto-detect).
        image_size: If set, override the model's internal input resolution
            (default 1008). Lower values (e.g. 560) reduce memory/compute at
            the cost of accuracy. Applied via Sam3VideoConfig + processor size.

    Returns:
        SAM3VideoPredictor wrapping the model and processor.
    """
    if Sam3VideoModel is None or Sam3VideoProcessor is None:
        raise ImportError(
            "transformers with SAM3 Video support is required. "
            "Install/upgrade with: pip install -U transformers"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if image_size is not None:
        from transformers import Sam3VideoConfig
        config = Sam3VideoConfig.from_pretrained(model_name)
        config.image_size = int(image_size)
        model = Sam3VideoModel.from_pretrained(model_name, config=config, device_map=device)
        processor = Sam3VideoProcessor.from_pretrained(
            model_name,
            size={"height": int(image_size), "width": int(image_size)},
        )
    else:
        model = Sam3VideoModel.from_pretrained(model_name, device_map=device)
        processor = Sam3VideoProcessor.from_pretrained(model_name)

    return SAM3VideoPredictor(model, processor, device)


def process_video(
    predictor: SAM3VideoPredictor,
    video_frames,
    text_prompts: Union[str, List[str]],
    max_frame_num_to_track: Optional[int] = None,
    start_frame_idx: int = 0,
    reverse: bool = False,
    show_progress_bar: bool = False,
    processing_device: str = "cpu",
    video_storage_device: str = "cpu",
) -> Dict[int, Dict]:
    """Run pre-loaded SAM3 video inference with one or more text prompts.

    Detects and tracks every object instance matching the given concept(s) across
    all frames, preserving object identities.

    Args:
        predictor: SAM3VideoPredictor instance.
        video_frames: Video as a list/array of frames (e.g. from
            `transformers.video_utils.load_video`).
        text_prompts: Single prompt or list of prompts (multi-prompt is processed
            in a single pass with shared vision features).
        max_frame_num_to_track: Cap on frames to track; None = full video.
        start_frame_idx: Frame to start propagation from.
        reverse: Propagate backwards through video.
        show_progress_bar: Display tqdm progress bar during propagation.
        processing_device: Device for video preprocessing.
        video_storage_device: Device to hold decoded frames.

    Returns:
        Dict mapping frame_idx → postprocessed outputs dict with keys:
        'object_ids', 'scores', 'boxes' (XYXY abs coords),
        'masks' (binary at original resolution), 'prompt_to_obj_ids'.
    """
    model, processor, device = predictor.model, predictor.processor, predictor.device

    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        processing_device=processing_device,
        video_storage_device=video_storage_device,
    )
    processor.add_text_prompt(inference_session=inference_session, text=text_prompts)

    outputs_per_frame: Dict[int, Dict] = {}
    propagate_kwargs = {
        "inference_session": inference_session,
        "start_frame_idx": start_frame_idx,
        "reverse": reverse,
        "show_progress_bar": show_progress_bar,
    }
    if max_frame_num_to_track is not None:
        propagate_kwargs["max_frame_num_to_track"] = max_frame_num_to_track

    for model_outputs in model.propagate_in_video_iterator(**propagate_kwargs):
        processed = processor.postprocess_outputs(inference_session, model_outputs)
        outputs_per_frame[model_outputs.frame_idx] = processed

    return outputs_per_frame


def init_video_streaming_session(
    predictor: SAM3VideoPredictor,
    text_prompts: Union[str, List[str]],
    processing_device: str = "cpu",
    video_storage_device: str = "cpu",
):
    """Initialize a streaming SAM3 video session (no preloaded video).

    Note: streaming disables hotstart heuristics, so expect more false positives
    and duplicate tracks than the pre-loaded path. Use `process_video` when all
    frames are available.

    Args:
        predictor: SAM3VideoPredictor instance.
        text_prompts: Single prompt or list of prompts.
        processing_device: Device used for per-frame preprocessing.
        video_storage_device: Device used to retain processed frames.

    Returns:
        Sam3VideoInferenceSession ready to consume frames via `process_streaming_frame`.
    """
    processor = predictor.processor
    session = processor.init_video_session(
        inference_device=predictor.device,
        processing_device=processing_device,
        video_storage_device=video_storage_device,
    )
    processor.add_text_prompt(inference_session=session, text=text_prompts)
    return session


def process_streaming_frame(
    predictor: SAM3VideoPredictor,
    inference_session,
    frame,
    reverse: bool = False,
) -> Dict:
    """Run streaming SAM3 video inference on a single frame.

    Args:
        predictor: SAM3VideoPredictor instance.
        inference_session: Session from `init_video_streaming_session`.
        frame: Single video frame (PIL Image, numpy array, or tensor).
        reverse: Propagate backwards (rarely used in streaming).

    Returns:
        Postprocessed outputs dict (same shape as `process_video` per-frame).
    """
    model, processor = predictor.model, predictor.processor

    inputs = processor(images=frame, device=predictor.device, return_tensors="pt").to(
        model.device
    )
    model_outputs = model(
        inference_session=inference_session,
        frame=inputs.pixel_values[0],
        reverse=reverse,
    )
    return processor.postprocess_outputs(
        inference_session, model_outputs, original_sizes=inputs.original_sizes
    )
