# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from ..common import prepare_image
from ..backends.tensorrt import TrtRunner


def load_jafar_torch(
    backbone_name: str = "vit_small_patch14_reg4_dinov2",
    project_root: str | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Load JAFAR model and backbone using local Hydra configs.

    Returns:
        (model, backbone)
    """
    from ..common import get_sem_map_root

    # JAFAR is optional and only needed for the torch backend / ONNX export.
    # Point JAFAR_ROOT at a clone of https://github.com/PaulCouairon/JAFAR;
    # defaults to <repo>/external/JAFAR.
    repo_root = get_sem_map_root()
    jafar_project_root = Path(
        os.environ.get("JAFAR_ROOT", repo_root / "external" / "JAFAR")
    ).expanduser().resolve()
    if not jafar_project_root.exists():
        raise RuntimeError(
            f"JAFAR repo not found at {jafar_project_root}. Clone "
            "https://github.com/PaulCouairon/JAFAR there or set JAFAR_ROOT. "
            "(Only needed for the torch upsampler backend and ONNX export; "
            "the runtime uses pre-exported JAFAR engines or bilinear upsampling.)"
        )
    jafar_config_dir = jafar_project_root / "config"

    overrides = [
        "val_dataloader.batch_size=1",
        "ratio=fixed",
        f"project_root={project_root or str(repo_root)}",
        f"backbone.name={backbone_name}",
    ]

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    script_dir = Path(__file__).parent
    rel_config_path = os.path.relpath(jafar_config_dir, start=str(script_dir))
    initialize(config_path=rel_config_path, version_base=None)
    cfg = compose(config_name="base", overrides=overrides)

    # Ensure Hydra targets like 'src.upsampler.JAFAR' and 'utils.img' resolve
    # by adding JAFAR's own package roots to sys.path.
    jafar_src_root = jafar_project_root / "src"
    jafar_utils_root = jafar_project_root / "utils"
    # Ensure JAFAR paths are first so their 'src' and 'utils' are imported
    for p in (jafar_utils_root, jafar_src_root, jafar_project_root):
        p_str = str(p)
        if p_str in sys.path:
            sys.path.remove(p_str)
        sys.path.insert(0, p_str)
    # Avoid package name conflicts: drop any preloaded 'src' or 'utils'
    for mod_name in ("src", "utils"):
        if mod_name in sys.modules:
            try:
                del sys.modules[mod_name]
            except Exception:
                pass

    try:
        import hydra_plugins.resolvers  # noqa: F401
    except Exception:
        pass

    from ..common import get_device
    device = get_device()
    backbone = instantiate(cfg.backbone)
    backbone.to(device)

    model = instantiate(cfg.model)
    model.to(device)
    model.eval()

    ckpt_path_a = jafar_project_root / "output" / "jafar" / f"{backbone.name}.pth"
    ckpt_path_b = Path.cwd() / "output" / "jafar" / f"{backbone.name}.pth"
    ckpt_path = ckpt_path_a if ckpt_path_a.exists() else ckpt_path_b
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"JAFAR checkpoint not found at {ckpt_path_a}. Download the JAFAR "
            "weights for this backbone from the upstream JAFAR releases."
        )

    state = torch.load(str(ckpt_path), weights_only=False)
    model.load_state_dict(state["jafar"])  # key name follows notebooks.py

    return model, backbone


def load_jafar_onnx(onnx_path: str):
    """Load JAFAR ONNX session."""
    from ..backends.onnx import create_session
    return create_session(onnx_path)


def load_jafar_trt(engine_path: str) -> TrtRunner:
    """Load JAFAR TensorRT runner."""
    return TrtRunner(engine_path)


def compute_jafar_upscaled_feats_from_batch_torch(
    image_batch: torch.Tensor,
    model: torch.nn.Module,
    lr_feats: torch.Tensor,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Compute JAFAR upscaled features from batch using PyTorch.

    Args:
        image_batch: Tensor (B, 3, H, W) normalized
        model: JAFAR upsampler model
        lr_feats: Low-resolution features from backbone
        output_size: Optional target output size (H, W). If None, uses image_batch spatial dimensions.

    Returns:
        hr_feats: High-resolution upscaled features
    """
    if output_size is None:
        target_h, target_w = int(image_batch.shape[2]), int(image_batch.shape[3])
    else:
        target_h, target_w = output_size

    hr_feats = model(image_batch, lr_feats, (target_h, target_w))
    return hr_feats


def compute_jafar_upscaled_feats_from_batch_onnx(
    session,
    image_batch: torch.Tensor,
    lr_feats: torch.Tensor,
    output_name: str = "hr_feats",
) -> torch.Tensor:
    """Compute JAFAR upscaled features from batch using ONNX."""
    from ..backends.onnx import run_session
    import numpy as np

    # Resolve two-input ONNX upsampler: expects (pixel_values, lr_feats)
    in_metas = list(session.get_inputs())
    if len(in_metas) < 2:
        raise RuntimeError(
            "ONNX upsampler expects two inputs: pixel_values and lr_feats"
        )

    # Map names robustly (fallback to positional if names differ)
    name_to_idx = {m.name: i for i, m in enumerate(in_metas)}
    if ("pixel_values" in name_to_idx) and ("lr_feats" in name_to_idx):
        idx_img = int(name_to_idx["pixel_values"])
        idx_lr = int(name_to_idx["lr_feats"])
    else:
        idx_img, idx_lr = 0, 1

    # Prepare numpy inputs per ONNX expected dtype
    def _to_numpy(t: torch.Tensor, meta) -> tuple[str, "np.ndarray"]:
        name = meta.name
        want_fp16 = "float16" in getattr(meta, "type", "tensor(float)")
        ten = t.detach()
        if want_fp16 and ten.dtype != torch.float16:
            ten = ten.half()
        elif (not want_fp16) and ten.dtype != torch.float32:
            ten = ten.float()
        arr = ten.cpu().numpy()
        if want_fp16 and arr.dtype != np.float16:
            arr = arr.astype(np.float16, copy=False)
        if (not want_fp16) and arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        return name, arr

    name_img, arr_img = _to_numpy(image_batch, in_metas[idx_img])
    name_lr, arr_lr = _to_numpy(lr_feats, in_metas[idx_lr])

    ort_out = run_session(session, {name_img: arr_img, name_lr: arr_lr}, [output_name])[0]
    device = image_batch.device
    hr_feats = torch.from_numpy(ort_out).to(device)
    return hr_feats


def compute_jafar_upscaled_feats_from_batch_trt(
    runner: TrtRunner,
    image_batch: torch.Tensor,
    lr_feats: torch.Tensor,
) -> torch.Tensor:
    """Compute JAFAR upscaled features from batch using TensorRT."""
    # Set input bindings
    in_names = [n for n, is_in in runner.binding_is_input.items() if is_in]
    if ("pixel_values" in in_names) and ("lr_feats" in in_names):
        input_img = "pixel_values"
        input_lr = "lr_feats"
    else:
        if len(in_names) < 2:
            raise RuntimeError("JAFAR engine requires two inputs")
        input_img, input_lr = in_names[:2]

    runner.set_bindings({input_img: image_batch, input_lr: lr_feats})

    # Determine output name
    out_names = [n for n, is_in in runner.binding_is_input.items() if not is_in]
    if len(out_names) < 1:
        raise RuntimeError("JAFAR engine requires at least one output")
    output_name = out_names[0]

    # Run inference
    outputs = runner.infer([output_name])
    return outputs[output_name]
