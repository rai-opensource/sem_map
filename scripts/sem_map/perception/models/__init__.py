# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Models package - Model-specific logic and loaders

import yaml
from pathlib import Path

_RESOLVER_CACHE = {}

def load_models_yaml(yaml_path=None):
    """
    Load (and memoize) the models.yaml configuration, defaulting to config/models.yaml if not provided.
    Returns the parsed dict.
    """
    global _RESOLVER_CACHE
    if yaml_path is None:
        from sem_map.perception.common import get_sem_map_root
        yaml_path = get_sem_map_root() / "config" / "models.yaml"
    yaml_path = Path(yaml_path)
    if "_last_path" not in _RESOLVER_CACHE or _RESOLVER_CACHE["_last_path"] != str(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        _RESOLVER_CACHE.clear()
        _RESOLVER_CACHE.update(config)
        _RESOLVER_CACHE["_last_path"] = str(yaml_path)
    return _RESOLVER_CACHE


def resolve_artifact_path(family, kind, artifact_type, models=None, seg_layer=None, upsampler=None, upsample_factor=None):
    """
    Args:
        family (str): e.g. "dinov3s"
        kind (str): one of "backbone", "upsampler", "seg_layer"
        artifact_type (str): one of "torch", "onnx", "trt"
        models (dict): result from load_models_yaml(), or None
        seg_layer (str): for kind=="seg_layer", pick this key
        upsampler (str): for kind=="upsampler", pick this upsampler name (e.g. "jafar")
        upsample_factor (str|int): for upsampler factor, e.g. "x4" or 4
    Returns:
        Absolute Path to artifact as Path (rooted at sem_map_root when relative)
    """
    if models is None:
        models = load_models_yaml()
    top_dirs = {k: Path(models[k]) for k in ("torch_dir", "onnx_dir", "trt_dir") if k in models}
    fam = models["families"][family]

    def _abspath(p: Path) -> Path:
        if p.is_absolute():
            return p
        from sem_map.perception.common import get_sem_map_root
        return (get_sem_map_root() / p)

    if kind == "backbone":
        entry = fam["backbone"]
        fname = entry.get(artifact_type)
        if fname is None:
            return None
        dir_prefix = top_dirs.get(f"{artifact_type}_dir", Path(""))
        return _abspath(dir_prefix / fname)
    elif kind == "upsampler":
        upsampler_entry = fam.get("upsamplers", {}).get(upsampler, {})
        entry = upsampler_entry.get(artifact_type)
        if entry is None:
            return None
        if upsample_factor is not None:
            key = str(upsample_factor)
            if not key.startswith("x"):
                key = f"x{key}"
            fname = entry.get(key)
            if fname is None:
                return None
        else:
            fname = entry
        dir_prefix = top_dirs.get(f"{artifact_type}_dir", Path(""))
        return _abspath(dir_prefix / fname)
    elif kind == "seg_layer":
        seg_entry = fam.get("seg_layers", {}).get(seg_layer)
        if seg_entry is None:
            return None
        if isinstance(seg_entry, str):
            # Compact format: value is seg_layer_config, derive artifact name
            ext_map = {"torch": ".pth", "onnx": ".onnx", "trt": ".plan"}
            fname = f"{seg_layer}{ext_map[artifact_type]}"
        else:
            # Verbose format (backward compat)
            fname = seg_entry.get(artifact_type)
            if fname is None:
                return None
        dir_prefix = top_dirs.get(f"{artifact_type}_dir", Path(""))
        return _abspath(dir_prefix / fname)
    else:
        raise ValueError(f"Unknown kind {kind}. Must be backbone, upsampler, seg_layer.")


def get_num_classes(family, seg_layer, models=None):
    if models is None:
        models = load_models_yaml()
    fam = models["families"][family]
    entry = fam["seg_layers"][seg_layer]
    if isinstance(entry, str):
        # Compact format: read num_classes from seg_layer YAML
        cfg = load_seg_layer_config(entry)
        return cfg.get("num_classes")
    return entry.get("num_classes")


def load_seg_layer_config(config_name: str, config_dir: Path | None = None) -> dict:
    """Load seg_layer config YAML from config/seg_layers/.

    Args:
        config_name: Name of the config file (e.g., 'cocostuff_27.yaml')
        config_dir: Optional directory path; defaults to sem_map_root/config/seg_layers/

    Returns:
        Parsed YAML dict with 'name', 'num_classes', and 'categories' keys
    """
    if config_dir is None:
        from sem_map.perception.common import get_sem_map_root
        config_dir = get_sem_map_root() / "config" / "seg_layers"

    config_path = Path(config_dir) / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Seg layer config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_seg_layer_config_name(family: str, seg_layer: str, models=None) -> str | None:
    """Get the seg_layer_config filename for a given family/seg_layer.

    Args:
        family: Model family name (e.g., 'dinov3s')
        seg_layer: Segmentation layer name (e.g., 'dinov3s_sem_seg_city_v1')
        models: Optional pre-loaded models config

    Returns:
        Config filename (e.g., 'cityscapes_19.yaml') or None if not defined
    """
    if models is None:
        models = load_models_yaml()
    fam = models["families"].get(family)
    if fam is None:
        return None
    seg_layers = fam.get("seg_layers", {})
    entry = seg_layers.get(seg_layer)
    if entry is None:
        return None
    if isinstance(entry, str):
        return entry  # Compact format: value IS the config name
    return entry.get("seg_layer_config")


def get_categories(family: str, seg_layer: str, models=None) -> list[dict]:
    """Get full category list for a seg_layer.

    Args:
        family: Model family name (e.g., 'dinov3s')
        seg_layer: Segmentation layer name (e.g., 'dinov3s_sem_seg_city_v1')
        models: Optional pre-loaded models config

    Returns:
        List of category dicts with keys: id, name, isthing, optional color/traversability
    """
    config_name = get_seg_layer_config_name(family, seg_layer, models)
    if config_name is None:
        raise ValueError(f"No seg_layer_config defined for {family}/{seg_layer}")

    config = load_seg_layer_config(config_name)
    return config.get("categories", [])


def get_traversability_scores(family: str, seg_layer: str, models=None) -> dict[int, float]:
    """Extract traversability scores as {id: score} dict, defaulting missing to 0.0.

    Args:
        family: Model family name (e.g., 'dinov3s')
        seg_layer: Segmentation layer name (e.g., 'dinov3s_sem_seg_city_v1')
        models: Optional pre-loaded models config

    Returns:
        Dict mapping class ID to traversability score in [0.0, 1.0]
    """
    categories = get_categories(family, seg_layer, models)
    scores = {}
    for cat in categories:
        class_id = cat["id"]
        traversability = cat.get("traversability", 0.0)
        scores[class_id] = float(traversability)
    return scores
