# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
from pathlib import Path

import tensorrt as trt


def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool,
    workspace_gb: float,
    batch: int | None,
    min_shape: tuple[int, int, int, int] | None,
    opt_shape: tuple[int, int, int, int] | None,
    max_shape: tuple[int, int, int, int] | None,
    input_name: str | None,
    # Optional second input (e.g., lr_feats for JAFAR upsampler-only)
    input_name2: str | None = None,
    min_shape2: tuple[int, int, int, int] | None = None,
    opt_shape2: tuple[int, int, int, int] | None = None,
    max_shape2: tuple[int, int, int, int] | None = None,
):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # Network creation flags differ by TRT major version:
    #   TRT 11+: precision/INT8/FP16 BuilderFlags were removed — the network is
    #            STRONGLY_TYPED and precision is taken from the ONNX dtypes.
    #   TRT 8.x: needs the EXPLICIT_BATCH flag.
    #   TRT 10.x: explicit-batch is implicit; STRONGLY_TYPED is optional.
    NDC = trt.NetworkDefinitionCreationFlag
    strongly_typed = hasattr(NDC, "STRONGLY_TYPED")
    if strongly_typed:
        network_flags = 1 << int(NDC.STRONGLY_TYPED)
    elif hasattr(NDC, "EXPLICIT_BATCH"):
        network_flags = 1 << int(NDC.EXPLICIT_BATCH)
    else:
        network_flags = 0
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error {i}: {parser.get_error(i)}")
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    workspace_bytes = int(workspace_gb * (1024 ** 3))
    # TensorRT >= 8.6 uses memory pool API; older versions use max_workspace_size
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    except AttributeError:
        # Fallback for older TRT
        try:
            config.max_workspace_size = workspace_bytes
        except AttributeError:
            pass

    # FP16: only a *weakly-typed* build (TRT 8/10) uses the FP16 builder flag.
    # A strongly-typed network (TRT 10.1+/11) takes precision from the ONNX
    # dtypes — these ONNX are FP16, so the engine is FP16 with no flag (and the
    # FP16 flag was removed entirely in TRT 11).
    if fp16 and not strongly_typed and hasattr(trt.BuilderFlag, "FP16"):
        if getattr(builder, "platform_has_fast_fp16", True):
            config.set_flag(trt.BuilderFlag.FP16)

    profile = None
    # Determine input tensor(s)
    input_tensor = network.get_input(0)
    if input_name and input_tensor.name != input_name:
        for i in range(network.num_inputs):
            if network.get_input(i).name == input_name:
                input_tensor = network.get_input(i)
                break
    input_tensor2 = None
    if network.num_inputs > 1:
        # Default to second input if present; allow override by name
        candidate2 = network.get_input(1)
        input_tensor2 = candidate2
        if input_name2:
            for i in range(network.num_inputs):
                if network.get_input(i).name == input_name2:
                    input_tensor2 = network.get_input(i)
                    break

    if any(v is not None for v in (min_shape, opt_shape, max_shape, min_shape2, opt_shape2, max_shape2)):
        # Dynamic shapes path
        profile = builder.create_optimization_profile()
        min_s = min_shape or opt_shape or max_shape
        opt_s = opt_shape or max_shape or min_shape
        max_s = max_shape or opt_shape or min_shape
        if min_s is None or opt_s is None or max_s is None:
            raise ValueError("Provide at least one of min/opt/max shapes; others will be inferred in order min<=opt<=max")
        profile.set_shape(input_tensor.name, min_s, opt_s, max_s)
        if input_tensor2 is not None:
            # Second input shapes: if not provided, default to static current ONNX dims
            ms2 = min_shape2 or opt_shape2 or max_shape2
            os2 = opt_shape2 or max_shape2 or min_shape2
            xs2 = max_shape2 or opt_shape2 or min_shape2
            if ms2 is None or os2 is None or xs2 is None:
                # Fallback: use the ONNX-declared shape if fully static
                shp = tuple(input_tensor2.shape)
                if any(int(d) <= 0 for d in shp):
                    raise ValueError("Dynamic build requires min/opt/max for second input as well")
                ms2 = os2 = xs2 = shp  # type: ignore
            profile.set_shape(input_tensor2.name, ms2, os2, xs2)
        config.add_optimization_profile(profile)
    else:
        # Static shape; optionally set batch
        if batch is not None:
            shape = list(input_tensor.shape)
            shape[0] = batch
            input_tensor.shape = tuple(shape)
            if input_tensor2 is not None:
                shape2 = list(input_tensor2.shape)
                shape2[0] = batch
                input_tensor2.shape = tuple(shape2)

    # Build (handle API differences across TRT versions)
    engine_bytes = None
    try:
        # TRT >= 8.5
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        # Older API
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        engine_bytes = engine.serialize()
    if engine_bytes is None:
        raise RuntimeError("Failed to build TensorRT engine (no bytes)")
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(engine_bytes)

    if strongly_typed:
        precision = "strongly-typed (precision from ONNX dtypes)"
    elif hasattr(trt.BuilderFlag, "FP16"):
        precision = f"fp16={'on' if config.get_flag(trt.BuilderFlag.FP16) else 'off'}"
    else:
        precision = "fp32"
    print(f"built_engine={engine_path} {precision} workspace_gb={workspace_gb}")


def parse_shape(text: str):
    parts = [int(x) for x in text.split(",")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("shape must be b,c,h,w (4 ints) or b,c,s (3 ints)")
    return tuple(parts)


def main():
    p = argparse.ArgumentParser(description="Build TensorRT engine from ONNX")
    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument("--engine", required=True, help="Output path for .plan engine")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 if supported")
    p.add_argument("--workspace_gb", type=float, default=4.0, help="Builder workspace size in GB")
    p.add_argument("--batch", type=int, default=1, help="Static batch size (for static shape builds)")
    p.add_argument("--min_shape", type=parse_shape, help="Min shape (b,c,h,w) or (b,c,s) for dynamic build")
    p.add_argument("--opt_shape", type=parse_shape, help="Opt shape (b,c,h,w) or (b,c,s) for dynamic build")
    p.add_argument("--max_shape", type=parse_shape, help="Max shape (b,c,h,w) or (b,c,s) for dynamic build")
    p.add_argument("--input_name", type=str, help="Input tensor name (default: first input)")
    # Optional second input (e.g., lr_feats for JAFAR upsampler-only)
    p.add_argument("--input_name2", type=str, help="Second input tensor name (if present)")
    p.add_argument("--min_shape2", type=parse_shape, help="Second input min shape b,c,h,w (dynamic build)")
    p.add_argument("--opt_shape2", type=parse_shape, help="Second input opt shape b,c,h,w (dynamic build)")
    p.add_argument("--max_shape2", type=parse_shape, help="Second input max shape b,c,h,w (dynamic build)")
    args = p.parse_args()

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=args.fp16,
        workspace_gb=args.workspace_gb,
        batch=args.batch,
        min_shape=args.min_shape,
        opt_shape=args.opt_shape,
        max_shape=args.max_shape,
        input_name=args.input_name,
        input_name2=args.input_name2,
        min_shape2=args.min_shape2,
        opt_shape2=args.opt_shape2,
        max_shape2=args.max_shape2,
    )


if __name__ == "__main__":
    main()
