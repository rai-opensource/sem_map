# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
import argparse
from collections import Counter
from pathlib import Path

import onnx


def _fmt_shape(t):
    # t: onnx.TensorShapeProto
    dims = []
    for d in t.dim:
        if d.dim_param:
            dims.append(d.dim_param)
        elif d.dim_value:
            dims.append(str(d.dim_value))
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"


def _print_ios(model: onnx.ModelProto):
    print("Inputs:")
    for vi in model.graph.input:
        name = vi.name
        t = vi.type.tensor_type
        elem = onnx.TensorProto.DataType.Name(t.elem_type)
        shape = _fmt_shape(t.shape)
        print(f"  - {name}: {elem} {shape}")
    print("Outputs:")
    for vi in model.graph.output:
        name = vi.name
        t = vi.type.tensor_type
        elem = onnx.TensorProto.DataType.Name(t.elem_type)
        shape = _fmt_shape(t.shape)
        print(f"  - {name}: {elem} {shape}")


def _print_summary(model: onnx.ModelProto):
    opset_imports = {op.domain or "ai.onnx": op.version for op in model.opset_import}
    print(f"ir_version={model.ir_version} opsets={opset_imports} nodes={len(model.graph.node)} initializers={len(model.graph.initializer)}")


def _print_node_stats(model: onnx.ModelProto, limit: int | None):
    kinds = Counter(n.op_type for n in model.graph.node)
    print("Node type histogram:")
    for op, cnt in kinds.most_common(limit):
        print(f"  - {op}: {cnt}")


def _has_dynamic(t) -> bool:
    for d in t.dim:
        if d.dim_param:
            return True
    return False


def _print_dynamic_axes(model: onnx.ModelProto):
    print("Dynamic axes (by dim_param):")
    for vi in list(model.graph.input) + list(model.graph.output):
        t = vi.type.tensor_type
        if _has_dynamic(t.shape):
            dims = [d.dim_param or ("?" if not d.dim_value else str(d.dim_value)) for d in t.shape.dim]
            print(f"  - {vi.name}: {dims}")


def main():
    p = argparse.ArgumentParser(description="Inspect ONNX model IO shapes and node stats")
    p.add_argument("--model", required=True, help="Path to .onnx model")
    p.add_argument("--nodes", action="store_true", help="Print node type histogram")
    p.add_argument("--limit", type=int, default=20, help="Limit number of node types to display")
    args = p.parse_args()

    path = Path(args.model)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    _print_summary(model)
    _print_ios(model)
    _print_dynamic_axes(model)
    if args.nodes:
        _print_node_stats(model, args.limit)


if __name__ == "__main__":
    main()
