#!/usr/bin/env python3
"""
Simple sanity checker for ONNX detection models.

Verifies that the network has:
- An input shaped [1,3,960,960] (default expected)
- An output shaped [1,18900,6] (detectNet expectation)

Usage:
  python3 tools/check_model.py models/best.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    import onnx  # type: ignore
except ImportError:
    print("ERROR: onnx package not installed. Install with `python3 -m pip install onnx`.", file=sys.stderr)
    sys.exit(1)


def tensor_dims(value_info) -> list[int]:
    dims = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.dim_value:
            dims.append(int(d.dim_value))
        elif d.dim_param:
            # symbolic dim, treat as -1 for reporting
            dims.append(-1)
        else:
            dims.append(-1)
    return dims


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX detection model IO shapes for detectNet.")
    parser.add_argument("onnx", type=Path, help="Path to ONNX model")
    parser.add_argument("--input-shape", default="1x3x960x960", help="Expected input shape (default: 1x3x960x960)")
    parser.add_argument("--output-shape", default="1x18900x6", help="Expected output shape (default: 1x18900x6)")
    args = parser.parse_args()

    if not args.onnx.exists():
        print(f"ERROR: {args.onnx} not found", file=sys.stderr)
        sys.exit(1)

    expected_in = [int(x) for x in args.input_shape.lower().split("x")]
    expected_out = [int(x) for x in args.output_shape.lower().split("x")]

    model = onnx.load(str(args.onnx))
    graph = model.graph

    if not graph.input:
        print("ERROR: model has no inputs", file=sys.stderr)
        sys.exit(1)

    if not graph.output:
        print("ERROR: model has no outputs", file=sys.stderr)
        sys.exit(1)

    in_dims = tensor_dims(graph.input[0])
    out_dims = tensor_dims(graph.output[0])

    ok = True
    if in_dims != expected_in:
        print(f"ERROR: input shape {in_dims} does not match expected {expected_in}", file=sys.stderr)
        ok = False
    if out_dims != expected_out:
        print(f"ERROR: output shape {out_dims} does not match expected {expected_out}", file=sys.stderr)
        ok = False

    if ok:
        print(f"Model {args.onnx} OK: input {in_dims}, output {out_dims}")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
