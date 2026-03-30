#!/usr/bin/env python3
"""Convert ONNX model from FP32 to FP16 for faster GPU inference."""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to FP16")
    parser.add_argument("input", help="Input FP32 ONNX model path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output FP16 model path (default: input_fp16.onnx)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        print("Install dependencies: pip install onnx onnxconverter-common")
        sys.exit(1)

    output_path = args.output
    if not output_path:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_fp16{ext}"

    print(f"Loading model: {args.input}")
    model = onnx.load(args.input)

    print("Converting to FP16 (mixed precision — Resize ops stay FP32)...")
    # keep_io_types: preserve FP32 inputs/outputs for compatibility
    # op_block_list: ops that don't support FP16 well in ONNX Runtime
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        op_block_list=["Resize", "GridSample"],
    )

    print(f"Saving to: {output_path}")
    onnx.save(model_fp16, output_path)

    original_size = os.path.getsize(args.input) / (1024 * 1024)
    fp16_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Size: {original_size:.1f} MB -> {fp16_size:.1f} MB ({fp16_size/original_size*100:.0f}%)")


if __name__ == "__main__":
    main()
