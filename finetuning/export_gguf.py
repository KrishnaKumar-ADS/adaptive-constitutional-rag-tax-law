"""
finetuning/export_gguf.py

Convert the merged FP16 model to GGUF Q4_K_M format using llama.cpp.

This was run on Google Colab A100 (see notebooks/Copy_of_Untitled0.ipynb cell 17).
The Unsloth save_pretrained_gguf path was skipped due to a DoRA
(lora_magnitude_vector) key bug in Unsloth 2026.6.x — llama.cpp is the
correct export path for DoRA-trained models.

Prerequisites:
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
    pip install gguf sentencepiece protobuf
    cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON
    cmake --build llama.cpp/build --config Release -j$(nproc)

Usage (from project root):
    python finetuning/export_gguf.py \\
        --merged-path outputs/qwen-tax-merged \\
        --output-path finetuning/checkpoints/qwen3-8b-tax-q4km.gguf \\
        --llama-cpp-dir /path/to/llama.cpp

If you already have finetuning/checkpoints/qwen3-8b-tax-q4km.gguf (~5.03 GB),
this script is documentation only.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys


def patch_config(merged_path: str):
    """Ensure config.json has model_type='qwen3' and fix tokenizer quirks."""
    # Patch config.json
    cfg_path = os.path.join(merged_path, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    if cfg.get("model_type") != "qwen3":
        cfg["model_type"] = "qwen3"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print("✅ Patched config.json model_type → qwen3")

    # Patch tokenizer_config.json
    tok_path = os.path.join(merged_path, "tokenizer_config.json")
    with open(tok_path) as f:
        tok_cfg = json.load(f)
    if isinstance(tok_cfg.get("extra_special_tokens"), list):
        tok_cfg["extra_special_tokens"] = {}
        with open(tok_path, "w") as f:
            json.dump(tok_cfg, f, indent=2)
        print("✅ Patched tokenizer_config.json extra_special_tokens")


def convert_to_gguf(
    merged_path: str,
    output_path: str,
    llama_cpp_dir: str,
):
    """
    Two-step conversion:
      1. Merged HF safetensors → F16 GGUF (via convert_hf_to_gguf.py)
      2. F16 GGUF → Q4_K_M GGUF (via llama-quantize)
    """
    f16_path = output_path.replace(".gguf", "-f16.gguf")

    # Step 1: HF → F16 GGUF
    print("\nStep 1/2: Converting merged model → F16 GGUF ...")
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    subprocess.run(
        [sys.executable, convert_script, merged_path,
         "--outfile", f16_path, "--outtype", "f16"],
        check=True,
    )

    f16_size = os.path.getsize(f16_path) / 1024**3
    print(f"✅ F16 GGUF: {f16_size:.2f} GB")

    # Step 2: F16 → Q4_K_M
    print("\nStep 2/2: Quantizing F16 → Q4_K_M ...")
    quantize_bin = os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize")
    subprocess.run(
        [quantize_bin, f16_path, output_path, "Q4_K_M"],
        check=True,
    )

    # Clean up F16 intermediate
    os.remove(f16_path)
    print("✅ Deleted F16 intermediate")

    final_size = os.path.getsize(output_path) / 1024**3
    print(f"\n✅ GGUF ready: {output_path}")
    print(f"Size: {final_size:.2f} GB")


def main():
    parser = argparse.ArgumentParser(description="Export merged model to GGUF Q4_K_M")
    parser.add_argument("--merged-path", default="outputs/qwen-tax-merged")
    parser.add_argument("--output-path", default="finetuning/checkpoints/qwen3-8b-tax-q4km.gguf")
    parser.add_argument("--llama-cpp-dir", default="/content/llama.cpp")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    patch_config(args.merged_path)
    convert_to_gguf(args.merged_path, args.output_path, args.llama_cpp_dir)


if __name__ == "__main__":
    main()
