"""
finetuning/merge_adapter.py

Merge the QLoRA/DoRA adapter into the base Qwen3-8B model and export
the result as a full FP16 checkpoint.  This was run on Google Colab A100
(see notebooks/Copy_of_Untitled0.ipynb cells 13-14).

If you already have finetuning/checkpoints/qwen3-8b-tax-q4km.gguf,
this script is documentation only — kept for reproducibility.
"""

import gc
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_and_save(
    base_model_name: str = "Qwen/Qwen3-8B",
    adapter_path: str = "outputs/qwen-tax-lora/final_adapter",
    output_path: str = "outputs/qwen-tax-merged",
    hf_token: str = None,
):
    """
    1. Load the base Qwen3-8B in FP16
    2. Attach the DoRA adapter
    3. Merge adapter weights into base
    4. Save the merged model as safetensors shards
    """
    print("1/4  Loading base model in FP16 ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=hf_token,
    )

    print("2/4  Attaching DoRA adapter ...")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    print("3/4  Merging DoRA weights ...")
    merged = peft_model.merge_and_unload()

    print("4/4  Saving merged FP16 model ...")
    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(
        output_path,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(output_path)

    del merged, peft_model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n✅ Merged model saved → {output_path}")


if __name__ == "__main__":
    merge_and_save()
