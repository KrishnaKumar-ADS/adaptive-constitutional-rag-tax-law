"""
finetuning/train_qlora.py
QLoRA fine-tuning for Qwen3-8B (base model) on legal Q&A data.
"""

import os
import sys
import yaml

from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig


# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────

def load_config(path: str = "finetuning/qlora_config.yaml") -> dict:
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ─────────────────────────────────────────────
# 2. Dataset
# ─────────────────────────────────────────────

def load_datasets(config: dict):
    for key in ("train_file", "eval_file"):
        if not os.path.exists(config["dataset"][key]):
            sys.exit(f"[ERROR] Dataset not found: {config['dataset'][key]}")

    train_ds = load_dataset("json", data_files=config["dataset"]["train_file"], split="train")
    eval_ds  = load_dataset("json", data_files=config["dataset"]["eval_file"],  split="train")

    print(f"  Train samples : {len(train_ds)}")
    print(f"  Eval  samples : {len(eval_ds)}")
    print(f"  Sample keys   : {list(train_ds[0].keys())}")
    return train_ds, eval_ds


def apply_chat_template(dataset, tokenizer):
    def _format(example):
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        }
    return dataset.map(_format, remove_columns=dataset.column_names,
                       desc="Applying chat template")


# ─────────────────────────────────────────────
# 3. Model + tokenizer
# ─────────────────────────────────────────────

def load_model(config: dict):
    q = config["quantization"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name        = config["model"]["name"],
        max_seq_length    = 4096,
        dtype             = None,
        load_in_4bit      = q["load_in_4bit"],
        trust_remote_code = config["model"]["trust_remote_code"],
    )

    # Set chat template explicitly for base model
    for template in ("qwen-3", "qwen-2.5"):
        try:
            tokenizer = get_chat_template(tokenizer, chat_template=template)
            print(f"  Chat template  : {template} (via Unsloth)")
            break
        except Exception:
            continue
    else:
        print("  Chat template  : using tokenizer built-in")

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ─────────────────────────────────────────────
# 4. LoRA / DoRA
# ─────────────────────────────────────────────

def attach_lora(model, config: dict):
    lora = config["lora"]

    target = lora["target_modules"]

    # Guard: PEFT 0.19+ iterates a bare string as characters → explodes.
    # If YAML still has "all-linear" string, convert to explicit list.
    if isinstance(target, str):
        print(f"  WARNING: target_modules is a string '{target}' — converting to explicit list")
        target = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

    print(f"  Target modules : {target}")

    model = FastLanguageModel.get_peft_model(
        model,
        r              = lora["r"],
        target_modules = target,
        lora_alpha     = lora["lora_alpha"],
        lora_dropout   = lora["lora_dropout"],
        bias           = lora["bias"],
        use_dora       = lora["use_dora"],
        use_gradient_checkpointing = "unsloth",
        random_state   = 42,
    )
    model.print_trainable_parameters()
    return model


# ─────────────────────────────────────────────
# 5. Trainer
# ─────────────────────────────────────────────

def build_trainer(model, tokenizer, train_ds, eval_ds, config: dict):
    t = config["training"]

    sft_config = SFTConfig(
        output_dir                  = t["output_dir"],
        num_train_epochs            = t["num_train_epochs"],
        per_device_train_batch_size = t["per_device_train_batch_size"],
        per_device_eval_batch_size  = t["per_device_eval_batch_size"],
        gradient_accumulation_steps = t["gradient_accumulation_steps"],
        learning_rate               = float(t["learning_rate"]),
        optim                       = t["optim"],
        weight_decay                = t["weight_decay"],
        max_grad_norm               = t["max_grad_norm"],
        warmup_ratio                = t["warmup_ratio"],
        lr_scheduler_type           = t["lr_scheduler_type"],
        bf16                        = t["bf16"],
        fp16                        = False,
        eval_strategy               = t["evaluation_strategy"],
        save_strategy               = t["save_strategy"],
        save_total_limit            = t["save_total_limit"],
        load_best_model_at_end      = t["load_best_model_at_end"],
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        logging_steps               = t["logging_steps"],
        report_to                   = t["report_to"],
        max_seq_length              = 4096,
        dataset_text_field          = "text",
        packing                     = False,
    )

    return SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        args             = sft_config,
    )


# ─────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────

def main():
    config_path = os.environ.get("QLORA_CONFIG", "finetuning/qlora_config.yaml")

    print(f"\n{'='*62}")
    print(f"  QLoRA Fine-Tuning  |  Qwen3-8B (base)  |  A100")
    print(f"  Config: {config_path}")
    print(f"{'='*62}\n")

    config = load_config(config_path)
    os.makedirs(config["training"]["output_dir"], exist_ok=True)

    print("[1/5]  Loading datasets ...")
    train_ds, eval_ds = load_datasets(config)

    print("\n[2/5]  Loading Qwen3-8B in 4-bit (NF4) ...")
    model, tokenizer = load_model(config)

    print("\n[3/5]  Injecting LoRA / DoRA adapters ...")
    model = attach_lora(model, config)

    print("\n[4/5]  Applying chat template to datasets ...")
    train_ds = apply_chat_template(train_ds, tokenizer)
    eval_ds  = apply_chat_template(eval_ds,  tokenizer)
    print(f"  Sample (first 200 chars): {train_ds[0]['text'][:200]!r}")

    eff_batch = (config["training"]["per_device_train_batch_size"]
                 * config["training"]["gradient_accumulation_steps"])
    print(f"\n[5/5]  Starting training ...  (effective batch = {eff_batch})\n")

    trainer = build_trainer(model, tokenizer, train_ds, eval_ds, config)
    trainer.train()

    adapter_dir = os.path.join(config["training"]["output_dir"], "final_adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    print(f"\n{'='*62}")
    print(f"  Training complete.")
    print(f"  Adapter saved -> {adapter_dir}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()