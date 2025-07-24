"""Fine-tune a LLaMA/Vicuna model on Q/A data using LoRA (PEFT) and optional 4-bit quantization.

Usage example:
    python finetune_llama_lora.py \
        --data_path data/qa.json \
        --model_name lmsys/vicuna-7b-v1.5 \
        --output_dir ./llama-base_fine-tuned \
        --batch_size 2 \
        --epochs 3 \
        --learning_rate 5e-5 \
        --max_length 512 \
        --save_steps 500 \
        --quantization

Environment:
    HF_AUTH_TOKEN must be set if the model is gated on Hugging Face Hub.

The script:
- Cleans up imports and structure (PEP 8 compliant).
- Adds safety checks and helpful CLI flags (seed, grad accumulation, eval steps, etc.).
- Implements LoRA with PEFT for efficient fine-tuning.
- Optionally loads the model in 4-bit using bitsandbytes (bnb).
- Uses a simple Q/A dataset where each item has keys: {"question": str, "answer": str}.
- Saves tokenizer and adapter weights in output_dir.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
)
from peft import LoraConfig, TaskType, get_peft_model


# --------------------------------------------------------------------------------------
# DATASET
# --------------------------------------------------------------------------------------


class QADataset(Dataset):
    """Simple Q/A dataset. Expects a JSON array of {"question": str, "answer": str}.

    The prompt is built as: "Question: {question}\nAnswer: {answer}" (customize as needed).
    """

    def __init__(self, data_path: str, tokenizer: AutoTokenizer, max_length: int = 512) -> None:
        with open(data_path, "r", encoding="utf-8") as f:
            self.data: List[Dict[str, str]] = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        qa = self.data[idx]
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip()
        prompt = f"Question: {question}\nAnswer: {answer}"

        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# --------------------------------------------------------------------------------------
# ARGUMENTS
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune LLaMA/Vicuna model on Q/A data (LoRA + optional 4-bit).")

    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSON file with Q/A data.")
    parser.add_argument("--model_name", type=str, default="lmsys/vicuna-7b-v1.5", help="Pretrained model name or path.")
    parser.add_argument("--output_dir", type=str, default="./llama-base_fine-tuned", help="Directory to save the fine-tuned model.")

    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X steps.")
    parser.add_argument("--eval_steps", type=int, default=0, help="Run evaluation every X steps (0 to disable).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")

    parser.add_argument("--quantization", action="store_true", help="Use 4-bit quantization via bitsandbytes.")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated list of target modules for LoRA (depends on model architecture)",
    )

    return parser.parse_args()


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # GPU check
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU not detected. Please ensure CUDA is available.")

    # Hugging Face token
    token = os.getenv("HF_AUTH_TOKEN")
    if token is None:
        raise EnvironmentError("HF_AUTH_TOKEN not set. Export your Hugging Face access token.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_auth_token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading (4-bit optional)
    if args.quantization:
        print("Using 4-bit quantization with bitsandbytes.")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            device_map="auto",
            use_auth_token=token,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=token,
        )

    # Gradient checkpointing & input grads
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # LoRA setup
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)

    # Dataset & collator
    dataset = QADataset(args.data_path, tokenizer, max_length=args.max_length)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=True,
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        evaluation_strategy="steps" if args.eval_steps > 0 else "no",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        seed=args.seed,
        report_to=["none"],  # disable wandb/hf by default; change if needed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save adapter / model
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()