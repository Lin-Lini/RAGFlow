"""Fine-tune a Causal LM (Vicuna/LLaMA family) on Q/A pairs stored in Elasticsearch using LoRA (PEFT).

Highlights & improvements:
- PEP 8 formatting, clearer structure and typing.
- Robust Elasticsearch fetch with scroll API (handles >10k docs safely).
- Prompt/label building that masks the prompt tokens (labels=-100) so only the answer is learned.
- Optional 8-bit quantization with bitsandbytes (CPU offload), or standard fp16 training.
- CLI flags for ES connection, LoRA hyperparams, eval/logging/seed, etc.
- Proper tokenizer pad token setup.
- Graceful HF token handling.

Example:
    python finetune_from_es_lora.py \
        --es_url https://localhost:9200 \
        --es_index learning-score \
        --es_user elastic \
        --es_password Aetoa1Tahyor \
        --model_name lmsys/vicuna-3b-v1.5 \
        --output_dir ./fine-tuned-model \
        --batch_size 1 \
        --epochs 1 \
        --learning_rate 5e-5 \
        --max_length 256 \
        --quantization

Environment:
    HF_AUTH_TOKEN   # if model is gated on HF Hub
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model
from elasticsearch import Elasticsearch

warnings.filterwarnings("ignore")  # suppress SSL and other noisy warnings

# --------------------------------------------------------------------------------------
# DATASET
# --------------------------------------------------------------------------------------


class QADataset(Dataset):
    """Dataset that reads question/answer/score records from Elasticsearch.

    Expected ES doc structure in index:
        {
            "question": "...",
            "answer": "...",
            "score": 5
        }
    """

    def __init__(
        self,
        es_client: Elasticsearch,
        index_name: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
    ) -> None:
        self.es_client = es_client
        self.index_name = index_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data: List[Dict[str, str]] = list(self._fetch_all())

    def _fetch_all(self) -> Iterator[Dict[str, str]]:
        """Use scroll API to safely pull all docs from the index."""
        page_size = 1000
        query = {"query": {"match_all": {}}}
        resp = self.es_client.search(index=self.index_name, body=query, size=page_size, scroll="2m")
        scroll_id = resp.get("_scroll_id")
        hits = resp.get("hits", {}).get("hits", [])
        while hits:
            for h in hits:
                yield h.get("_source", {})
            resp = self.es_client.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = resp.get("_scroll_id")
            hits = resp.get("hits", {}).get("hits", [])
        # clear scroll context
        if scroll_id:
            try:
                self.es_client.clear_scroll(scroll_id=scroll_id)
            except Exception:  # noqa: BLE001
                pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        qa = self.data[idx]
        question: str = qa.get("question", "").strip()
        answer: str = qa.get("answer", "").strip()
        score: str = str(qa.get("score", "")).strip()

        prompt = f"Question: {question}\nAnswer (Score: {score}):"
        target = f" {answer}"  # leading space for cleaner tokenization separation

        # Tokenize prompt & target separately
        prompt_tokens = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        target_tokens = self.tokenizer(
            target,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        # Concatenate
        input_ids = torch.cat([prompt_tokens["input_ids"], target_tokens["input_ids"]], dim=1).squeeze(0)
        attention_mask = torch.cat([prompt_tokens["attention_mask"], target_tokens["attention_mask"]], dim=1).squeeze(0)

        labels = input_ids.clone()
        # ignore prompt tokens in loss
        prompt_len = prompt_tokens["input_ids"].size(1)
        labels[:prompt_len] = -100

        # pad/crop to max_length
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[: self.max_length]
            attention_mask = attention_mask[: self.max_length]
            labels = labels[: self.max_length]
        elif input_ids.size(0) < self.max_length:
            pad_len = self.max_length - input_ids.size(0)
            pad_id = self.tokenizer.pad_token_id
            input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# --------------------------------------------------------------------------------------
# ARGUMENTS
# --------------------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a model on Q/A data (with scores) from Elasticsearch.")

    # Model / training args
    parser.add_argument("--model_name", type=str, default="lmsys/vicuna-3b-v1.5", help="HF model name or local path.")
    parser.add_argument("--output_dir", type=str, default="./fine-tuned-model", help="Where to save outputs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X steps.")
    parser.add_argument("--eval_steps", type=int, default=0, help="Eval every X steps (0 disables).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Quantization / optimization
    parser.add_argument("--quantization", action="store_true", help="Use 8-bit quantization (bnb) with CPU offload.")

    # LoRA params
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout.")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,v_proj",
        help="Comma-separated list of modules to inject LoRA into.",
    )

    # Elasticsearch params
    parser.add_argument("--es_url", type=str, default="https://localhost:9200", help="Elasticsearch URL.")
    parser.add_argument("--es_user", type=str, default="elastic", help="Elasticsearch username.")
    parser.add_argument("--es_password", type=str, default="changeme", help="Elasticsearch password.")
    parser.add_argument("--es_index", type=str, default="learning-score", help="Index with Q/A documents.")

    return parser.parse_args()


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # GPU check
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU not detected. Ensure CUDA is installed and visible.")

    # HF token (optional/gated models)
    token = os.getenv("HF_AUTH_TOKEN")
    use_auth = {"use_auth_token": token} if token else {}

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **use_auth)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model loading
    if args.quantization:
        print("Using 8-bit quantization with bitsandbytes and CPU offload.")
        quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quant_config,
            device_map={"": "cpu"},
            **use_auth,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **use_auth,
        )

    # Enable gradient checkpointing & req grads
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # LoRA config
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

    # Elasticsearch client
    es_client = Elasticsearch(
        args.es_url,
        basic_auth=(args.es_user, args.es_password),
        verify_certs=False,
        ssl_show_warn=False,
    )

    # Dataset
    dataset = QADataset(es_client, args.es_index, tokenizer, max_length=args.max_length)

    # Training arguments
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
        optim="paged_adamw_8bit" if args.quantization else "adamw_torch",
        report_to=["none"],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

    trainer.train()

    # Save model + tokenizer
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Model successfully saved to {args.output_dir}")


if __name__ == "__main__":
    main()