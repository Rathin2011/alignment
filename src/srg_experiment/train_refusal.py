"""LoRA refusal-training baseline for the reconstructed SRG setting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


def structured_reasoning_target(record):
    harms = ", ".join(record["harm_types"]) or "none"
    return """<thinking>
<step>User intent: {intent}.</step>
<step>Potential harm categories: {harms}; severity: {severity}/3.</step>
<step>Requested harmful actionability: {actionability}/2.</step>
<step>Context: {context}.</step>
<reflection>{rationale}</reflection>
<step>Response policy: {policy}.</step>
</thinking>
<answer>{response}</answer>""".format(
        intent=record["intent"], harms=harms,
        severity=record["harm_severity"],
        actionability=record["actionability"], context=record["context"],
        rationale=record["rationale"], policy=record["policy"],
        response=record["response"],
    )


class RefusalDataset(Dataset):
    def __init__(self, path, tokenizer, cutoff_length=4096,
                 target_mode="final_response"):
        with Path(path).open() as stream:
            self.records = [json.loads(line) for line in stream]
        self.tokenizer = tokenizer
        self.cutoff_length = int(cutoff_length)
        self.target_mode = target_mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": record["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target = (record["response"] if self.target_mode == "final_response"
                  else structured_reasoning_target(record))
        response_ids = self.tokenizer.encode(target + "<|im_end|>\n",
                                             add_special_tokens=False)
        input_ids = (prompt_ids + response_ids)[:self.cutoff_length]
        prompt_length = min(len(prompt_ids), len(input_ids))
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        if not any(label != -100 for label in labels):
            raise ValueError(f"record {record['id']} has no response tokens")
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


class PadCollator:
    def __init__(self, pad_token_id, multiple=8):
        self.pad_token_id = int(pad_token_id)
        self.multiple = int(multiple)

    def __call__(self, features):
        length = max(len(row["input_ids"]) for row in features)
        length = math.ceil(length / self.multiple) * self.multiple
        batch = {key: [] for key in ("input_ids", "attention_mask", "labels")}
        for row in features:
            padding = length - len(row["input_ids"])
            batch["input_ids"].append(
                row["input_ids"] + [self.pad_token_id] * padding
            )
            batch["attention_mask"].append(
                row["attention_mask"] + [0] * padding
            )
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in batch.items()}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cutoff-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--maximum-steps", type=int, default=-1)
    parser.add_argument("--target-mode", choices=(
        "final_response", "structured_reasoning"),
        default="final_response")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.global_batch_size % args.micro_batch_size:
        raise ValueError("global batch size must divide by micro batch size")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = RefusalDataset(args.data, tokenizer, args.cutoff_length,
                             args.target_mode)
    lengths = [len(dataset[index]["input_ids"])
               for index in range(len(dataset))]
    print(json.dumps({
        "records": len(dataset),
        "minimum_tokens": min(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "maximum_tokens": max(lengths),
        "truncated": sum(length == args.cutoff_length for length in lengths),
        "data_sha256": _sha256(args.data),
    }, indent=2), flush=True)
    if args.dry_run:
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ))
    model.print_trainable_parameters()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=args.epochs,
        max_steps=args.maximum_steps,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=(
            args.global_batch_size // args.micro_batch_size
        ),
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    checkpoint = get_last_checkpoint(str(output))
    result = trainer.train(resume_from_checkpoint=checkpoint)
    final = output / "final_adapter"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(str(final))
    (output / "run_metadata.json").write_text(json.dumps({
        "data_sha256": _sha256(args.data),
        "model": args.model,
        "train_metrics": result.metrics,
        "seed": args.seed,
        "arm": args.target_mode,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
