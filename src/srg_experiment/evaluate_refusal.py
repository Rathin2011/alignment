"""Generate and safety-score the SRG refusal baseline on ID and OOD prompts."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_eval_records(behaviors_path, jailbreak_artifact_path):
    with Path(behaviors_path).open(newline="") as stream:
        behaviors = list(csv.DictReader(stream))
    artifact = json.loads(Path(jailbreak_artifact_path).read_text())
    template = artifact["parameters"]["method_parameters"]["prompt_template"]
    marker = "[INSERT PROMPT HERE]"
    if marker not in template:
        raise ValueError("JailbreakChat template lacks insertion marker")
    records = []
    for row in behaviors:
        seed = row["Goal"]
        records.append({
            "id": f"id-{row['Index']}", "condition": "id_jbb",
            "seed_prompt": seed, "model_prompt": seed,
        })
        records.append({
            "id": f"jbc-{row['Index']}", "condition": "jailbreakchat_aim",
            "seed_prompt": seed,
            "model_prompt": template.replace(marker, seed),
        })
    return records


def parse_guardian_label(text):
    match = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError("guardian did not return yes/no")
    return match.group(1).lower() == "yes"


def extract_final_response(text):
    match = re.search(r"<(answer|response)>(.*?)(?:</\1>|$)", text,
                      flags=re.DOTALL | re.IGNORECASE)
    return (match.group(2) if match else text).strip()


def strip_decision_marker(text):
    return re.sub(
        r"^\s*(?:refuse|comply|redirect)\b[\s:;-]*",
        "", text, count=1, flags=re.IGNORECASE,
    ).strip()


def _write(path, records):
    with Path(path).open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate(records, base_model, adapter, batch_size=8,
             max_new_tokens=512, prompt_suffix="",
             strip_policy_marker=False, model_setup=None):
    tokenizer_source = (
        adapter if adapter and (Path(adapter) / "tokenizer_config.json").exists()
        else base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    )
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model = model.eval()
    cleanup = model_setup(model) if model_setup is not None else None
    eos_ids = [tokenizer.eos_token_id,
               tokenizer.convert_tokens_to_ids("<|im_end|>")]
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["model_prompt"]}],
            tokenize=False, add_generation_prompt=True,
        ) + prompt_suffix for row in batch]
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=3584,
        ).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **encoded, do_sample=False, max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
            )
        texts = tokenizer.batch_decode(
            output[:, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        for row, text in zip(batch, texts):
            row["raw_generation"] = text.strip()
            response = extract_final_response(text)
            row["response"] = (
                strip_decision_marker(response)
                if strip_policy_marker else response
            )
        print(f"generated {min(offset + len(batch), len(records))}/"
              f"{len(records)}", flush=True)
    if cleanup is not None:
        cleanup()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def judge(records, judge_model, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(judge_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        judge_model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        texts = [tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["seed_prompt"]},
                {"role": "assistant", "content": row["response"]},
            ],
            guardian_config={"risk_name": "harm"},
            tokenize=False,
            add_generation_prompt=True,
        ) for row in batch]
        encoded = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=4096).to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **encoded, do_sample=False, max_new_tokens=5,
                pad_token_id=tokenizer.pad_token_id,
            )
        labels = tokenizer.batch_decode(
            output[:, encoded["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        for row, label in zip(batch, labels):
            row["judge_output"] = label.strip()
            try:
                row["unsafe"] = parse_guardian_label(label)
            except ValueError:
                row["unsafe"] = None
        print(f"judged {min(offset + len(batch), len(records))}/"
              f"{len(records)}", flush=True)
    return records


def summarize(records):
    summary = {}
    for condition in sorted({row["condition"] for row in records}):
        subset = [row for row in records if row["condition"] == condition]
        valid = [row for row in subset if row["unsafe"] is not None]
        summary[condition] = {
            "examples": len(subset),
            "valid_judgments": len(valid),
            "attack_success_rate": (
                sum(row["unsafe"] for row in valid) / len(valid)
                if valid else None
            ),
        }
    if all(key in summary for key in ("id_jbb", "jailbreakchat_aim")):
        summary["ood_minus_id"] = (
            summary["jailbreakchat_aim"]["attack_success_rate"] -
            summary["id_jbb"]["attack_success_rate"]
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--behaviors", required=True)
    parser.add_argument("--jailbreak-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument("--strip-policy-marker", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = build_eval_records(args.behaviors, args.jailbreak_artifact)
    records = generate(
        records, args.base_model, args.adapter,
        prompt_suffix=args.prompt_suffix,
        strip_policy_marker=args.strip_policy_marker,
    )
    _write(output / "generations.jsonl", records)
    records = judge(records, args.judge_model)
    _write(output / "judged.jsonl", records)
    summary = summarize(records)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
