"""Route each prompt to a base or refusal adapter through a predicted safety Z."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from causal.evaluate_no_sft import summarize, write_jsonl


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def harmful_probability(joint_logits, z_classes):
    """Return probability mass assigned to joint-Z classes marked harmful."""
    probabilities = joint_logits.float().softmax(-1)
    mask = torch.tensor(
        [item["intent"] == "harmful" for item in z_classes],
        device=probabilities.device,
    )
    if not mask.any() or mask.all():
        raise ValueError("joint-Z vocabulary needs benign and harmful classes")
    return probabilities[:, mask].sum(-1)


def route_records(base_rows, refusal_rows, probabilities, threshold):
    """Select an already judged deterministic response for each gate decision."""
    refusal_by_id = {row["id"]: row for row in refusal_rows}
    if len(refusal_by_id) != len(refusal_rows):
        raise ValueError("duplicate IDs in refusal results")
    if len(base_rows) != len(probabilities):
        raise ValueError("one gate probability is required per record")

    routed = []
    for base, probability in zip(base_rows, probabilities):
        if base["id"] not in refusal_by_id:
            raise ValueError(f"missing refusal result for {base['id']}")
        use_refusal = float(probability) >= float(threshold)
        selected = dict(refusal_by_id[base["id"]] if use_refusal else base)
        selected.update({
            "gate_harmful_probability": float(probability),
            "gate_policy": "refuse" if use_refusal else "comply",
            "gate_branch": "refusal_adapter" if use_refusal else "base_model",
        })
        routed.append(selected)
    return routed


def gate_metrics(rows):
    harmful = [row for row in rows if row["kind"] == "harmful"]
    benign = [row for row in rows if row["kind"] == "benign"]
    return {
        "harmful_gate_recall": sum(
            row["gate_policy"] == "refuse" for row in harmful
        ) / len(harmful),
        "benign_gate_specificity": sum(
            row["gate_policy"] == "comply" for row in benign
        ) / len(benign),
        "gate_accuracy": sum(
            (row["gate_policy"] == "refuse") == (row["kind"] == "harmful")
            for row in rows
        ) / len(rows),
    }


def classify_prompts(rows, args, state, config):
    tokenizer = AutoTokenizer.from_pretrained(args.controller_adapter)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(
        model, args.controller_adapter, is_trainable=False
    ).eval()
    weights = state["joint_z_head"]["weight"]
    head = torch.nn.Linear(weights.shape[1], weights.shape[0]).to(
        device=model.device, dtype=torch.float32
    )
    head.load_state_dict(state["joint_z_head"])
    head.eval()
    z_classes = config["z_classes"]
    probabilities = []
    captured = []
    decoder = model.base_model.model.model
    layer = decoder.layers[int(config["z_boundary"]) - 1]

    def capture_z(unused_module, unused_args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured.append(hidden[:, -1, :].detach())

    handle = layer.register_forward_hook(capture_z)

    for offset in range(0, len(rows), args.batch_size):
        batch = rows[offset:offset + args.batch_size]
        prompts = [tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        ) for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=3584,
        ).to(model.device)
        with torch.inference_mode():
            decoder(
                **encoded,
                use_cache=False,
                return_dict=True,
            )
            if len(captured) != 1:
                raise RuntimeError("expected exactly one captured Z state")
            z_state = captured.pop()
            probabilities.extend(
                harmful_probability(head(z_state.float()), z_classes).cpu().tolist()
            )
        print(
            f"classified {min(offset + len(batch), len(rows))}/{len(rows)}",
            flush=True,
        )

    handle.remove()
    del head, model
    gc.collect()
    torch.cuda.empty_cache()
    return probabilities


def write_condition(output, name, rows):
    destination = output / name
    destination.mkdir(parents=True, exist_ok=False)
    write_jsonl(destination / "routed_judged.jsonl", rows)
    result = summarize(rows)
    result["gate"] = gate_metrics(rows)
    (destination / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--controller-adapter", required=True)
    parser.add_argument("--controller-state", required=True)
    parser.add_argument("--controller-config", required=True)
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--refusal-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_rows = load_jsonl(args.base_results)
    refusal_rows = load_jsonl(args.refusal_results)
    base_ids = [row["id"] for row in base_rows]
    refusal_ids = [row["id"] for row in refusal_rows]
    if base_ids != refusal_ids:
        raise ValueError("base and refusal results must have identical ordered IDs")
    config = json.loads(Path(args.controller_config).read_text())
    state = torch.load(args.controller_state, map_location="cpu", weights_only=True)
    if len(config["z_classes"]) != state["joint_z_head"]["weight"].shape[0]:
        raise ValueError("joint-Z head and vocabulary sizes differ")
    print(json.dumps({
        "records": len(base_rows),
        "harmful": sum(row["kind"] == "harmful" for row in base_rows),
        "benign": sum(row["kind"] == "benign" for row in base_rows),
        "joint_z_classes": len(config["z_classes"]),
        "reuses_existing_deterministic_generations": True,
    }, indent=2), flush=True)
    if args.dry_run:
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    probabilities = classify_prompts(base_rows, args, state, config)
    classification_rows = [
        {
            "id": row["id"],
            "condition": row["condition"],
            "kind": row["kind"],
            "harmful_probability": float(probability),
        }
        for row, probability in zip(base_rows, probabilities)
    ]
    write_jsonl(output / "classifications.jsonl", classification_rows)

    results = {}
    for threshold in [float(value) for value in args.thresholds.split(",")]:
        routed = route_records(base_rows, refusal_rows, probabilities, threshold)
        name = f"threshold_{threshold:.2f}"
        results[name] = write_condition(output, name, routed)

    oracle_probabilities = [
        1.0 if row["kind"] == "harmful" else 0.0 for row in base_rows
    ]
    oracle = route_records(base_rows, refusal_rows, oracle_probabilities, 0.5)
    results["oracle_gate"] = write_condition(output, "oracle_gate", oracle)
    (output / "threshold_sweep.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
