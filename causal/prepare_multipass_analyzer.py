"""Build balanced textual-Z training data for the multipass safety analyzer."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.srg_experiment.causal_scaffold import HARM_TYPES
from src.srg_experiment.surface_invariance import surface_variant


def canonical_z(record):
    harms = ",".join(
        name for name in HARM_TYPES if name in record["harm_types"]
    ) or "none"
    return (
        f"<intent>{record['intent']}</intent>\n"
        f"<harm_types>{harms}</harm_types>\n"
        f"<severity>{int(record['harm_severity'])}</severity>\n"
        f"<actionability>{int(record['actionability'])}</actionability>\n"
        f"<policy>{record['policy']}</policy>"
    )


def styled_prompt(prompt, style):
    return prompt if style == 0 else surface_variant(prompt, style - 1)


def build(records, seed):
    harmful = [row for row in records if row["kind"] == "harmful"]
    benign = [row for row in records if row["kind"] == "benign"]
    if not harmful or len(benign) % len(harmful):
        raise ValueError("benign/harmful counts must have an integer ratio")
    repeats = len(benign) // len(harmful)
    output = []
    for row in benign:
        style = len(output) % 5
        output.append({
            "id": f"analyzer-{row['id']}-0",
            "prompt": styled_prompt(row["prompt"], style),
            "response": canonical_z(row),
            "source_id": row["id"],
            "kind": row["kind"],
            "surface_style": style,
        })
    for row in harmful:
        for repeat in range(repeats):
            style = repeat % 5
            output.append({
                "id": f"analyzer-{row['id']}-{repeat}",
                "prompt": styled_prompt(row["prompt"], style),
                "response": canonical_z(row),
                "source_id": row["id"],
                "kind": row["kind"],
                "surface_style": style,
            })
    random.Random(seed).shuffle(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    output = build(records, args.seed)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "records": len(output),
        "harmful": sum(row["kind"] == "harmful" for row in output),
        "benign": sum(row["kind"] == "benign" for row in output),
    }, indent=2))


if __name__ == "__main__":
    main()
