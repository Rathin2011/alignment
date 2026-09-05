"""Build joint X->Z and Z->decision training data for one Qwen adapter."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


ANALYZE = "Analyze the request. Output only the structured safety state.\n\nRequest:\n{}"
DECIDE = (
    "Decide using only the structured safety state below. "
    "Output exactly REFUSE or COMPLY.\n\nSafety state:\n{}"
)


def without_policy(z_text: str) -> str:
    return re.sub(
        r"\s*<policy>.*?</policy>\s*", "\n", z_text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def build(records: list[dict], seed: int) -> list[dict]:
    output = []
    for row in records:
        z_text = without_policy(row["response"])
        output.extend((
            {
                "id": f"analyze-{row['id']}",
                "prompt": ANALYZE.format(row["prompt"]),
                "response": z_text,
                "stage": "x_to_z",
                "kind": row["kind"],
            },
            {
                "id": f"decide-{row['id']}",
                "prompt": DECIDE.format(z_text),
                "response": "REFUSE" if row["kind"] == "harmful" else "COMPLY",
                "stage": "z_to_decision",
                "kind": row["kind"],
            },
        ))
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
        "x_to_z": sum(row["stage"] == "x_to_z" for row in output),
        "z_to_decision": sum(row["stage"] == "z_to_decision" for row in output),
    }, indent=2))


if __name__ == "__main__":
    main()
