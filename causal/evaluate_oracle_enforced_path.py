"""Evaluate a fully enforced oracle safety path using a fixed refusal branch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal.evaluate_no_sft import judge, summarize, write_jsonl


REFUSAL = "I cannot assist with that request."


def enforce_oracle_path(rows: list[dict]) -> list[dict]:
    routed = []
    for source in rows:
        row = dict(source)
        harmful = row["kind"] == "harmful"
        row.update({
            "oracle_z": "harmful" if harmful else "benign",
            "policy": "refuse" if harmful else "comply",
            "response": REFUSAL if harmful else row["response"],
            "response_branch": "fixed_refusal" if harmful else "base_model",
        })
        routed.append(row)
    return routed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--judge-batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines()]
    routed = enforce_oracle_path(rows)
    assert all(
        row["response"] == REFUSAL
        for row in routed if row["kind"] == "harmful"
    )
    assert all(
        row["response"] == source["response"]
        for row, source in zip(routed, rows) if row["kind"] == "benign"
    )
    print(json.dumps({
        "records": len(routed),
        "harmful": sum(row["kind"] == "harmful" for row in routed),
        "benign": sum(row["kind"] == "benign" for row in routed),
        "path": "X -> oracle Z -> deterministic policy -> enforced response",
    }, indent=2), flush=True)
    if args.dry_run:
        return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "responses.jsonl", routed)
    judge(routed, args.judge, args.judge_batch_size)
    write_jsonl(output / "judged.jsonl", routed)
    result = summarize(routed)
    result["provenance"] = {
        "z": "oracle harmful/benign evaluation label",
        "policy": "deterministic",
        "harmful_response": REFUSAL,
        "benign_response": args.input,
        "claim_scope": "positive control; no learned X-to-Z path",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
