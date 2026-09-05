"""Build a small causal-safety pilot from COCA's released annotations."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from src.srg_experiment.causal_scaffold import HARM_TYPES


HARM_PATTERNS = {
    "violence": r"\b(?:violence|violent|kill|murder|assault|bomb|weapon|terror)",
    "self_harm": r"\b(?:self[- ]?harm|suicid|cutting oneself)",
    "sexual": r"\b(?:sexual|porn|rape|non-consensual|explicit sex)",
    "privacy": r"\b(?:privacy|private data|surveillance|track(?:ing)?|recording)",
    "cyber": r"\b(?:hack|malware|phishing|ddos|password|cyber|ransomware)",
    "fraud": r"\b(?:fraud|scam|counterfeit|forgery|identity theft|cheat)",
    "illegal_goods": r"\b(?:illegal drug|narcotic|shoplift|stolen goods|firearm)",
    "hate_harassment": r"\b(?:hate|harass|bully|slur|discriminat|mocking)",
    "misinformation": r"\b(?:misinformation|disinformation|fake news|propaganda)",
    "manipulation": r"\b(?:manipulat|coerc|blackmail|extortion|deceiv)",
}


def extract_tag(text: str, tag: str) -> str:
    match = re.search(
        rf"<{tag}>(.*?)(?:</{tag}>|$)", text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return "" if match is None else match.group(1).strip()


def infer_harm_types(text: str) -> list[str]:
    lowered = text.lower()
    found = [name for name, pattern in HARM_PATTERNS.items()
             if re.search(pattern, lowered)]
    return [name for name in HARM_TYPES if name in found] or ["other"]


def infer_severity(text: str) -> int:
    if re.search(r"\b(?:kill|murder|suicid|rape|bomb|terror|poison)", text,
                 flags=re.IGNORECASE):
        return 3
    if re.search(r"\b(?:illegal|theft|fraud|hack|weapon|drug|privacy)", text,
                 flags=re.IGNORECASE):
        return 2
    return 1


def infer_actionability(prompt: str) -> int:
    return 2 if re.search(
        r"\b(?:how (?:can|do|to)|steps?|instructions?|tips?|methods?|ways?|code)\b",
        prompt, flags=re.IGNORECASE,
    ) else 1


def _prompt(row: dict) -> str:
    extra = row.get("input", "").strip()
    return row["instruction"].strip() + (f"\n{extra}" if extra else "")


def harmful_record(row: dict, index: int) -> dict:
    prompt = _prompt(row)
    concept = extract_tag(row["output"], "concept")
    response = extract_tag(row["output"], "response")
    if not response:
        raise ValueError(f"harmful record {index} has no <response>")
    evidence = f"{prompt}\n{concept}"
    return {
        "id": f"coca-harmful-{index}", "prompt": prompt,
        "kind": "harmful", "response": response,
        "reasoning_response": row["output"].strip(),
        "intent": "harmful", "harm_types": infer_harm_types(evidence),
        "harm_severity": infer_severity(evidence),
        "actionability": infer_actionability(prompt),
        "context": "direct", "policy": "refuse", "concept": concept,
        "z_label_source": "deterministic mapping of COCA concept annotation",
    }


def benign_record(source: str, index: int, row: dict) -> dict:
    prompt = _prompt(row)
    response = extract_tag(row["output"], "response")
    if not response:
        raise ValueError(f"benign record {source}:{index} has no <response>")
    return {
        "id": f"coca-benign-{source}-{index}", "prompt": prompt,
        "kind": "benign", "response": response,
        "reasoning_response": row["output"].strip(),
        "intent": "benign", "harm_types": [], "harm_severity": 0,
        "actionability": 0, "context": "direct", "policy": "comply",
        "concept": extract_tag(row["output"], "concept"),
        "z_label_source": "COCA benign training split",
    }


def build_records(data_dir: str | Path, harmful_count: int,
                  benign_count: int, seed: int) -> list[dict]:
    root = Path(data_dir)
    harmful_rows = json.loads((root / "beaver_think.json").read_text())
    benign_rows = []
    for path in sorted(root.glob("wizardlm_think_*.json")):
        for index, row in enumerate(json.loads(path.read_text())):
            benign_rows.append((path.stem, index, row))
    if harmful_count > len(harmful_rows) or benign_count > len(benign_rows):
        raise ValueError("requested sample exceeds released COCA data")
    rng = random.Random(seed)
    harmful_indices = rng.sample(range(len(harmful_rows)), harmful_count)
    benign_indices = rng.sample(range(len(benign_rows)), benign_count)
    records = [harmful_record(harmful_rows[index], index)
               for index in harmful_indices]
    records.extend(benign_record(*benign_rows[index])
                   for index in benign_indices)
    rng.shuffle(records)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--harmful-count", type=int, default=500)
    parser.add_argument("--benign-count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()
    records = build_records(
        args.data_dir, args.harmful_count, args.benign_count, args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output), "records": len(records),
        "harmful": sum(row["kind"] == "harmful" for row in records),
        "benign": sum(row["kind"] == "benign" for row in records),
        "seed": args.seed,
    }, indent=2))


if __name__ == "__main__":
    main()
