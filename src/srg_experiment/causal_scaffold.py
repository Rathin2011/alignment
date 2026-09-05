"""Fixed-slot data and graph routing for causal safety fine-tuning."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset


PAD = -1
SOURCE, INTENT, HARM, SEVERITY, ACTIONABILITY, POLICY, ANSWER = range(7)
HARM_TYPES = (
    "violence", "self_harm", "sexual", "privacy", "cyber", "fraud",
    "illegal_goods", "hate_harassment", "misinformation", "manipulation",
    "other",
)
INTENT_IDS = {"benign": 0, "ambiguous": 1, "harmful": 2}
POLICY_IDS = {"comply": 0, "safe_complete": 1, "refuse": 2}


class ScaffoldDataset(Dataset):
    def __init__(self, path, tokenizer, cutoff_length=4096):
        with Path(path).open() as stream:
            self.records = [json.loads(line) for line in stream]
        self.tokenizer = tokenizer
        self.cutoff_length = int(cutoff_length)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.encode_record(self.records[index])

    def encode_record(self, record, prompt=None):
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": (
                record["prompt"] if prompt is None else prompt
            )}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        segments = [SOURCE] * len(input_ids)
        positions = []
        for text, segment in (
            (" Intent:", INTENT), (" Harm:", HARM),
            (" Severity:", SEVERITY),
            (" Actionability:", ACTIONABILITY),
            (" Policy:", POLICY), (" Answer:", ANSWER),
        ):
            piece = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids.extend(piece)
            segments.extend([segment] * len(piece))
            positions.append(len(input_ids) - 1)
        response = self.tokenizer.encode(
            record["response"] + "<|im_end|>\n", add_special_tokens=False
        )
        response_start = len(input_ids)
        input_ids.extend(response)
        segments.extend([ANSWER] * len(response))
        if len(input_ids) > self.cutoff_length:
            raise ValueError(f"record {record['id']} exceeds cutoff")
        harm_target = [float(name in record["harm_types"])
                       for name in HARM_TYPES]
        return {
            "input_ids": input_ids,
            "segments": segments,
            "labels": [-100] * response_start + response,
            "positions": positions,
            "intent_target": INTENT_IDS[record["intent"]],
            "harm_target": harm_target,
            "severity_target": int(record["harm_severity"]),
            "actionability_target": int(record["actionability"]),
            "policy_target": POLICY_IDS[record["policy"]],
        }


def allowed_attention(segments, policy_target, graph=True):
    """Return a causal allowed-attention matrix for one example."""

    segment = torch.tensor(segments, dtype=torch.long)
    query, key = segment[:, None], segment[None, :]
    index = torch.arange(len(segments))
    causal = index[None, :] <= index[:, None]
    valid = (query != PAD) & (key != PAD)
    if not graph:
        return causal & valid
    z_query = ((query == INTENT) | (query == HARM) |
               (query == SEVERITY) | (query == ACTIONABILITY))
    z_key = ((key == INTENT) | (key == HARM) |
             (key == SEVERITY) | (key == ACTIONABILITY))
    source_route = (query == SOURCE) & (key == SOURCE)
    z_route = z_query & ((key == SOURCE) | (key == query))
    policy_route = (query == POLICY) & (z_key | (key == POLICY))
    answer_keys = (key == POLICY) | (key == ANSWER)
    if policy_target == POLICY_IDS["comply"]:
        answer_keys = answer_keys | (key == SOURCE)
    else:
        answer_keys = answer_keys | z_key
    answer_route = (query == ANSWER) & answer_keys
    return causal & valid & (source_route | z_route | policy_route |
                             answer_route)


class ScaffoldCollator:
    def __init__(self, pad_token_id, graph=True, multiple=8):
        self.pad_token_id = int(pad_token_id)
        self.graph = bool(graph)
        self.multiple = int(multiple)

    def __call__(self, features):
        length = max(len(row["input_ids"]) for row in features)
        length = math.ceil(length / self.multiple) * self.multiple
        batch = {key: [] for key in (
            "input_ids", "labels", "positions", "intent_target",
            "harm_target", "severity_target", "actionability_target",
            "policy_target",
        )}
        masks, segment_rows = [], []
        for row in features:
            padding = length - len(row["input_ids"])
            segments = row["segments"] + [PAD] * padding
            segment_rows.append(segments)
            batch["input_ids"].append(
                row["input_ids"] + [self.pad_token_id] * padding
            )
            batch["labels"].append(row["labels"] + [-100] * padding)
            for key in batch:
                if key not in {"input_ids", "labels"}:
                    batch[key].append(row[key])
            masks.append(allowed_attention(
                segments, row["policy_target"], self.graph
            ))
        allowed = torch.stack(masks)[:, None]
        attention_mask = torch.zeros(allowed.shape, dtype=torch.float32)
        attention_mask.masked_fill_(~allowed, torch.finfo(torch.float32).min)
        tensors = {
            key: torch.tensor(value, dtype=(
                torch.float32 if key == "harm_target" else torch.long
            )) for key, value in batch.items()
        }
        tensors["attention_mask"] = attention_mask
        tensors["segment_ids"] = torch.tensor(segment_rows, dtype=torch.long)
        return tensors


class ScaffoldHeads(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.intent = nn.Linear(hidden_size, 3)
        self.harm = nn.Linear(hidden_size, len(HARM_TYPES))
        self.severity = nn.Linear(hidden_size, 4)
        self.actionability = nn.Linear(hidden_size, 3)
        self.policy = nn.Linear(hidden_size, 3)


def scaffold_loss(model, heads, batch, lambda_z=1.0, lambda_policy=1.0):
    labels = batch["labels"]
    decoder = model.base_model.model.model
    lm_head = model.base_model.model.lm_head
    output = decoder(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"].to(
            dtype=decoder.embed_tokens.weight.dtype
        ),
        use_cache=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state
    logits = lm_head(hidden)
    y_loss = F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1), ignore_index=-100,
    )
    positions = batch["positions"]
    row = torch.arange(hidden.shape[0], device=hidden.device)
    slot_states = [hidden[row, positions[:, index]].float()
                   for index in range(5)]
    intent_loss = F.cross_entropy(
        heads.intent(slot_states[0]), batch["intent_target"])
    harm_loss = F.binary_cross_entropy_with_logits(
        heads.harm(slot_states[1]), batch["harm_target"])
    severity_loss = F.cross_entropy(
        heads.severity(slot_states[2]), batch["severity_target"])
    actionability_loss = F.cross_entropy(
        heads.actionability(slot_states[3]),
        batch["actionability_target"],
    )
    policy_loss = F.cross_entropy(
        heads.policy(slot_states[4]), batch["policy_target"])
    z_loss = (intent_loss + harm_loss + severity_loss +
              actionability_loss) / 4
    total = y_loss + float(lambda_z) * z_loss + float(lambda_policy) * policy_loss
    return total, {
        "loss": total.detach(), "y_loss": y_loss.detach(),
        "z_loss": z_loss.detach(), "policy_loss": policy_loss.detach(),
    }
