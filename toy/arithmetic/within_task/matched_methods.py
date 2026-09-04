"""Matched comparison of six ways to learn permutation composition."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from core_task.data import (
    build_experiment_data,
    connected_cycle_edges,
    sample_primitive_library,
    validate_experiment_data,
)
from core_task.transformer import PreLayerNormBlock


METHODS = (
    "regular_sft",
    "multipass_z",
    "z_supervision",
    "iit",
    "full_scaffold",
    "staged_removal",
)
CHECKPOINTS = (0, 100, 300, 1000, 2000, 4000, 6000, 9000, 12000)
STAGES = (
    ("graph_hard", 1500, "graph", "hard", True),
    ("dense_hard", 1500, "dense", "hard", True),
    ("dense_soft", 1000, "dense", "soft", True),
    ("dense_blend", 3000, "dense", "blend", True),
    ("dense_raw_lz", 2000, "dense", "raw", True),
    ("dense_raw_ly", 3000, "dense", "raw", False),
)
Z_POSITION = 1
Z_LAYER = 2


class Vocabulary:
    def __init__(self, m=8, n=8, q=16):
        self.m, self.n, self.q = int(m), int(n), int(q)
        self.a_offset = 0
        self.b_offset = self.a_offset + self.m
        self.value_offset = self.b_offset + self.n
        self.null_token = self.value_offset + self.q
        self.out_token = self.null_token + 1
        self.size = self.out_token + 1

    def composite(self, example):
        return [
            self.a_offset + int(example.a_index),
            self.value_offset + int(example.x),
            self.b_offset + int(example.b_index),
            self.out_token,
        ]

    def atomic_a(self, a_index, x):
        return [
            self.a_offset + int(a_index), self.value_offset + int(x),
            self.null_token, self.out_token,
        ]

    def atomic_b(self, b_index, z):
        return [
            self.null_token, self.value_offset + int(z),
            self.b_offset + int(b_index), self.out_token,
        ]


class EpochBatcher:
    def __init__(self, examples, batch_size, seed):
        self.examples = tuple(examples)
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(int(seed))
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0

    def next(self):
        result = []
        while len(result) < self.batch_size:
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(len(self.examples))
                self.cursor = 0
            count = min(self.batch_size - len(result), len(self.order) - self.cursor)
            result.extend(self.examples[i] for i in self.order[self.cursor:self.cursor + count])
            self.cursor += count
        return result


def causal_mask(length, device=None):
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), 1)


def graph_mask(phase, device=None):
    if phase == "pre":
        allowed = ((0,), (0, 1), (2,), (2, 3))
    elif phase == "post":
        allowed = ((0,), (1,), (1, 2), (1, 2, 3))
    else:
        raise ValueError("phase must be pre or post")
    mask = torch.ones(4, 4, dtype=torch.bool, device=device)
    for query, keys in enumerate(allowed):
        mask[query, list(keys)] = False
    return mask


def graph_has_no_bypass(layers=4):
    masks = [graph_mask("pre")] * Z_LAYER + [graph_mask("post")] * (layers - Z_LAYER)
    edges = {}
    for layer, mask in enumerate(masks):
        for source in range(4):
            edges.setdefault((layer, source), []).append((layer + 1, source))
        for query in range(4):
            for key in range(4):
                if not bool(mask[query, key]):
                    edges.setdefault((layer, key), []).append((layer + 1, query))
    blocked = (Z_LAYER, Z_POSITION)
    for source in (0, 1):
        queue, seen = [(0, source)], {(0, source)}
        while queue:
            node = queue.pop(0)
            if node == (layers, 3):
                return False
            for neighbor in edges.get(node, ()):
                if neighbor != blocked and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    return True


class PermutationTransformer(torch.nn.Module):
    """One architecture supporting dense, routed, and canonical interfaces."""

    def __init__(self, vocabulary_size, q=16, width=64, heads=4, mlp_width=128):
        super().__init__()
        self.width = int(width)
        self.q = int(q)
        self.token_embeddings = torch.nn.Embedding(vocabulary_size, width)
        self.position_embeddings = torch.nn.Parameter(torch.empty(4, width))
        self.blocks = torch.nn.ModuleList([
            PreLayerNormBlock(width, heads, mlp_width) for _ in range(4)
        ])
        self.z_norm = torch.nn.LayerNorm(width)
        self.z_head = torch.nn.Linear(width, q)
        self.final_norm = torch.nn.LayerNorm(width)
        self.output_head = torch.nn.Linear(width, q)
        self.register_buffer("canonical_bank", torch.empty(0, width), persistent=True)
        self.routing = "dense"
        self.interface = "raw"
        self.blend_alpha = 0.0
        torch.nn.init.normal_(self.token_embeddings.weight, 0.0, 0.02)
        torch.nn.init.normal_(self.position_embeddings, 0.0, 0.02)

    def set_stage(self, routing="dense", interface="raw", blend_alpha=0.0):
        if routing not in ("dense", "graph"):
            raise ValueError("routing must be dense or graph")
        if interface not in ("raw", "hard", "soft", "blend"):
            raise ValueError("unknown interface")
        if interface != "raw" and tuple(self.canonical_bank.shape) != (self.q, self.width):
            raise ValueError("canonical interface requires a complete state bank")
        if not 0.0 <= float(blend_alpha) <= 1.0:
            raise ValueError("blend_alpha must be in [0, 1]")
        self.routing, self.interface = routing, interface
        self.blend_alpha = float(blend_alpha)

    def set_canonical_bank(self, bank):
        if tuple(bank.shape) != (self.q, self.width):
            raise ValueError("wrong canonical bank shape")
        self.canonical_bank = bank.detach().clone()

    def forward(self, tokens, patch=None, return_z=False):
        residual = self.token_embeddings(tokens) + self.position_embeddings.unsqueeze(0)
        dense = causal_mask(4, tokens.device)
        pre, post = graph_mask("pre", tokens.device), graph_mask("post", tokens.device)
        z_logits = boundary = None
        for layer, block in enumerate(self.blocks):
            if layer == Z_LAYER:
                raw = residual[:, Z_POSITION, :]
                z_logits = self.z_head(self.z_norm(raw))
                probabilities = z_logits.softmax(-1)
                boundary = raw
                if self.interface == "hard":
                    one_hot = F.one_hot(probabilities.argmax(-1), self.q).to(probabilities.dtype)
                    one_hot = one_hot + probabilities - probabilities.detach()
                    boundary = one_hot @ self.canonical_bank
                elif self.interface == "soft":
                    boundary = probabilities @ self.canonical_bank
                elif self.interface == "blend":
                    canonical = probabilities @ self.canonical_bank
                    boundary = self.blend_alpha * canonical + (1.0 - self.blend_alpha) * raw
                if patch is not None:
                    boundary = patch
                residual = residual.clone()
                residual[:, Z_POSITION, :] = boundary
            mask = dense if self.routing == "dense" else (pre if layer < Z_LAYER else post)
            residual = block(residual, mask)
        logits = self.output_head(self.final_norm(residual[:, -1, :]))
        return (logits, z_logits, boundary) if return_z else logits


def atomic_records(library, vocabulary):
    records = []
    for a_index, permutation in enumerate(library.a_permutations):
        records.extend((vocabulary.atomic_a(a_index, x), int(permutation[x])) for x in range(library.q))
    for b_index, permutation in enumerate(library.b_permutations):
        records.extend((vocabulary.atomic_b(b_index, z), int(permutation[z])) for z in range(library.q))
    return tuple(records)


def composite_batch(examples, library, vocabulary):
    return (
        torch.tensor([vocabulary.composite(e) for e in examples]),
        torch.tensor([int(e.y) for e in examples]),
        torch.tensor([int(library.a_permutations[e.a_index][e.x]) for e in examples]),
    )


def train_atomic(model, records, updates, seed, learning_rate=3e-4):
    batcher = EpochBatcher(records, 64, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.set_stage()
    for _ in range(int(updates)):
        current = batcher.next()
        tokens = torch.tensor([row[0] for row in current])
        targets = torch.tensor([row[1] for row in current])
        optimizer.zero_grad()
        loss = F.cross_entropy(model(tokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def fit_z_probe(model, library, vocabulary, updates, seed):
    records = tuple(
        (vocabulary.atomic_a(a, x), int(permutation[x]))
        for a, permutation in enumerate(library.a_permutations)
        for x in range(library.q)
    )
    batcher = EpochBatcher(records, 64, seed)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.z_norm.parameters()) + list(model.z_head.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    for _ in range(int(updates)):
        current = batcher.next()
        tokens = torch.tensor([row[0] for row in current])
        targets = torch.tensor([row[1] for row in current])
        optimizer.zero_grad()
        _, z_logits, _ = model(tokens, return_z=True)
        F.cross_entropy(z_logits, targets).backward()
        optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def make_canonical_bank(model, vocabulary):
    tokens = torch.tensor([vocabulary.atomic_b(0, z) for z in range(model.q)])
    model.eval()
    with torch.no_grad():
        _, _, states = model(tokens, return_z=True)
    return states.detach()


def accuracy(model, examples, library, vocabulary):
    tokens, targets, z_targets = composite_batch(examples, library, vocabulary)
    model.eval()
    with torch.no_grad():
        logits, z_logits, _ = model(tokens, return_z=True)
    return {
        "accuracy": float((logits.argmax(-1) == targets).float().mean()),
        "z_accuracy": float((z_logits.argmax(-1) == z_targets).float().mean()),
    }


def atomic_accuracy(model, records):
    model.eval()
    old = (model.routing, model.interface, model.blend_alpha)
    model.set_stage()
    tokens = torch.tensor([row[0] for row in records])
    targets = torch.tensor([row[1] for row in records])
    with torch.no_grad():
        value = float((model(tokens).argmax(-1) == targets).float().mean())
    model.set_stage(*old)
    return value


def sample_pairs(examples, library, count, seed):
    rng = np.random.RandomState(int(seed))
    result = []
    while len(result) < int(count):
        source = examples[int(rng.randint(len(examples)))]
        base = examples[int(rng.randint(len(examples)))]
        z_source = int(library.a_permutations[source.a_index][source.x])
        z_base = int(library.a_permutations[base.a_index][base.x])
        target = int(library.b_permutations[base.b_index][z_source])
        if z_source != z_base and target != base.y:
            result.append((source, base, target))
    return tuple(result)


def patch_accuracy(model, pairs, library, vocabulary):
    source_tokens, _, _ = composite_batch([row[0] for row in pairs], library, vocabulary)
    base_tokens, _, _ = composite_batch([row[1] for row in pairs], library, vocabulary)
    targets = torch.tensor([row[2] for row in pairs])
    model.eval()
    with torch.no_grad():
        _, _, source_states = model(source_tokens, return_z=True)
        predictions = model(base_tokens, patch=source_states).argmax(-1)
    return float((predictions == targets).float().mean())


def multipass_accuracy(model, examples, library, vocabulary):
    a_tokens = torch.tensor([vocabulary.atomic_a(e.a_index, e.x) for e in examples])
    true_z = torch.tensor([int(library.a_permutations[e.a_index][e.x]) for e in examples])
    model.eval()
    with torch.no_grad():
        predicted_z = model(a_tokens).argmax(-1)
        b_tokens = torch.tensor([
            vocabulary.atomic_b(e.b_index, int(z)) for e, z in zip(examples, predicted_z)
        ])
        predictions = model(b_tokens).argmax(-1)
    targets = torch.tensor([int(e.y) for e in examples])
    return {
        "accuracy": float((predictions == targets).float().mean()),
        "z_accuracy": float((predicted_z == true_z).float().mean()),
    }


def multipass_patch_accuracy(model, pairs, library, vocabulary):
    a_tokens = torch.tensor([vocabulary.atomic_a(row[0].a_index, row[0].x) for row in pairs])
    model.eval()
    with torch.no_grad():
        source_z = model(a_tokens).argmax(-1)
        b_tokens = torch.tensor([
            vocabulary.atomic_b(row[1].b_index, int(z)) for row, z in zip(pairs, source_z)
        ])
        predictions = model(b_tokens).argmax(-1)
    targets = torch.tensor([row[2] for row in pairs])
    return float((predictions == targets).float().mean())


def multipass_record(model, update, sets, pairs, library, vocabulary):
    observed = multipass_accuracy(model, sets[0], library, vocabulary)
    heldout = multipass_accuracy(model, sets[1], library, vocabulary)
    return {
        "update": int(update),
        "phase": "multipass_z",
        "observed": observed,
        "heldout": heldout,
        "patch_accuracy": multipass_patch_accuracy(
            model, pairs, library, vocabulary
        ),
    }


def train_multipass(model, examples, library, vocabulary, updates, seed,
                    checkpoints, sets, pairs):
    batcher = EpochBatcher(examples, 64, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95)
    )
    history = [multipass_record(
        model, 0, sets, pairs, library, vocabulary
    )]
    for update in range(int(updates) + 1):
        if update == updates:
            break
        current = batcher.next()
        a_tokens = torch.tensor([vocabulary.atomic_a(e.a_index, e.x) for e in current])
        z_targets = torch.tensor([int(library.a_permutations[e.a_index][e.x]) for e in current])
        optimizer.zero_grad()
        z_logits = model(a_tokens)
        predicted_z = z_logits.argmax(-1)
        b_tokens = torch.tensor([
            vocabulary.atomic_b(e.b_index, int(z)) for e, z in zip(current, predicted_z)
        ])
        y_targets = torch.tensor([int(e.y) for e in current])
        loss = F.cross_entropy(z_logits, z_targets) + F.cross_entropy(model(b_tokens), y_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update + 1 in checkpoints:
            history.append(multipass_record(
                model, update + 1, sets, pairs, library, vocabulary
            ))
    return history


def train_one_pass(model, method, examples, library, vocabulary, updates, seed, pairs, evaluation_sets):
    if method == "staged_removal" and int(updates) != sum(stage[1] for stage in STAGES):
        raise ValueError("finetune_updates must equal the staged schedule")
    batcher = EpochBatcher(examples, 64, seed)
    source_batcher = EpochBatcher(examples, 64, seed + 100003)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, betas=(0.9, 0.95)
    )
    history = []
    phases = STAGES if method == "staged_removal" else ((
        method, updates,
        "graph" if method == "full_scaffold" else "dense",
        "hard" if method == "full_scaffold" else "raw",
        method != "regular_sft",
    ),)
    model.set_stage(phases[0][2], phases[0][3])
    history.append(evaluate_record(
        model, 0, evaluation_sets, library, vocabulary, pairs, phases[0][0]
    ))
    completed = 0
    for phase, phase_updates, routing, interface, use_lz in phases:
        for offset in range(int(phase_updates)):
            alpha = 1.0 - float(offset + 1) / phase_updates if interface == "blend" else 0.0
            model.set_stage(routing, interface, alpha)
            current = batcher.next()
            tokens, targets, z_targets = composite_batch(current, library, vocabulary)
            optimizer.zero_grad()
            logits, z_logits, _ = model(tokens, return_z=True)
            loss = F.cross_entropy(logits, targets)
            if use_lz:
                loss = loss + F.cross_entropy(z_logits, z_targets)
            if method == "iit":
                sources = source_batcher.next()
                source_tokens, _, _ = composite_batch(sources, library, vocabulary)
                _, _, source_states = model(source_tokens, return_z=True)
                crossed = torch.tensor([
                    int(library.b_permutations[base.b_index][
                        library.a_permutations[source.a_index][source.x]
                    ]) for source, base in zip(sources, current)
                ])
                loss = loss + F.cross_entropy(model(tokens, patch=source_states), crossed)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed += 1
            if completed in CHECKPOINTS:
                history.append(evaluate_record(model, completed, evaluation_sets, library, vocabulary, pairs, phase))
        history.append(evaluate_record(model, completed, evaluation_sets, library, vocabulary, pairs, phase))
    if method == "staged_removal":
        model.set_stage()
    return history


def evaluate_record(model, update, sets, library, vocabulary, pairs, phase):
    return {
        "update": int(update),
        "phase": phase,
        "observed": accuracy(model, sets[0], library, vocabulary),
        "heldout": accuracy(model, sets[1], library, vocabulary),
        "patch_accuracy": patch_accuracy(model, pairs, library, vocabulary),
    }


def run_seed(primitive_seed, init_seed, atomic_updates=3000, probe_updates=1000,
             finetune_updates=12000, methods=METHODS):
    torch.manual_seed(int(init_seed))
    library = sample_primitive_library(primitive_seed)
    data = build_experiment_data(library, connected_cycle_edges(library.m), "connected_cycle")
    integrity = validate_experiment_data(data)
    vocabulary = Vocabulary(library.m, library.n, library.q)
    records = atomic_records(library, vocabulary)
    base = PermutationTransformer(vocabulary.size, library.q)
    train_atomic(base, records, atomic_updates, init_seed + 1)
    fit_z_probe(base, library, vocabulary, probe_updates, init_seed + 2)
    base.set_canonical_bank(make_canonical_bank(base, vocabulary))
    pairs = sample_pairs(data.heldout_examples, library, 512, init_seed + 3)
    sets = (data.observed_examples, data.heldout_examples)
    results = {}
    for method in methods:
        if method not in METHODS:
            raise ValueError("unknown method: {}".format(method))
        model = copy.deepcopy(base)
        if method == "multipass_z":
            checkpoints = set(CHECKPOINTS)
            history = train_multipass(
                model, data.observed_examples, library, vocabulary,
                finetune_updates, init_seed + 10, checkpoints, sets, pairs,
            )
            observed = multipass_accuracy(model, data.observed_examples, library, vocabulary)
            heldout = multipass_accuracy(model, data.heldout_examples, library, vocabulary)
            patch = multipass_patch_accuracy(model, pairs, library, vocabulary)
        else:
            history = train_one_pass(
                model, method, data.observed_examples, library, vocabulary,
                finetune_updates, init_seed + 10, pairs, sets,
            )
            observed = accuracy(model, data.observed_examples, library, vocabulary)
            heldout = accuracy(model, data.heldout_examples, library, vocabulary)
            patch = patch_accuracy(model, pairs, library, vocabulary)
        results[method] = {
            "observed": observed,
            "heldout": heldout,
            "patch_accuracy": patch,
            "atomic_accuracy": atomic_accuracy(model, records),
            "history": history,
            "final_inference": (
                "two_forward_passes" if method == "multipass_z"
                else "one_pass_dense_raw_without_auxiliary_loss"
                if method == "staged_removal" else "one_forward_pass"
            ),
        }
    return {
        "primitive_seed": primitive_seed,
        "init_seed": init_seed,
        "base_atomic_accuracy": atomic_accuracy(base, records),
        "integrity": integrity,
        "results": results,
    }


def run(output, seeds=(51001, 51002, 51003), methods=METHODS, **kwargs):
    torch.set_num_threads(1)
    if not graph_has_no_bypass():
        raise AssertionError("graph mask permits an input-to-output bypass")
    runs = [run_seed(31001 + index, seed, methods=methods, **kwargs)
            for index, seed in enumerate(seeds)]
    summary = {
        "task": "held-out composition y = B(A(x))",
        "methods": list(methods),
        "matched_controls": {
            "same_architecture": True,
            "same_atomic_checkpoint_within_seed": True,
            "same_256_composite_training_examples": True,
            "same_768_heldout_examples": True,
            "same_update_budget": int(kwargs.get("finetune_updates", 12000)),
        },
        "staged_schedule": [list(stage) for stage in STAGES],
        "runs": runs,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="toy/arithmetic/within_task/results/matched_methods")
    parser.add_argument("--seeds", type=int, nargs="+", default=[51001, 51002, 51003])
    parser.add_argument("--atomic-updates", type=int, default=3000)
    parser.add_argument("--probe-updates", type=int, default=1000)
    parser.add_argument("--finetune-updates", type=int, default=12000)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    run(args.output, args.seeds, methods=args.methods,
        atomic_updates=args.atomic_updates,
        probe_updates=args.probe_updates, finetune_updates=args.finetune_updates)


if __name__ == "__main__":
    main()
