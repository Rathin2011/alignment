"""Matched comparison of six methods on toy safety OOD generalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from toy.safety.core_task.model import FixedSizeEpochBatcher, verify_graph_paths
from toy.safety.core_task.task import (
    LATENT_COUNT,
    OOD_STYLES,
    TRAIN_STYLES,
    SafetyVocabulary,
    atomic_accuracy as raw_atomic_accuracy,
    atomic_examples,
    batch,
    build_data,
    evaluate,
    policy_action,
)
from toy.safety.core_task.training import (
    build_retention_records,
    fit_z_head,
    make_model,
    new_base_model,
    policy_state_bank,
    pretrain,
    retention_loss,
)


METHODS = (
    "regular_sft",
    "multipass_z",
    "z_supervision",
    "iit",
    "full_scaffold",
    "staged_removal",
)
CHECKPOINTS = (0, 100, 300, 1000, 2000, 4000, 6000)
PHASES = (
    ("graph_hard", 1500, "graph", "hard", True),
    ("dense_hard", 750, "dense", "hard", True),
    ("dense_soft", 750, "dense", "soft", True),
    ("dense_blend", 1500, "dense", "blend", True),
    ("dense_raw_lz", 500, "dense", "raw", True),
    ("dense_raw_ly", 1000, "dense", "raw", False),
)


def atomic_accuracy(model, records):
    """Evaluate retained atomic skills under their original dense interface."""
    old = (model.routing, model.interface, model.blend_alpha)
    model.set_stage("dense", "raw")
    value = raw_atomic_accuracy(model, records)
    model.set_stage(*old)
    return value


def effective_boundary(model, tokens):
    """Return the actual state sent across the model's Z boundary."""
    _, z_logits, raw = model(tokens, return_z=True)
    probabilities = z_logits.softmax(-1)
    if model.interface == "hard":
        return model.canonical_bank[probabilities.argmax(-1)]
    if model.interface == "soft":
        return probabilities @ model.canonical_bank
    if model.interface == "blend":
        canonical = probabilities @ model.canonical_bank
        return model.blend_alpha * canonical + (1.0 - model.blend_alpha) * raw
    return raw


def counterfactual_pairs(examples):
    """Pair each base request with a source requiring another policy action."""
    return tuple(
        (
            next(
                source for source in examples
                if source.style == base.style
                and policy_action(source.z) != policy_action(base.z)
            ),
            base,
        )
        for base in examples
    )


def patch_accuracy(model, pairs, vocabulary):
    sources, bases = zip(*pairs)
    source_tokens, _, _ = batch(sources, vocabulary)
    base_tokens, _, _ = batch(bases, vocabulary)
    targets = torch.tensor([source.target for source in sources])
    model.eval()
    with torch.no_grad():
        boundary = effective_boundary(model, source_tokens)
        predictions = model(base_tokens, z_patch=boundary).argmax(-1)
    return float((predictions == targets).float().mean())


def one_pass_record(model, update, phase, train, ood, pairs, vocabulary):
    return {
        "update": int(update),
        "phase": phase,
        "observed": evaluate(model, train, vocabulary),
        "heldout": evaluate(model, ood, vocabulary),
        "patch_accuracy": patch_accuracy(model, pairs, vocabulary),
    }


def multipass_accuracy(model, examples, vocabulary):
    recognition = torch.tensor([
        vocabulary.atomic_recognition(example) for example in examples
    ])
    true_z = torch.tensor([example.z for example in examples])
    model.eval()
    with torch.no_grad():
        predicted_z = model(recognition)[:, :LATENT_COUNT].argmax(-1)
        policy = torch.tensor([
            vocabulary.atomic_policy(int(z_value)) for z_value in predicted_z
        ])
        predictions = model(policy).argmax(-1)
    targets = torch.tensor([example.target for example in examples])
    return {
        "accuracy": float((predictions == targets).float().mean()),
        "z_accuracy": float((predicted_z == true_z).float().mean()),
    }


def multipass_patch_accuracy(model, pairs, vocabulary):
    sources, _ = zip(*pairs)
    recognition = torch.tensor([
        vocabulary.atomic_recognition(source) for source in sources
    ])
    targets = torch.tensor([source.target for source in sources])
    model.eval()
    with torch.no_grad():
        routed_z = model(recognition)[:, :LATENT_COUNT].argmax(-1)
        policy = torch.tensor([
            vocabulary.atomic_policy(int(z_value)) for z_value in routed_z
        ])
        predictions = model(policy).argmax(-1)
    return float((predictions == targets).float().mean())


def multipass_record(model, update, train, ood, pairs, vocabulary):
    return {
        "update": int(update),
        "phase": "multipass_z",
        "observed": multipass_accuracy(model, train, vocabulary),
        "heldout": multipass_accuracy(model, ood, vocabulary),
        "patch_accuracy": multipass_patch_accuracy(model, pairs, vocabulary),
    }


def train_multipass(model, train, ood, pairs, vocabulary, retention, updates, seed):
    examples = FixedSizeEpochBatcher(train, 64, seed)
    retained = FixedSizeEpochBatcher(retention, 64, seed + 100003)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    history = [multipass_record(model, 0, train, ood, pairs, vocabulary)]
    for update in range(1, int(updates) + 1):
        current = examples.next()
        recognition = torch.tensor([
            vocabulary.atomic_recognition(example) for example in current
        ])
        z_targets = torch.tensor([example.z for example in current])
        optimizer.zero_grad()
        z_logits = model(recognition)[:, :LATENT_COUNT]
        routed_z = z_logits.argmax(-1)
        policy = torch.tensor([
            vocabulary.atomic_policy(int(z_value)) for z_value in routed_z
        ])
        y_targets = torch.tensor([example.target for example in current])
        loss = F.cross_entropy(z_logits, z_targets)
        loss = loss + F.cross_entropy(model(policy), y_targets)
        loss = loss + retention_loss(model, retained.next())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update in CHECKPOINTS:
            history.append(multipass_record(
                model, update, train, ood, pairs, vocabulary
            ))
    return history


def train_one_pass(model, method, train, ood, pairs, vocabulary, retention,
                   updates, seed):
    if method == "staged_removal" and int(updates) != sum(row[1] for row in PHASES):
        raise ValueError("finetune_updates must equal the staged schedule")
    examples = FixedSizeEpochBatcher(train, 64, seed)
    sources = FixedSizeEpochBatcher(train, 64, seed + 100003)
    retained = FixedSizeEpochBatcher(retention, 64, seed + 200003)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    phases = PHASES if method == "staged_removal" else ((
        method,
        updates,
        "graph" if method == "full_scaffold" else "dense",
        "hard" if method == "full_scaffold" else "raw",
        method != "regular_sft",
    ),)
    model.set_stage(phases[0][2], phases[0][3])
    history = [one_pass_record(
        model, 0, phases[0][0], train, ood, pairs, vocabulary
    )]
    completed = 0
    for phase, phase_updates, routing, interface, use_lz in phases:
        for offset in range(int(phase_updates)):
            alpha = (
                1.0 - float(offset + 1) / float(phase_updates)
                if interface == "blend" else 0.0
            )
            model.set_stage(routing, interface, alpha)
            current = examples.next()
            tokens, targets, z_targets = batch(current, vocabulary)
            optimizer.zero_grad()
            output, z_logits, _ = model(tokens, return_z=True)
            loss = F.cross_entropy(output, targets)
            if use_lz:
                loss = loss + F.cross_entropy(z_logits, z_targets)
            if method == "iit":
                source_examples = sources.next()
                source_tokens, source_targets, _ = batch(
                    source_examples, vocabulary
                )
                with torch.no_grad():
                    source_boundary = effective_boundary(model, source_tokens)
                loss = loss + F.cross_entropy(
                    model(tokens, z_patch=source_boundary), source_targets
                )
            loss = loss + retention_loss(model, retained.next())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            completed += 1
            if completed in CHECKPOINTS:
                history.append(one_pass_record(
                    model, completed, phase, train, ood, pairs, vocabulary
                ))
    if method == "staged_removal":
        model.set_stage("dense", "raw")
    return history


def run_seed(seed, atomic_updates=3000, probe_updates=1000,
             finetune_updates=6000, methods=METHODS):
    torch.manual_seed(int(seed))
    all_examples = build_data(seed)
    vocabulary = SafetyVocabulary()
    atomic = atomic_examples(all_examples, vocabulary)
    base = new_base_model(vocabulary)
    pretrain(base, atomic, atomic_updates, seed + 1)
    fit_z_head(base, all_examples, vocabulary, probe_updates, seed + 2)
    bank = policy_state_bank(base, vocabulary)
    retention = build_retention_records(base, atomic)
    train = tuple(row for row in all_examples if row.style in TRAIN_STYLES)
    ood = tuple(row for row in all_examples if row.style in OOD_STYLES)
    pairs = counterfactual_pairs(ood)
    results = {}
    for method in methods:
        model = make_model(base, vocabulary, bank)
        if method == "multipass_z":
            history = train_multipass(
                model, train, ood, pairs, vocabulary, retention,
                finetune_updates, seed + 10,
            )
            observed = multipass_accuracy(model, train, vocabulary)
            heldout = multipass_accuracy(model, ood, vocabulary)
            patch = multipass_patch_accuracy(model, pairs, vocabulary)
        else:
            history = train_one_pass(
                model, method, train, ood, pairs, vocabulary, retention,
                finetune_updates, seed + 10,
            )
            observed = evaluate(model, train, vocabulary)
            heldout = evaluate(model, ood, vocabulary)
            patch = patch_accuracy(model, pairs, vocabulary)
        results[method] = {
            "observed": observed,
            "heldout": heldout,
            "patch_accuracy": patch,
            "atomic_accuracy": atomic_accuracy(model, atomic),
            "history": history,
            "final_inference": (
                "two_forward_passes" if method == "multipass_z"
                else "one_pass_dense_raw_without_auxiliary_loss"
                if method == "staged_removal" else "one_forward_pass"
            ),
        }
    base_model = make_model(base, vocabulary, bank)
    return {
        "seed": int(seed),
        "base_atomic_accuracy": atomic_accuracy(base_model, atomic),
        "results": results,
    }


def run(output, seeds=(61001, 61002, 61003), methods=METHODS, **kwargs):
    torch.set_num_threads(1)
    if not verify_graph_paths():
        raise AssertionError("graph mask permits an input-to-output bypass")
    runs = [run_seed(seed, methods=methods, **kwargs) for seed in seeds]
    summary = {
        "task": "toy safety policy under held-out surface styles",
        "train_styles": ["direct", "role_play"],
        "ood_styles": ["cipher", "past_tense"],
        "methods": list(methods),
        "matched_controls": {
            "same_architecture": True,
            "same_atomic_checkpoint_within_seed": True,
            "same_128_training_examples": True,
            "same_128_ood_examples": True,
            "same_update_budget": int(kwargs.get("finetune_updates", 6000)),
            "same_atomic_teacher_retention_loss": True,
        },
        "staged_schedule": [list(row) for row in PHASES],
        "runs": runs,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="toy/safety/results/matched_methods")
    parser.add_argument("--seeds", type=int, nargs="+", default=[61001, 61002, 61003])
    parser.add_argument("--atomic-updates", type=int, default=3000)
    parser.add_argument("--probe-updates", type=int, default=1000)
    parser.add_argument("--finetune-updates", type=int, default=6000)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    args = parser.parse_args()
    run(
        args.output,
        args.seeds,
        methods=args.methods,
        atomic_updates=args.atomic_updates,
        probe_updates=args.probe_updates,
        finetune_updates=args.finetune_updates,
    )


if __name__ == "__main__":
    main()
