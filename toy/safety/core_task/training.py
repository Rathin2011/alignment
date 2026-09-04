"""Shared pretraining and retention utilities for toy safety experiments."""

import torch
import torch.nn.functional as F

from toy.safety.core_task.model import FixedSizeEpochBatcher, NoSlotTransformer, StagedSafetyTransformer
from toy.safety.core_task.task import ACTION_COUNT, LATENT_COUNT


def pretrain(model, examples, updates, seed, learning_rate=3e-4):
    batcher = FixedSizeEpochBatcher(examples, 64, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    for _ in range(int(updates)):
        current = batcher.next()
        tokens = torch.tensor([item[0] for item in current])
        targets = torch.tensor([item[1] for item in current])
        optimizer.zero_grad()
        loss = F.cross_entropy(model(tokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def fit_z_head(model, examples, vocabulary, updates, seed):
    records = [
        (vocabulary.atomic_recognition(example), int(example.z))
        for example in examples
    ]
    batcher = FixedSizeEpochBatcher(records, 64, seed)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters = list(model.z_norm.parameters()) + list(model.z_head.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=1e-3)
    for _ in range(int(updates)):
        current = batcher.next()
        tokens = torch.tensor([item[0] for item in current])
        targets = torch.tensor([item[1] for item in current])
        optimizer.zero_grad()
        _, logits, _ = model(tokens, return_z=True)
        F.cross_entropy(logits, targets).backward()
        optimizer.step()
    for parameter in model.parameters():
        parameter.requires_grad_(True)


def policy_state_bank(model, vocabulary):
    tokens = torch.tensor([
        vocabulary.atomic_policy(z) for z in range(LATENT_COUNT)
    ])
    model.eval()
    with torch.no_grad():
        _, _, states = model(tokens, return_z=True)
    return states.detach().clone()


def build_retention_records(model, atomic):
    tokens = torch.tensor([item[0] for item in atomic])
    model.eval()
    with torch.no_grad():
        logits = model(tokens)
    return tuple(
        (list(item[0]), logits[index].detach().clone())
        for index, item in enumerate(atomic)
    )


def make_model(base, vocabulary, bank):
    model = StagedSafetyTransformer(
        vocabulary.size,
        output_classes=LATENT_COUNT + ACTION_COUNT,
        z_classes=LATENT_COUNT,
    )
    model.load_state_dict(base.state_dict())
    model.enable_canonicalization(bank)
    model.set_stage("dense", "raw")
    return model


def retention_loss(model, records):
    tokens = torch.tensor([item[0] for item in records])
    teacher = torch.stack([item[1] for item in records])
    old = (model.routing, model.interface, model.blend_alpha)
    model.set_stage("dense", "raw")
    student = model(tokens)
    model.set_stage(*old)
    return F.kl_div(
        student.log_softmax(-1), teacher.softmax(-1), reduction="batchmean"
    )


def new_base_model(vocabulary):
    return NoSlotTransformer(
        vocabulary.size,
        output_classes=LATENT_COUNT + ACTION_COUNT,
        z_classes=LATENT_COUNT,
    )
