"""Transformer and removable causal scaffold for the toy safety task."""

import numpy as np
import torch
import torch.nn.functional as F

from toy.arithmetic.within_task.core_task.transformer import (
    PreLayerNormBlock,
    attention_mask,
)
from toy.safety.core_task.task import LATENT_COUNT


Z_POSITION = 1
BOUNDARY_LAYER = 2


class FixedSizeEpochBatcher:
    def __init__(self, examples, batch_size, seed):
        self.examples = tuple(examples)
        self.batch_size = int(batch_size)
        self.rng = np.random.RandomState(int(seed))
        self.order = np.empty(0, dtype=np.int64)
        self.cursor = 0

    def next(self):
        selected = []
        while len(selected) < self.batch_size:
            if self.cursor >= len(self.order):
                self.order = self.rng.permutation(len(self.examples))
                self.cursor = 0
            count = min(
                self.batch_size - len(selected), len(self.order) - self.cursor
            )
            selected.extend(
                self.examples[index]
                for index in self.order[self.cursor:self.cursor + count]
            )
            self.cursor += count
        return selected


class NoSlotTransformer(torch.nn.Module):
    """Dense Transformer whose existing second token carries Z."""

    def __init__(self, vocabulary_size, output_classes, z_classes,
                 layers=4, heads=4, width=64, mlp_width=128):
        super().__init__()
        self.width = int(width)
        self.token_embeddings = torch.nn.Embedding(vocabulary_size, width)
        self.position_embeddings = torch.nn.Parameter(torch.empty(4, width))
        self.blocks = torch.nn.ModuleList([
            PreLayerNormBlock(width, heads, mlp_width) for _ in range(layers)
        ])
        self.final_norm = torch.nn.LayerNorm(width)
        self.output_head = torch.nn.Linear(width, output_classes)
        self.z_norm = torch.nn.LayerNorm(width)
        self.z_head = torch.nn.Linear(width, z_classes)
        self.register_buffer("canonical_bank", torch.empty(0, width), persistent=True)
        torch.nn.init.normal_(self.token_embeddings.weight, 0.0, 0.02)
        torch.nn.init.normal_(self.position_embeddings, 0.0, 0.02)

    def enable_canonicalization(self, bank):
        expected = (self.z_head.out_features, self.width)
        if tuple(bank.shape) != expected:
            raise ValueError("canonical bank must have shape {}".format(expected))
        self.canonical_bank = bank.detach().clone()

    def forward(self, tokens, z_patch=None, return_z=False):
        if tokens.ndim != 2 or tokens.shape[1] != 4:
            raise ValueError("tokens must have shape [batch, 4]")
        residual = self.token_embeddings(tokens) + self.position_embeddings.unsqueeze(0)
        mask = attention_mask(4, "dense", tokens.device)
        z_state = z_logits = None
        for layer_index, block in enumerate(self.blocks):
            if layer_index == BOUNDARY_LAYER:
                z_state = residual[:, Z_POSITION, :]
                z_logits = self.z_head(self.z_norm(z_state))
                if z_patch is not None:
                    residual = residual.clone()
                    residual[:, Z_POSITION, :] = z_patch
            residual = block(residual, mask)
        output = self.output_head(self.final_norm(residual[:, -1, :]))
        return (output, z_logits, z_state) if return_z else output


def graph_mask(phase, device=None):
    allowed = (
        ((0,), (0, 1), (2,), (2, 3))
        if phase == "pre"
        else ((0,), (1,), (1, 2), (1, 2, 3))
        if phase == "post"
        else None
    )
    if allowed is None:
        raise ValueError("phase must be pre or post")
    mask = torch.ones(4, 4, dtype=torch.bool, device=device)
    for query, keys in enumerate(allowed):
        mask[query, list(keys)] = False
    return mask


def verify_graph_paths(layers=4):
    adjacency = {}
    for layer in range(int(layers)):
        mask = graph_mask("pre" if layer < BOUNDARY_LAYER else "post")
        for source in range(4):
            adjacency.setdefault((layer, source), []).append((layer + 1, source))
        for query in range(4):
            for key in range(4):
                if not mask[query, key]:
                    adjacency.setdefault((layer, key), []).append((layer + 1, query))
    for source in (0, 1):
        queue, visited = [(0, source)], {(0, source)}
        while queue:
            node = queue.pop(0)
            if node == (layers, 3):
                return False
            for neighbor in adjacency.get(node, ()):
                if neighbor != (BOUNDARY_LAYER, Z_POSITION) and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return True


class StagedSafetyTransformer(NoSlotTransformer):
    """One model supporting graph/dense routing and canonical/raw Z."""

    def __init__(self, vocabulary_size, output_classes, z_classes):
        super().__init__(vocabulary_size, output_classes, z_classes)
        self.routing = "dense"
        self.interface = "raw"
        self.blend_alpha = 0.0

    def set_stage(self, routing, interface, blend_alpha=0.0):
        if routing not in ("dense", "graph"):
            raise ValueError("routing must be dense or graph")
        if interface not in ("raw", "hard", "soft", "blend"):
            raise ValueError("unknown interface")
        if interface != "raw" and self.canonical_bank.numel() == 0:
            raise ValueError("canonical interface requires a state bank")
        self.routing, self.interface = routing, interface
        self.blend_alpha = float(blend_alpha)

    def forward(self, tokens, z_patch=None, return_z=False):
        if tokens.ndim != 2 or tokens.shape[1] != 4:
            raise ValueError("tokens must have shape [batch, 4]")
        residual = self.token_embeddings(tokens) + self.position_embeddings.unsqueeze(0)
        dense = attention_mask(4, "dense", tokens.device)
        pre, post = graph_mask("pre", tokens.device), graph_mask("post", tokens.device)
        z_state = z_logits = None
        for layer_index, block in enumerate(self.blocks):
            if layer_index == BOUNDARY_LAYER:
                z_state = residual[:, Z_POSITION, :]
                z_logits = self.z_head(self.z_norm(z_state))
                probabilities = z_logits.softmax(-1)
                boundary = z_state
                if self.interface == "hard":
                    ids = probabilities.argmax(-1)
                    one_hot = F.one_hot(ids, LATENT_COUNT).to(probabilities.dtype)
                    boundary = (one_hot + probabilities - probabilities.detach()) @ self.canonical_bank
                elif self.interface == "soft":
                    boundary = probabilities @ self.canonical_bank
                elif self.interface == "blend":
                    canonical = probabilities @ self.canonical_bank
                    boundary = self.blend_alpha * canonical + (1.0 - self.blend_alpha) * z_state
                if z_patch is not None:
                    boundary = z_patch
                residual = residual.clone()
                residual[:, Z_POSITION, :] = boundary
            mask = dense if self.routing == "dense" else (
                pre if layer_index < BOUNDARY_LAYER else post
            )
            residual = block(residual, mask)
        output = self.output_head(self.final_norm(residual[:, -1, :]))
        return (output, z_logits, z_state) if return_z else output
