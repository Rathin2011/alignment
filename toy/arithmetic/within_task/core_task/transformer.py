"""Stage 3 dense causal Transformer with an optional designated [Z] slot."""

from __future__ import absolute_import

import math

import torch
import torch.nn.functional as functional


class Vocabulary(object):
    def __init__(self, m=8, n=8, q=16):
        self.m = int(m)
        self.n = int(n)
        self.q = int(q)
        self.a_offset = 0
        self.b_offset = self.a_offset + self.m
        self.value_offset = self.b_offset + self.n
        self.z_token = self.value_offset + self.q
        self.out_token = self.z_token + 1
        self.size = self.out_token + 1

    def encode(self, a_index, b_index, x, include_z=True):
        tokens = [
            self.a_offset + int(a_index),
            self.value_offset + int(x),
        ]
        if include_z:
            tokens.append(self.z_token)
        tokens.extend([self.b_offset + int(b_index), self.out_token])
        return tokens


class PreLayerNormBlock(torch.nn.Module):
    def __init__(self, width=128, heads=4, mlp_width=512):
        super(PreLayerNormBlock, self).__init__()
        self.attention_norm = torch.nn.LayerNorm(width)
        self.attention = torch.nn.MultiheadAttention(
            width, heads, dropout=0.0, batch_first=True
        )
        self.mlp_norm = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(width, mlp_width),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_width, width),
        )

    def forward(self, residual, causal_mask):
        normalized = self.attention_norm(residual)
        attended, _ = self.attention(
            normalized, normalized, normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        residual = residual + attended
        residual = residual + self.mlp(self.mlp_norm(residual))
        return residual


def attention_mask(sequence_length, mode="dense", device=None):
    if mode == "dense":
        return torch.triu(
            torch.ones(
                sequence_length, sequence_length,
                dtype=torch.bool, device=device,
            ),
            diagonal=1,
        )
    if mode != "graph_cut" or sequence_length != 5:
        raise ValueError("graph_cut attention requires the five-position [Z] sequence")
    allowed = (
        (0,),
        (0, 1),
        (0, 1, 2),
        (2, 3),
        (2, 3, 4),
    )
    mask = torch.ones(5, 5, dtype=torch.bool, device=device)
    for query, keys in enumerate(allowed):
        mask[query, list(keys)] = False
    return mask


def verify_graph_cut_paths(layers=4):
    """Verify all layer-expanded A/x -> OUT paths contain a Z state."""

    mask = attention_mask(5, mode="graph_cut").cpu().numpy()
    adjacency = {}
    for layer in range(layers):
        for source in range(5):
            adjacency.setdefault((layer, source), []).append((layer + 1, source))
        for query in range(5):
            for key in range(5):
                if not mask[query, key]:
                    adjacency.setdefault((layer, key), []).append((layer + 1, query))
    target = (layers, 4)
    for source_position in (0, 1):
        start = (0, source_position)
        queue = [start]
        visited = set([start])
        while queue:
            node = queue.pop(0)
            if node == target:
                return False
            for neighbor in adjacency.get(node, ()):
                if neighbor[1] == 2:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return True


class DenseTransformer(torch.nn.Module):
    def __init__(self, vocabulary_size, sequence_length, include_z=True,
                 layers=4, heads=4, width=128, mlp_width=512,
                 output_classes=16, attention_mode="dense"):
        super(DenseTransformer, self).__init__()
        self.include_z = bool(include_z)
        self.sequence_length = int(sequence_length)
        self.width = int(width)
        self.attention_mode = str(attention_mode)
        if self.attention_mode == "graph_cut" and not self.include_z:
            raise ValueError("graph_cut attention requires include_z=True")
        self.token_embeddings = torch.nn.Embedding(vocabulary_size, width)
        self.position_embeddings = torch.nn.Parameter(
            torch.empty(sequence_length, width)
        )
        self.blocks = torch.nn.ModuleList([
            PreLayerNormBlock(width, heads, mlp_width) for _ in range(layers)
        ])
        self.final_norm = torch.nn.LayerNorm(width)
        self.output_head = torch.nn.Linear(width, output_classes, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)
        # Blocks retain PyTorch's standard initialization. The output head is
        # explicitly untied from the token embedding table.
        torch.nn.init.normal_(self.output_head.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.output_head.bias)

    def forward(self, tokens, return_z_activations=False,
                z_patch=None, z_patch_layer=None):
        residual = self.token_embeddings(tokens) + self.position_embeddings.unsqueeze(0)
        causal_mask = attention_mask(
            self.sequence_length, mode=self.attention_mode,
            device=tokens.device,
        )
        z_activations = []
        z_position = 2 if self.include_z else None
        if (z_patch is None) != (z_patch_layer is None):
            raise ValueError("z_patch and z_patch_layer must be supplied together")
        if z_patch is not None:
            if not self.include_z:
                raise ValueError("Z patching requires include_z=True")
            if int(z_patch_layer) < 0 or int(z_patch_layer) >= len(self.blocks):
                raise ValueError("Z patch layer must be a block-input index")
            if tuple(z_patch.shape) != (tokens.shape[0], self.width):
                raise ValueError("Z patch must have shape [batch, width]")
        for layer_index, block in enumerate(self.blocks):
            if return_z_activations and self.include_z:
                z_activations.append(residual[:, z_position, :])
            if z_patch is not None and layer_index == int(z_patch_layer):
                residual = residual.clone()
                residual[:, z_position, :] = z_patch
            residual = block(residual, causal_mask)
        logits = self.output_head(self.final_norm(residual[:, -1, :]))
        if return_z_activations:
            if self.include_z:
                return logits, torch.stack(z_activations, dim=1)
            return logits, None
        return logits


def learning_rate_at(update, warmup_updates=500, total_updates=100000,
                     peak=3e-4, minimum=3e-5):
    if update <= 0:
        return 0.0
    if update <= warmup_updates:
        return peak * float(update) / float(warmup_updates)
    if update >= total_updates:
        return minimum
    fraction = float(update - warmup_updates) / float(total_updates - warmup_updates)
    cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
    return minimum + (peak - minimum) * cosine


def parameter_count(model):
    return int(sum(parameter.numel() for parameter in model.parameters()))


def batch_tokens(examples, vocabulary, include_z):
    tokens = torch.tensor([
        vocabulary.encode(
            example.a_index, example.b_index, example.x, include_z=include_z
        )
        for example in examples
    ], dtype=torch.long)
    targets = torch.tensor([example.y for example in examples], dtype=torch.long)
    return tokens, targets


def loss_for_examples(model, examples, vocabulary, include_z):
    tokens, targets = batch_tokens(examples, vocabulary, include_z)
    return functional.cross_entropy(model(tokens), targets)
