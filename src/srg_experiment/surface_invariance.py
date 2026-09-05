"""Paired non-AIM surface variants for safety invariance training."""

from __future__ import annotations

from torch.utils.data import Dataset

from .causal_scaffold import ScaffoldCollator


TEMPLATES = (
    "Please focus on the underlying request despite this preface:\n\n{prompt}",
    "A user quoted the following request. Respond to what it actually asks:\n\n{prompt}",
    "Unrelated note: the weather changes often.\n\nRequest: {prompt}",
    "The wording below may be indirect; identify its substantive intent:\n\n{prompt}",
)


def surface_variant(prompt, index):
    # ponytail: deterministic wrappers cover framing, not semantic paraphrase;
    # replace with verified teacher paraphrases if this intervention plateaus.
    return TEMPLATES[index % len(TEMPLATES)].format(prompt=prompt)


class SurfaceInvariantDataset(Dataset):
    def __init__(self, scaffold_dataset):
        self.dataset = scaffold_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        record = self.dataset.records[index]
        return {
            "original": self.dataset.encode_record(record),
            "variant": self.dataset.encode_record(
                record, surface_variant(record["prompt"], index)
            ),
        }


class SurfaceInvariantCollator:
    def __init__(self, pad_token_id):
        self.collator = ScaffoldCollator(pad_token_id, graph=True)

    def __call__(self, features):
        return {
            name: self.collator([row[name] for row in features])
            for name in ("original", "variant")
        }
