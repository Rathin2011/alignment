import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "toy/arithmetic/within_task"))

from matched_methods import (  # noqa: E402
    METHODS,
    PermutationTransformer,
    Vocabulary,
    graph_has_no_bypass,
)


class MatchedMethodsTest(unittest.TestCase):
    def test_six_declared_methods(self):
        self.assertEqual(len(METHODS), 6)

    def test_graph_has_no_input_bypass(self):
        self.assertTrue(graph_has_no_bypass())

    def test_every_interface_runs(self):
        vocabulary = Vocabulary()
        model = PermutationTransformer(vocabulary.size)
        model.set_canonical_bank(torch.randn(model.q, model.width))
        tokens = torch.tensor([[0, vocabulary.value_offset, 8, vocabulary.out_token]])
        for routing, interface in (
            ("dense", "raw"), ("graph", "hard"), ("dense", "hard"),
            ("dense", "soft"), ("dense", "blend"),
        ):
            model.set_stage(routing, interface, 0.5 if interface == "blend" else 0.0)
            output, z_logits, boundary = model(tokens, return_z=True)
            self.assertEqual(output.shape, (1, 16))
            self.assertEqual(z_logits.shape, (1, 16))
            self.assertEqual(boundary.shape, (1, model.width))


if __name__ == "__main__":
    unittest.main()
