import unittest

import torch

from toy.safety.core_task.model import StagedSafetyTransformer, verify_graph_paths
from toy.safety.core_task.task import ACTION_COUNT, LATENT_COUNT, SafetyVocabulary
from toy.safety.matched_methods import METHODS


class SafetyMatchedMethodsTest(unittest.TestCase):
    def test_six_declared_methods(self):
        self.assertEqual(len(METHODS), 6)

    def test_graph_has_no_input_bypass(self):
        self.assertTrue(verify_graph_paths())

    def test_every_interface_runs(self):
        vocabulary = SafetyVocabulary()
        model = StagedSafetyTransformer(
            vocabulary.size, LATENT_COUNT + ACTION_COUNT, LATENT_COUNT
        )
        model.enable_canonicalization(torch.randn(LATENT_COUNT, model.width))
        tokens = torch.tensor([[
            0, vocabulary.surface_token(0, 0),
            vocabulary.policy_token, vocabulary.out_token,
        ]])
        for routing, interface in (
            ("dense", "raw"), ("graph", "hard"), ("dense", "hard"),
            ("dense", "soft"), ("dense", "blend"),
        ):
            model.set_stage(
                routing, interface, 0.5 if interface == "blend" else 0.0
            )
            output, z_logits, z_state = model(tokens, return_z=True)
            self.assertEqual(output.shape, (1, LATENT_COUNT + ACTION_COUNT))
            self.assertEqual(z_logits.shape, (1, LATENT_COUNT))
            self.assertEqual(z_state.shape, (1, model.width))


if __name__ == "__main__":
    unittest.main()
