import unittest

from causal.evaluate_learned_multipass import flip_z, format_z, parse_decision
from causal.evaluate_oracle_enforced_path import REFUSAL, enforce_oracle_path
from causal.prepare_multipass_analyzer import build as build_balanced
from causal.prepare_learned_multipass import build, without_policy


class QTest(unittest.TestCase):
    def test_balanced_analyzer_data(self):
        records = [
            {
                "id": "harm", "prompt": "harm", "kind": "harmful",
                "response": "refuse", "intent": "harmful",
                "harm_types": ["violence"], "harm_severity": 3,
                "actionability": 2, "policy": "refuse",
            }
        ] + [
            {
                "id": str(index), "prompt": "safe", "kind": "benign",
                "response": "answer", "intent": "benign",
                "harm_types": [], "harm_severity": 0,
                "actionability": 0, "policy": "comply",
            }
            for index in range(2)
        ]
        examples = build_balanced(records, 1)
        self.assertEqual(len(examples), 4)
        self.assertEqual(sum(row["kind"] == "harmful" for row in examples), 2)
        self.assertEqual(sum(row["kind"] == "benign" for row in examples), 2)

    def test_learned_multipass_interface(self):
        row = {
            "id": "one", "prompt": "request", "kind": "harmful",
            "response": "<intent>harmful</intent>\n"
                        "<harm_types>fraud</harm_types>\n"
                        "<severity>2</severity>\n"
                        "<actionability>1</actionability>\n"
                        "<policy>refuse</policy>",
        }
        examples = build([row], 1)
        self.assertEqual(
            {item["stage"] for item in examples},
            {"x_to_z", "z_to_decision"},
        )
        self.assertTrue(all("<policy>" not in item["prompt"] for item in examples))
        self.assertTrue(without_policy(row["response"]).endswith("</actionability>"))
        self.assertEqual(parse_decision("REFUSE"), "REFUSE")
        flipped = flip_z({"intent": "harmful"})
        self.assertEqual(flipped["intent"], "benign")
        self.assertIn("<intent>benign</intent>", format_z(flipped))

    def test_oracle_path_is_non_bypassable(self):
        source = [
            {"id": "h", "kind": "harmful", "response": "unsafe"},
            {"id": "b", "kind": "benign", "response": "helpful"},
        ]
        routed = enforce_oracle_path(source)
        self.assertEqual(routed[0]["response"], REFUSAL)
        self.assertEqual(routed[0]["policy"], "refuse")
        self.assertEqual(routed[1]["response"], "helpful")
        self.assertEqual(routed[1]["policy"], "comply")
        self.assertEqual(source[0]["response"], "unsafe")


if __name__ == "__main__":
    unittest.main()
